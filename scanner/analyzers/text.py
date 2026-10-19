"""Model-free text checks: wording, rhythm, formatting and hidden characters.

These are weak on their own. The trained detectors in ml.py and the paid APIs in
apis.py are the stronger text signals; these explain *why* text reads as AI.
"""

from __future__ import annotations

import re
import statistics
import unicodedata

from ..core import MODERATE, TEXT, WEAK, Item, Signal, SkipAnalyzer, analyzer, clamp

MIN_WORDS = 40

# Words and phrases that chat models overuse relative to human writing.
AI_PHRASES = [
    "delve", "delves", "delving", "tapestry", "testament to", "it's important to note", "it is important to note",
    "it's worth noting", "it is worth noting", "in conclusion", "in summary", "overall,", "furthermore", "moreover",
    "additionally,", "navigate the", "navigating the", "complexities of", "realm of", "in the realm", "multifaceted",
    "intricate", "intricacies", "pivotal", "foster", "fostering", "underscore", "underscores", "seamless", "seamlessly",
    "leverage", "leveraging", "robust", "showcase", "showcasing", "embark", "vibrant", "nuanced", "holistic",
    "ever-evolving", "ever-changing", "fast-paced world", "in today's", "game-changer", "game changer", "unlock the",
    "unleash", "elevate", "landscape of", "the landscape", "crucial", "plays a crucial role", "plays a vital role",
    "a myriad of", "myriad", "paramount", "meticulous", "meticulously", "commendable", "invaluable", "resonate",
    "resonates", "bustling", "beacon", "symphony", "embrace", "harness", "spearhead", "streamline", "cutting-edge",
    "groundbreaking", "transformative", "a deep dive", "dive into", "let's dive", "when it comes to",
    "whether you're", "not only", "but also", "serves as", "stands as", "a rich", "rich history", "nestled",
    "boasts", "captivating", "enhance", "enhancing", "notably", "comprehensive", "ensuring", "aligns with",
]
ASSISTANT_BOILERPLATE = [
    "as an ai", "as a language model", "i hope this helps", "i hope this email finds you well", "certainly!",
    "great question", "feel free to", "let me know if you", "here's a", "here is a", "i'd be happy to",
    "i'm sorry, but", "happy to help", "knowledge cutoff", "i cannot browse", "regenerate response",
]
HUMAN_MARKERS = re.compile(
    r"(\blol\b|\blmao\b|\bidk\b|\btbh\b|\bimo\b|\bgonna\b|\bwanna\b|\bkinda\b|\bya\b|\bu\b|!!+|\?\?+|\.\.\.\s*[a-z]|"
    r"(?:^|[.!?]\s+)i\s|\bi'm\b(?=\s+[a-z]+\s+lol)|:\)|:\(|\bhaha\b)", re.I | re.M)
BOM = "\ufeff"
HIDDEN_CHARS = {
    "\u200b": "zero-width space", "\u200c": "zero-width non-joiner", "\u200d": "zero-width joiner",
    "\u2060": "word joiner", "\ufeff": "byte-order mark", "\u202f": "narrow no-break space",
    "\u00ad": "soft hyphen", "\u2062": "invisible times", "\u2063": "invisible separator",
}


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", text)


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'(])|\n{2,}", text)
    return [p.strip() for p in parts if len(words(p)) >= 3]


def _need_words(item: Item, n: int = MIN_WORDS) -> list[str]:
    w = words(item.text or "")
    if len(w) < n:
        raise SkipAnalyzer(f"needs at least {n} words (got {len(w)})")
    return w


@analyzer("text_wording", (TEXT,), "Counts words and phrases chat models overuse, and assistant boilerplate.")
def wording(item: Item) -> list[Signal]:
    w = _need_words(item)
    low = " " + re.sub(r"\s+", " ", item.text.lower()) + " "
    hits: dict[str, int] = {}
    for p in AI_PHRASES:
        n = len(re.findall(r"(?<![a-z])" + re.escape(p) + r"(?![a-z])", low))
        if n:
            hits[p] = n
    per_k = 1000 * sum(hits.values()) / len(w)
    # Human prose typically sits around 2-6 per 1000 words on this list; chat output often 15+.
    score = clamp(0.5 + (per_k - 7) / 30, 0.2, 0.9)
    out = [Signal("text_wording", "AI-typical vocabulary", score, WEAK,
                  f"{sum(hits.values())} AI-typical words or phrases ({per_k:.1f} per 1,000 words). "
                  f"Most frequent: {', '.join(sorted(hits, key=hits.get, reverse=True)[:6]) or 'none'}.",
                  {"per_1000_words": round(per_k, 1), "hits": hits, "highlights": sorted(hits)})]
    boiler = [b for b in ASSISTANT_BOILERPLATE if b in low]
    if boiler:
        out.append(Signal("text_wording", "Chatbot boilerplate", 0.85 if len(boiler) > 1 else 0.75, MODERATE,
                          f"Contains phrases typical of chatbot replies: {', '.join(boiler)}.", {"highlights": boiler}))
    return out


@analyzer("text_rhythm", (TEXT,), "Measures how much sentence length and vocabulary vary (burstiness).")
def rhythm(item: Item) -> list[Signal]:
    w = _need_words(item, 80)
    sents = sentences(item.text)
    if len(sents) < 5:
        raise SkipAnalyzer("needs at least 5 sentences")
    lengths = [len(words(s)) for s in sents]
    mean = statistics.mean(lengths)
    cv = statistics.pstdev(lengths) / mean if mean else 0
    # Human writing usually has CV of 0.55+; model output tends to be more even (0.3-0.45).
    burst_score = clamp(0.5 + (0.5 - cv) * 1.2, 0.2, 0.8)
    out = [Signal("text_rhythm", "Sentence-length variation", burst_score, WEAK,
                  f"Sentence lengths vary by {cv:.2f} (coefficient of variation, mean {mean:.0f} words). "
                  "Human writing tends to swing more between short and long sentences.",
                  {"cv": round(cv, 3), "mean_len": round(mean, 1), "n_sentences": len(sents)})]
    # Moving-average type-token ratio over 100-word windows, so length doesn't skew it.
    lw = [x.lower() for x in w]
    win = 100
    if len(lw) >= win:
        ttrs = [len(set(lw[i:i + win])) / win for i in range(0, len(lw) - win + 1, 25)]
        mattr = statistics.mean(ttrs)
        out.append(Signal("text_rhythm", "Vocabulary variety", clamp(0.5 + (0.66 - mattr) * 2, 0.3, 0.7), WEAK,
                          f"Average unique-word ratio per 100 words is {mattr:.2f}.", {"mattr": round(mattr, 3)}))
    return out


@analyzer("text_format", (TEXT,), "Looks at punctuation, markdown structure, informality and typos.")
def formatting(item: Item) -> list[Signal]:
    w = _need_words(item)
    t = item.text
    n = len(w)
    em = t.count("—")
    bold_heads = len(re.findall(r"^\s*(?:[-*•]|\d+\.)\s+\*\*[^*]+\*\*:?", t, re.M)) + len(re.findall(r"^#{1,4} ", t, re.M))
    bullets = len(re.findall(r"^\s*(?:[-*•]|\d+\.)\s+", t, re.M))
    human = HUMAN_MARKERS.findall(t)
    lower_starts = len(re.findall(r"(?:^|[.!?]\s+)[a-z]", t))
    out = []
    em_per_k = 1000 * em / n
    if em_per_k > 4:
        out.append(Signal("text_format", "Heavy em-dash use", clamp(0.55 + em_per_k / 60, 0, 0.75), WEAK,
                          f"{em} em-dashes ({em_per_k:.1f} per 1,000 words). Chat models use them a lot; so do some writers."))
    if bold_heads >= 3:
        out.append(Signal("text_format", "Chatbot-style structure", 0.65, WEAK,
                          f"{bold_heads} bolded list headings or markdown headers, a common chatbot layout.",
                          {"bullets": bullets}))
    if len(human) + lower_starts >= 3:
        out.append(Signal("text_format", "Casual, unpolished writing", clamp(0.4 - 0.02 * (len(human) + lower_starts), 0.2, 0.4),
                          WEAK, f"Found {len(human) + lower_starts} informal markers (slang, lowercase sentence starts, "
                          "repeated punctuation). Models rarely write like this unless asked to.",
                          {"examples": [h if isinstance(h, str) else h[0] for h in human[:8]]}))
    if not out:
        out.append(Signal("text_format", "Neutral formatting", 0.5, WEAK, "Nothing distinctive in punctuation or layout."))
    return out


@analyzer("text_hidden_chars", (TEXT,), "Finds invisible or unusual Unicode characters some AI tools leave behind.")
def hidden_chars(item: Item) -> list[Signal]:
    t = item.text or ""
    if not t.strip():
        raise SkipAnalyzer("empty text")
    found = {name: t.count(ch) for ch, name in HIDDEN_CHARS.items() if ch in t}
    if t.startswith(BOM) and t.count(BOM) == 1:
        found.pop("byte-order mark", None)  # a leading BOM is just an encoding detail
    has_latin = re.search(r"[A-Za-z]", t) is not None
    homoglyphs = has_latin and sum(1 for ch in t if ord(ch) > 0x370
                                   and unicodedata.name(ch, "").startswith(("CYRILLIC", "GREEK")))
    if found:
        return [Signal("text_hidden_chars", "Invisible characters", 0.7, MODERATE,
                       f"Contains invisible characters ({', '.join(f'{v}× {k}' for k, v in found.items())}). Some AI tools "
                       "emit these (narrow no-break spaces especially), and some text watermarkers use them.", found)]
    if homoglyphs > 3 and homoglyphs < len(t) * 0.05:
        return [Signal("text_hidden_chars", "Look-alike letters", 0.6, WEAK,
                       f"{homoglyphs} Cyrillic/Greek letters mixed into Latin text, a trick used to dodge detectors.")]
    return [Signal("text_hidden_chars", "No hidden characters", 0.5, WEAK, "No invisible or look-alike characters found.")]
