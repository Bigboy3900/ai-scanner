"""Writing-style detector: a small trained model over function words and sentence statistics.

Chat models write in a recognisable "house style" even when they avoid buzzwords:
even sentence lengths, many commas and connective words, few contractions or
first-person asides. This model measures those habits. It is trained by
tools/train_text_style.py and stored as plain numbers in models/text_style.json,
so it runs anywhere numpy does, including the browser version.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..core import MODERATE, TEXT, Item, Signal, SkipAnalyzer, analyzer, clamp

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "text_style.json"
WINDOW = 200  # words per scored chunk
MIN_WORDS = 120

FUNCTION_WORDS = """
a about above across after afterwards again against all almost alone along already also although always am among
amongst an and another any anyhow anyone anything anyway anywhere are around as at back be became because become
becomes been before beforehand behind being below beside besides between beyond both but by can cannot could did do
does doing done down due during each either else elsewhere enough even ever every everyone everything everywhere
except few first for former formerly from further had has have having he hence her here hereby herein hers herself
him himself his how however i if in indeed instead into is it its itself just last latter least less many may me
meanwhile might mine more moreover most mostly much must my myself namely neither never nevertheless next no nobody
none nor not nothing now nowhere of off often on once one only onto or other others otherwise our ours ourselves out
over own per perhaps quite rather really same seem seemed seems several she should since so some somehow someone
something sometimes somewhat still such than that the their theirs them themselves then thence there thereafter
thereby therefore these they this those though through throughout thus to together too toward towards under until up
upon us very via was we well were what whatever when whence whenever where whereas whereby wherever whether which
while who whoever whole whom whose why will with within without would yet you your yours yourself yourselves
s t ll ve re d m don didn doesn isn wasn won can't oh yeah ok okay like got get gets getting thing things stuff lot
""".split()

# Connective and "hedging" words chat models lean on. Topic words (legacy, impact...) are
# deliberately left out so the model judges style, not subject.
STYLE_WORDS = """
additionally ultimately crucial essential significant significantly various numerous key particularly notably
overall ensure ensuring enhance provide provides providing allows allowing however furthermore moreover
""".split()

VOCAB = sorted(set(FUNCTION_WORDS + STYLE_WORDS) - {"can't"})
STAT_NAMES = [
    "sent_len_mean", "sent_len_cv", "sent_len_max_ratio", "word_len_mean", "long_word_frac", "ttr",
    "comma", "semicolon", "colon", "dash", "paren", "question", "exclaim", "quote", "digit", "contraction",
    "para_len_cv",
]
FEATURE_NAMES = [f"w:{w}" for w in VOCAB] + STAT_NAMES
_VOCAB_INDEX = {w: i for i, w in enumerate(VOCAB)}


def normalize(text: str) -> str:
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = text.replace("``", '"').replace("''", '"')
    text = re.sub(r"[*#_]+", " ", text)  # markdown emphasis and headings
    return text


def features(text: str) -> np.ndarray:
    """Feature vector for one chunk of text. Robust to casing and to tokenized corpora."""
    t = normalize(text)
    low = t.lower()
    words = re.findall(r"[a-z]+", low)
    n = max(len(words), 1)
    counts = np.zeros(len(VOCAB))
    for w in words:
        i = _VOCAB_INDEX.get(w)
        if i is not None:
            counts[i] += 1
    freq = counts / n * 100

    sents = [s for s in re.split(r"[.!?]+(?:\s|$)|\n\s*\n", t) if len(re.findall(r"[A-Za-z]+", s)) >= 2]
    lens = np.array([len(re.findall(r"[A-Za-z]+", s)) for s in sents] or [n], dtype=float)
    paras = [p for p in re.split(r"\n\s*\n", t) if p.strip()]
    plens = np.array([len(re.findall(r"[A-Za-z]+", p)) for p in paras] or [n], dtype=float)
    wl = np.array([len(w) for w in words] or [0], dtype=float)
    per_word = lambda pat: len(re.findall(pat, t)) / n * 100  # noqa: E731
    stats = [
        lens.mean(), lens.std() / max(lens.mean(), 1), lens.max() / max(lens.mean(), 1), wl.mean(),
        float((wl >= 8).mean()), len(set(words)) / n,
        per_word(r","), per_word(r";"), per_word(r":"), per_word(r"--|—|–| - "), per_word(r"\("),
        per_word(r"\?"), per_word(r"!"), per_word(r'"'), per_word(r"\d+"),
        sum(counts[_VOCAB_INDEX[c]] for c in ("s", "t", "ll", "ve", "re", "d", "m")) / n * 100,
        plens.std() / max(plens.mean(), 1) if len(plens) > 1 else 0.0,
    ]
    return np.concatenate([freq, np.array(stats, dtype=float)])


def chunks(text: str, size: int = WINDOW) -> list[str]:
    """Split into consecutive ~size-word chunks, keeping the original punctuation and spacing."""
    spans = [m.start() for m in re.finditer(r"\S+", text)]
    out = []
    for i in range(0, len(spans), size):
        if len(spans) - i < size * 0.6 and out:
            break  # a short tail is noise; it's covered by the previous chunk
        start = spans[i]
        end = spans[i + size] if i + size < len(spans) else len(text)
        out.append(text[start:end])
    return out


@lru_cache(maxsize=1)
def load_model() -> dict | None:
    if not MODEL_PATH.exists():
        return None
    m = json.loads(MODEL_PATH.read_text())
    for k in ("mean", "scale", "coef"):
        m[k] = np.array(m[k], dtype=float)
    return m


def chunk_probabilities(text: str) -> list[float]:
    m = load_model()
    out = []
    for c in chunks(text):
        z = (features(c) - m["mean"]) / m["scale"]
        logit = float(z @ m["coef"] + m["intercept"])
        out.append(1 / (1 + np.exp(-logit)))
    return out


def top_reasons(text: str, k: int = 4) -> list[str]:
    """The features pushing this text most towards AI, in plain words."""
    m = load_model()
    z = (np.mean([features(c) for c in chunks(text)], axis=0) - m["mean"]) / m["scale"]
    contrib = z * m["coef"]
    labels = m.get("labels", {})
    # Only name features a person can recognise: punctuation, rhythm and the connective words.
    readable = [i for i, f in enumerate(FEATURE_NAMES) if not f.startswith("w:") or f[2:] in STYLE_WORDS]
    order = sorted(readable, key=lambda i: -contrib[i])[:k]
    return [labels.get(FEATURE_NAMES[i], FEATURE_NAMES[i]) for i in order if contrib[i] > 0.15]


@analyzer("text_style_model", (TEXT,), "Trained writing-style model (function words, sentence rhythm, punctuation habits).")
def style_model(item: Item) -> list[Signal]:
    text = item.text or ""
    n = len(re.findall(r"\S+", text))
    if n < MIN_WORDS:
        raise SkipAnalyzer(f"needs at least {MIN_WORDS} words (got {n})")
    if load_model() is None:
        raise SkipAnalyzer("style model file missing (run tools/train_text_style.py)")
    probs = chunk_probabilities(text)
    m = load_model()
    # Average in log-odds so one odd paragraph doesn't swing the result.
    logits = [np.log(p / (1 - p)) for p in np.clip(probs, 1e-4, 1 - 1e-4)]
    p = float(1 / (1 + np.exp(-np.mean(logits))))
    reasons = top_reasons(text)
    share = sum(q > 0.5 for q in probs)
    # Floor at 0.2: a style model trained on modest data may argue for AI, but shouldn't vouch strongly for a human.
    return [Signal("text_style_model", "Writing-style model", clamp(p, 0.2, 0.95), MODERATE,
                   f"A model trained on human and AI writing rates this {p:.0%} AI-like "
                   f"({share} of {len(probs)} sections lean AI)."
                   + (f" Strongest signs: {', '.join(reasons)}." if reasons else "")
                   + " It is trained on a modest dataset, so treat it as one piece of evidence.",
                   {"chunk_probabilities": [round(q, 3) for q in probs], "reasons": reasons,
                    "model_version": m.get("version"), "validation": m.get("validation")})]
