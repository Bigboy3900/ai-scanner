"""Train the writing-style text detector (scanner/models/text_style.json).

    pip install scikit-learn
    python tools/train_text_style.py [--cache ~/.cache/ai-scanner-corpora]

Human text: public corpora from nltk_data (news, fiction, essays, speeches, reviews,
forum posts), downloaded on first run. AI text: tools/data/ai_texts/*.txt, documents
separated by lines of "=====". Add more AI samples from more models there and re-run
to improve it; the validation numbers are written into the model file.
"""

import argparse
import io
import json
import random
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scanner.analyzers.text_style import FEATURE_NAMES, MODEL_PATH, chunks, features  # noqa: E402

NLTK = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/{}.zip"
CORPORA = ["brown", "gutenberg", "webtext", "movie_reviews", "inaugural", "state_union", "abc", "reuters"]
AI_DIR = ROOT / "tools" / "data" / "ai_texts"

LABELS = {
    "sent_len_cv": "very even sentence lengths", "sent_len_mean": "sentence length", "comma": "frequent commas",
    "contraction": "contractions", "ttr": "vocabulary variety", "word_len_mean": "longer words",
    "long_word_frac": "many long words", "question": "questions", "exclaim": "exclamations", "quote": "quotation marks",
    "dash": "dashes", "paren": "parentheses", "colon": "colons", "semicolon": "semicolons", "digit": "numbers",
    "sent_len_max_ratio": "no unusually long sentences", "para_len_cv": "evenly sized paragraphs",
}


def fetch(cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    for name in CORPORA:
        if not (cache / name).exists():
            print(f"downloading {name}…")
            data = urllib.request.urlopen(NLTK.format(name), timeout=120).read()
            zipfile.ZipFile(io.BytesIO(data)).extractall(cache)
    return cache


def human_docs(cache: Path) -> list[tuple[str, str]]:
    docs = []
    for f in sorted((cache / "brown").glob("c[a-r]*")):
        raw = f.read_text(errors="replace")
        text = re.sub(r"/[^\s]+", "", raw)  # strip part-of-speech tags
        docs.append(("brown", text))
    for f in sorted((cache / "gutenberg").glob("*.txt")):
        text = f.read_text(errors="replace")
        docs += [("gutenberg", text[i:i + 20000]) for i in range(0, min(len(text), 200000), 20000)]
    for f in sorted((cache / "webtext").glob("*.txt")):
        text = f.read_text(errors="replace")
        if "singles" in f.name or "overheard" in f.name:
            continue  # one-line fragments, not prose
        docs += [("webtext", text[i:i + 8000]) for i in range(0, len(text), 8000)]
    for f in sorted((cache / "movie_reviews").glob("*/*.txt")):
        docs.append(("movie_reviews", f.read_text(errors="replace")))
    for name in ("inaugural", "state_union"):
        for f in sorted((cache / name).glob("*.txt")):
            docs.append((name, f.read_text(errors="replace")))
    for f in sorted((cache / "abc").glob("*.txt")):
        text = f.read_text(errors="replace")
        docs += [("abc", text[i:i + 6000]) for i in range(0, len(text), 6000)]
    for f in sorted((cache / "reuters" / "training").glob("*"))[:3000]:
        docs.append(("reuters", f.read_text(errors="replace")))
    return docs


def ai_docs() -> list[tuple[str, str]]:
    docs = []
    for f in sorted(AI_DIR.glob("*.txt")):
        for part in re.split(r"^=====\s*$", f.read_text(), flags=re.M):
            if len(part.split()) > 80:
                docs.append((f"ai:{f.stem}", part.strip()))
    return docs


def build(docs, label, max_chunks_per_source=None, seed=0):
    rng = random.Random(seed)
    X, y, groups, sources = [], [], [], []
    per_source: dict[str, list] = {}
    for gi, (src, text) in enumerate(docs):
        for c in chunks(text):
            if len(c.split()) >= 120:
                per_source.setdefault(src, []).append((gi, c))
    for src, items in per_source.items():
        rng.shuffle(items)
        if max_chunks_per_source:
            items = items[:max_chunks_per_source]
        for gi, c in items:
            X.append(features(c))
            y.append(label)
            groups.append(f"{label}-{gi}")
            sources.append(src)
    return X, y, groups, sources


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(Path.home() / ".cache" / "ai-scanner-corpora"))
    args = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    cache = fetch(Path(args.cache))
    hX, hy, hg, hs = build(human_docs(cache), 0, max_chunks_per_source=400)
    aX, ay, ag, as_ = build(ai_docs(), 1)
    X = np.array(hX + aX)
    y = np.array(hy + ay)
    groups = np.array(hg + ag)
    sources = np.array(hs + as_)
    print(f"{(y == 0).sum()} human chunks, {(y == 1).sum()} AI chunks")

    def model(C):
        return LogisticRegression(C=C, class_weight="balanced", max_iter=5000)

    # Grouped cross-validation: chunks of one document never sit on both sides.
    best = None
    for C in (0.01, 0.03, 0.1, 0.3):
        probs = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
            sc = StandardScaler().fit(X[tr])
            probs[te] = model(C).fit(sc.transform(X[tr]), y[tr]).predict_proba(sc.transform(X[te]))[:, 1]
        fpr = float((probs[y == 0] > 0.5).mean())
        tpr = float((probs[y == 1] > 0.5).mean())
        print(f"C={C}: AI caught {tpr:.1%}, humans wrongly flagged {fpr:.1%}")
        if best is None or tpr - 2 * fpr > best[0]:
            best = (tpr - 2 * fpr, C, probs, tpr, fpr)
    _, C, probs, tpr, fpr = best
    print(f"chosen C={C}")
    per_source = {s: round(float((probs[(sources == s)] > 0.5).mean()), 3) for s in sorted(set(sources))}
    for s, v in per_source.items():
        print(f"  {s:20} flagged as AI: {v:.1%}")

    sc = StandardScaler().fit(X)
    clf = model(C).fit(sc.transform(X), y)
    labels = dict(LABELS)
    for i, name in enumerate(FEATURE_NAMES):
        if name.startswith("w:"):
            labels[name] = f"frequent use of '{name[2:]}'"
    MODEL_PATH.parent.mkdir(exist_ok=True)
    MODEL_PATH.write_text(json.dumps({
        "version": 1,
        "features": FEATURE_NAMES,
        "mean": sc.mean_.round(6).tolist(),
        "scale": np.where(sc.scale_ == 0, 1, sc.scale_).round(6).tolist(),
        "coef": clf.coef_[0].round(6).tolist(),
        "intercept": round(float(clf.intercept_[0]), 6),
        "labels": labels,
        "validation": {"ai_caught": round(tpr, 3), "human_flagged": round(fpr, 3),
                       "human_flagged_by_source": per_source,
                       "n_human_chunks": int((y == 0).sum()), "n_ai_chunks": int((y == 1).sum())},
    }))
    print(f"wrote {MODEL_PATH}")


if __name__ == "__main__":
    main()
