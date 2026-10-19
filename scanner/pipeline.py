"""Run every applicable analyzer over a file and its parts, then combine the evidence."""

from __future__ import annotations

import math
import os
import shutil
import tempfile
import time
import traceback
from pathlib import Path

from . import analyzers  # noqa: F401  (registers analyzers)
from .core import MODERATE, REGISTRY, STRONG, TEXT, WEAK, Item, Signal, SkipAnalyzer, Skipped
from .ingest import build_item

BANDS = [
    ("verified_ai", "Verified AI", "Signed provenance, embedded generation settings or a decoded watermark says this was AI-generated."),
    ("likely_ai", "Likely AI", "Trained detectors and other evidence point clearly to AI."),
    ("possibly_ai", "Possibly AI", "Some signs of AI, but not enough to be confident."),
    ("inconclusive", "Inconclusive", "The evidence doesn't point clearly either way."),
    ("possibly_human", "Possibly human-made", "Leans towards human or camera origin, but AI can't be ruled out."),
    ("likely_human", "Likely human-made", "Detectors and other evidence point clearly to human or camera origin."),
    ("verified_capture", "Verified capture", "Signed Content Credentials record a real camera capture with no AI steps."),
]
BAND_INFO = {k: (label, desc) for k, label, desc in BANDS}

CAVEATS = {
    TEXT: "Text detection is the least reliable: editing, paraphrasing or 'humanizer' tools defeat it, and it wrongly "
          "flags non-native and very formal writers more often. Never use it as sole proof.",
    "general": "No tool can prove content is human-made. Missing watermarks or metadata mean little, since most tools "
               "don't add them and they are easy to remove.",
}


def logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def combine(signals: list[Signal]) -> dict:
    for verdict in ("verified_ai", "verified_capture"):
        hits = [s for s in signals if s.verdict == verdict]
        if hits:
            if verdict == "verified_capture" and any(s.strength == STRONG and s.score > 0.8 for s in signals):
                continue
            return _verdict(verdict, hits[0].score, f"{hits[0].name}: {hits[0].explanation}")

    strong = sum(max(-4, min(4, logit(s.score))) for s in signals if s.strength == STRONG)
    moderate = sum(max(-2.5, min(2.5, logit(s.score))) for s in signals if s.strength == MODERATE)
    # Heuristics often measure overlapping things, so together they may move the result only so far.
    weak = max(-1.0, min(1.0, sum(0.5 * logit(s.score) for s in signals if s.strength == WEAK)))
    total = strong + moderate + weak
    p = 1 / (1 + math.exp(-total))
    has_model = any(s.strength in (STRONG, MODERATE) and abs(s.score - 0.5) > 0.1 for s in signals)
    if p >= 0.8 and has_model:
        band = "likely_ai"
    elif p >= 0.62:
        band = "possibly_ai"
    elif p <= 0.2 and has_model:
        band = "likely_human"
    elif p <= 0.38:
        band = "possibly_human"
    else:
        band = "inconclusive"
    drivers = sorted(signals, key=lambda s: -abs(logit(s.score)) * {STRONG: 3, MODERATE: 1.5, WEAK: 0.5}[s.strength])
    top = [s.name for s in drivers[:3] if abs(s.score - 0.5) > 0.04]
    reason = f"Main factors: {', '.join(top)}." if top else "No signal stood out."
    if not has_model and band != "inconclusive":
        reason += " Based on heuristics only; add trained models or an API key for a firmer answer."
    return _verdict(band, p, reason)


def _verdict(band: str, p: float, reason: str) -> dict:
    label, desc = BAND_INFO[band]
    return {"band": band, "label": label, "description": desc, "ai_probability": round(p, 3), "reason": reason}


def analyze(item: Item, workdir: str, only: set[str] | None = None) -> dict:
    if item.path is None and item.data:
        item.path = os.path.join(workdir, f"part_{id(item)}{Path(item.name).suffix[:8]}")
        Path(item.path).write_bytes(item.data)
    signals: list[Signal] = []
    skipped: list[Skipped] = []
    timings = {}
    for spec in REGISTRY:
        if item.kind not in spec.kinds or (only and spec.name not in only):
            continue
        t0 = time.time()
        try:
            signals.extend(spec.fn(item))
        except SkipAnalyzer as e:
            skipped.append(Skipped(spec.name, str(e)))
        except Exception as e:
            skipped.append(Skipped(spec.name, f"error: {type(e).__name__}: {e}"))
            if os.environ.get("SCANNER_DEBUG"):
                traceback.print_exc()
        timings[spec.name] = round(time.time() - t0, 2)

    parts = [analyze(child, workdir, only) for child in item.children]
    own = combine(signals) if signals else _verdict("inconclusive", 0.5, "No analyzer could run on this part.")
    overall = own
    # A container is as suspicious as its most suspicious part ("contains AI-generated content").
    order = [b for b, _, _ in BANDS]
    for part in parts:
        pv = part["overall"]
        if order.index(pv["band"]) < order.index(overall["band"]) and pv["band"] in ("verified_ai", "likely_ai", "possibly_ai"):
            overall = dict(pv, reason=f"Part '{part['name']}': {pv['reason']}")
    caveats = [CAVEATS["general"]]
    if item.kind == TEXT or any(p["kind"] == TEXT for p in parts):
        caveats.insert(0, CAVEATS[TEXT])
    if item.meta.get("error"):
        caveats.append(item.meta["error"])
    return {
        "name": item.name,
        "kind": item.kind,
        "verdict": own,
        "overall": overall,
        "signals": [s.to_dict() for s in signals],
        "skipped": [vars(s) for s in skipped],
        "parts": parts,
        "caveats": caveats,
        "timings": timings,
        "words": len((item.text or "").split()) if item.kind == TEXT else None,
    }


def scan_bytes(data: bytes, name: str = "upload", only: set[str] | None = None, decoded: dict | None = None) -> dict:
    workdir = tempfile.mkdtemp(prefix="aiscan_")
    try:
        item = build_item(data, name, workdir, decoded)
        return analyze(item, workdir, only)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def scan_path(path: str, only: set[str] | None = None) -> dict:
    return scan_bytes(Path(path).read_bytes(), Path(path).name, only)


def scan_text(text: str, only: set[str] | None = None) -> dict:
    return scan_bytes(text.encode("utf-8"), "pasted text.txt", only)
