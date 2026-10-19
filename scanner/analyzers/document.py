"""Document-level checks. Text and embedded images are scanned separately as child items."""

from __future__ import annotations

import re

from ..core import DOCUMENT, WEAK, Item, Signal, SkipAnalyzer, analyzer
from .provenance import GEN_RE


@analyzer("document_properties", (DOCUMENT,), "Reads authoring app, edit time and revision count from document properties.")
def properties(item: Item) -> list[Signal]:
    info = item.meta.get("doc_info") or {}
    if not info and "word_revisions" not in item.meta:
        raise SkipAnalyzer("no document properties")
    out = []
    tools = " ".join(str(info.get(k, "")) for k in ("Producer", "Creator", "Application", "AppVersion", "generator", "creator"))
    if GEN_RE.search(tools):
        out.append(Signal("document_properties", "AI tool listed as creator", 0.8, WEAK,
                          f"The document says it was produced by: {tools.strip()}.", {"tools": tools.strip()}))
    total_time = info.get("TotalTime")
    words = info.get("Words")
    if total_time is not None and words:
        try:
            minutes, n_words = int(total_time), int(words)
            if n_words > 300 and minutes <= 2:
                out.append(Signal("document_properties", "Very short editing time", 0.62, WEAK,
                                  f"{n_words} words but only {minutes} minute(s) of recorded editing time, which suggests the text "
                                  "was pasted in rather than written here.", {"minutes": minutes, "words": n_words}))
        except ValueError:
            pass
    revs = item.meta.get("word_revisions")
    if revs is not None and revs <= 2:
        out.append(Signal("document_properties", "Few editing sessions", 0.56, WEAK,
                          f"Word recorded {revs} editing session(s) for the whole text, consistent with one paste.",
                          {"revision_ids": revs}))
    if not out:
        summary = ", ".join(f"{k}: {v}" for k, v in list(info.items())[:4] if v and not re.search(r"\d{4}-\d\d-\d\dT", str(v)))
        out.append(Signal("document_properties", "Ordinary properties", 0.48, WEAK, f"Nothing unusual ({summary or 'few properties'})."))
    return out
