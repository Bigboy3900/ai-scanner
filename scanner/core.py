"""Shared data types for items, signals and analyzers."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Callable

# How much a signal should be trusted.
#   strong   - cryptographic provenance or a decoded watermark; can decide the verdict alone
#   moderate - a trained detector or a well-studied statistic
#   weak     - a heuristic; only meaningful when several agree
STRONG, MODERATE, WEAK = "strong", "moderate", "weak"

# Kinds of item the pipeline understands.
TEXT, IMAGE, AUDIO, VIDEO, DOCUMENT, UNKNOWN = "text", "image", "audio", "video", "document", "unknown"


@dataclass
class Item:
    """One piece of content to analyze. Containers produce child items."""

    kind: str
    name: str
    data: bytes = b""
    text: str | None = None
    mime: str | None = None
    path: str | None = None  # temp file on disk, for tools that need one
    meta: dict[str, Any] = field(default_factory=dict)
    children: list["Item"] = field(default_factory=list)


@dataclass
class Signal:
    """One piece of evidence from one analyzer.

    score is the probability-like belief that the content is AI-generated
    (0 = looks human/camera-made, 0.5 = no information, 1 = looks AI).
    """

    analyzer: str
    name: str
    score: float
    strength: str
    explanation: str
    details: dict[str, Any] = field(default_factory=dict)
    # Set when the signal is decisive by itself (e.g. a valid C2PA "AI generated" claim).
    verdict: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["score"] = round(float(self.score), 3)
        return d


@dataclass
class Skipped:
    analyzer: str
    reason: str


Analyzer = Callable[[Item], list[Signal]]


@dataclass
class AnalyzerSpec:
    name: str
    kinds: tuple[str, ...]
    fn: Analyzer
    description: str


REGISTRY: list[AnalyzerSpec] = []


class SkipAnalyzer(Exception):
    """Raised by an analyzer that cannot run (missing model, API key, too little input)."""


def analyzer(name: str, kinds: tuple[str, ...], description: str):
    def wrap(fn: Analyzer) -> Analyzer:
        REGISTRY.append(AnalyzerSpec(name, kinds, fn, description))
        return fn

    return wrap


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))
