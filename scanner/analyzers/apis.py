"""Commercial detection APIs. Each runs only when its API key is configured.

Uploading sends the file to that company, so these are off unless you add a key.
Response formats follow each vendor's public docs; if a vendor changes theirs the
analyzer reports the error rather than guessing.
"""

from __future__ import annotations

from .. import config
from ..core import IMAGE, MODERATE, TEXT, VIDEO, Item, Signal, SkipAnalyzer, analyzer, clamp

TIMEOUT = 120


def _http():
    try:
        import httpx
    except ImportError:
        raise SkipAnalyzer("httpx not installed (not available in the browser version)")

    return httpx.Client(timeout=TIMEOUT)


def _check(resp) -> dict:
    try:
        body = resp.json()
    except Exception:
        raise SkipAnalyzer(f"HTTP {resp.status_code}: {resp.text[:200]}")
    if resp.status_code >= 400:
        raise SkipAnalyzer(f"HTTP {resp.status_code}: {str(body)[:200]}")
    return body


@analyzer("api_sightengine", (IMAGE, VIDEO), "Sightengine AI-generated image/video detection (paid API).")
def sightengine(item: Item) -> list[Signal]:
    if not (config.SIGHTENGINE_USER and config.SIGHTENGINE_SECRET):
        raise SkipAnalyzer("no Sightengine key (SIGHTENGINE_API_USER / SIGHTENGINE_API_SECRET)")
    if item.meta.get("from_video"):
        raise SkipAnalyzer("the whole video is sent instead of single frames")
    auth = {"models": "genai", "api_user": config.SIGHTENGINE_USER, "api_secret": config.SIGHTENGINE_SECRET}
    with _http() as c:
        if item.kind == IMAGE:
            body = _check(c.post("https://api.sightengine.com/1.0/check.json", data=auth,
                                 files={"media": (item.name, item.data)}))
            p = body.get("type", {}).get("ai_generated")
        else:
            body = _check(c.post("https://api.sightengine.com/1.0/video/check-sync.json", data=auth,
                                 files={"media": (item.name, item.data)}))
            frames = body.get("data", {}).get("frames", [])
            scores = [f.get("type", {}).get("ai_generated") for f in frames if f.get("type")]
            p = max(scores) if scores else None
    if p is None:
        raise SkipAnalyzer(f"unexpected response: {str(body)[:200]}")
    return [Signal("api_sightengine", "Sightengine", clamp(p, 0.02, 0.98), MODERATE,
                   f"Sightengine's AI-content model gives {p:.0%} probability of AI.", {"response": body})]


@analyzer("api_hive", (IMAGE, VIDEO), "Hive AI-generated content detection (paid API).")
def hive(item: Item) -> list[Signal]:
    if not config.HIVE_API_KEY:
        raise SkipAnalyzer("no Hive key (HIVE_API_KEY)")
    if item.meta.get("from_video"):
        raise SkipAnalyzer("the whole video is sent instead of single frames")
    with _http() as c:
        body = _check(c.post("https://api.thehive.ai/api/v2/task/sync",
                             headers={"Authorization": f"Token {config.HIVE_API_KEY}"},
                             files={"media": (item.name, item.data)}))
    scores, sources = [], {}
    for status in body.get("status", []):
        for output in status.get("response", {}).get("output", []):
            for cls in output.get("classes", []):
                if cls.get("class") == "ai_generated":
                    scores.append(cls.get("score", 0))
                elif cls.get("score", 0) > 0.5 and cls.get("class") not in ("not_ai_generated", "none", "inconclusive"):
                    sources[cls["class"]] = round(cls["score"], 3)
    if not scores:
        raise SkipAnalyzer(f"unexpected response: {str(body)[:200]}")
    p = max(scores)
    src = f" Likely source: {max(sources, key=sources.get)}." if sources else ""
    return [Signal("api_hive", "Hive", clamp(p, 0.02, 0.98), MODERATE,
                   f"Hive's AI-content model gives {p:.0%} probability of AI.{src}", {"sources": sources})]


@analyzer("api_gptzero", (TEXT,), "GPTZero AI-text detection (paid API).")
def gptzero(item: Item) -> list[Signal]:
    if not config.GPTZERO_API_KEY:
        raise SkipAnalyzer("no GPTZero key (GPTZERO_API_KEY)")
    if len((item.text or "").split()) < 40:
        raise SkipAnalyzer("needs at least 40 words")
    with _http() as c:
        body = _check(c.post("https://api.gptzero.me/v2/predict/text",
                             headers={"x-api-key": config.GPTZERO_API_KEY},
                             json={"document": item.text[:50000]}))
    doc = (body.get("documents") or [{}])[0]
    probs = doc.get("class_probabilities") or {}
    p = probs.get("ai", doc.get("completely_generated_prob"))
    if p is None:
        raise SkipAnalyzer(f"unexpected response: {str(body)[:200]}")
    mixed = probs.get("mixed")
    extra = f" ({mixed:.0%} mixed human/AI)" if mixed else ""
    return [Signal("api_gptzero", "GPTZero", clamp(p, 0.02, 0.98), MODERATE,
                   f"GPTZero gives {p:.0%} probability the text is AI-written{extra}.", {"class_probabilities": probs})]
