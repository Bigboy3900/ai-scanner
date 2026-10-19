"""Trained detectors run locally with Hugging Face models.

Install the extras (`pip install -r requirements-ml.txt`) and the models download on
first use. Which models run is set in config.py / environment variables, so newer
detectors can be swapped in without code changes. If a model can't be loaded the
analyzer is skipped and the report says why.
"""

from __future__ import annotations

import io
import math
import re
from functools import lru_cache

from .. import config
from ..core import AUDIO, IMAGE, MODERATE, TEXT, Item, Signal, SkipAnalyzer, analyzer, clamp

AI_LABEL = re.compile(r"(artificial|\bai\b|ai[-_ ]?gen|fake|generated|synthetic|machine|gpt|chatgpt|spoof|deepfake|bot|label_1)", re.I)
HUMAN_LABEL = re.compile(r"(human|real|natural|authentic|bona[-_ ]?fide|genuine|label_0)", re.I)


def _require_transformers():
    if not config.ML_ENABLED:
        raise SkipAnalyzer("local models turned off (SCANNER_ML=0)")
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        raise SkipAnalyzer("local models not installed (pip install -r requirements-ml.txt)")


def _device() -> int | str:
    import torch

    if torch.cuda.is_available():
        return 0
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return -1


@lru_cache(maxsize=8)
def _pipeline(task: str, model: str):
    from transformers import pipeline

    return pipeline(task, model=model, device=_device())


def _ai_probability(preds: list[dict]) -> float | None:
    """Turn a classifier's label/score list into P(AI), whatever its labels are called."""
    ai = sum(p["score"] for p in preds if AI_LABEL.search(p["label"]) and not HUMAN_LABEL.search(p["label"]))
    human = sum(p["score"] for p in preds if HUMAN_LABEL.search(p["label"]) and not AI_LABEL.search(p["label"]))
    if ai + human == 0:
        return None
    return ai / (ai + human)


def _run_classifiers(task: str, models: list[str], inputs, analyzer_name: str, what: str) -> list[Signal]:
    _require_transformers()
    out, errors = [], []
    for model in models:
        try:
            preds = _pipeline(task, model)(inputs, top_k=None)
            if preds and isinstance(preds[0], list):
                preds = preds[0]
            p = _ai_probability(preds)
            if p is None:
                errors.append(f"{model}: unrecognised labels {[x['label'] for x in preds]}")
                continue
            out.append(Signal(analyzer_name, f"Detector model: {model.split('/')[-1]}", clamp(p, 0.02, 0.98), MODERATE,
                              f"The trained {what} detector {model} gives {p:.0%} probability of AI. Trained detectors are "
                              "accurate on generators they were trained on and much less so on newer ones.",
                              {"model": model, "raw": preds[:4]}))
        except Exception as e:
            errors.append(f"{model}: {type(e).__name__}: {str(e)[:160]}")
    if not out:
        raise SkipAnalyzer("; ".join(errors) or "no models configured")
    return out


@analyzer("ml_text_binoculars", (TEXT,), "Binoculars zero-shot detector: compares how two related language models see the text.")
def binoculars(item: Item) -> list[Signal]:
    _require_transformers()
    text = (item.text or "").strip()
    if len(text.split()) < 60:
        raise SkipAnalyzer("Binoculars needs at least 60 words")
    import torch

    try:
        tok, observer, performer = _binoculars_models(config.BINOCULARS_OBSERVER, config.BINOCULARS_PERFORMER)
    except Exception as e:
        raise SkipAnalyzer(f"could not load Binoculars models: {type(e).__name__}: {str(e)[:160]}")
    enc = tok(text[:20000], return_tensors="pt", truncation=True, max_length=512).to(observer.device)
    with torch.no_grad():
        obs = observer(**enc).logits[:, :-1].float()
        perf = performer(**enc).logits[:, :-1].float()
    targets = enc["input_ids"][:, 1:]
    log_ppl = torch.nn.functional.cross_entropy(perf.transpose(1, 2), targets).item()
    x_ppl = (torch.softmax(obs, -1) * -torch.log_softmax(perf, -1)).sum(-1).mean().item()
    score = log_ppl / x_ppl
    thr = config.BINOCULARS_THRESHOLD
    p = 1 / (1 + math.exp((score - thr) * 25))
    return [Signal("ml_text_binoculars", "Binoculars score", clamp(p, 0.03, 0.97), MODERATE,
                   f"Binoculars score {score:.3f} (threshold {thr}; lower means more machine-like). One of the strongest "
                   "published zero-shot text detectors, but still fooled by paraphrasing and heavy editing.",
                   {"binoculars": round(score, 4), "threshold": thr, "observer": config.BINOCULARS_OBSERVER,
                    "performer": config.BINOCULARS_PERFORMER})]


@lru_cache(maxsize=1)
def _binoculars_models(observer_name: str, performer_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    kw = {"torch_dtype": dtype}
    if torch.cuda.is_available():
        kw["device_map"] = "auto"
    tok = AutoTokenizer.from_pretrained(observer_name)
    observer = AutoModelForCausalLM.from_pretrained(observer_name, **kw).eval()
    performer = AutoModelForCausalLM.from_pretrained(performer_name, **kw).eval()
    return tok, observer, performer


@analyzer("ml_text_classifier", (TEXT,), "Trained text classifiers from Hugging Face (configurable).")
def text_classifiers(item: Item) -> list[Signal]:
    if len((item.text or "").split()) < 40:
        raise SkipAnalyzer("needs at least 40 words")
    return _run_classifiers("text-classification", config.TEXT_MODELS, item.text[:4000], "ml_text_classifier", "text")


@analyzer("ml_image_classifier", (IMAGE,), "Trained AI-image classifiers from Hugging Face (configurable).")
def image_classifiers(item: Item) -> list[Signal]:
    from PIL import Image

    try:
        im = Image.open(io.BytesIO(item.data)).convert("RGB")
    except Exception as e:
        raise SkipAnalyzer(f"could not decode image: {e}")
    return _run_classifiers("image-classification", config.IMAGE_MODELS, im, "ml_image_classifier", "image")


@analyzer("ml_audio_classifier", (AUDIO,), "Trained deepfake-voice classifiers from Hugging Face (configurable).")
def audio_classifiers(item: Item) -> list[Signal]:
    from .audio import load_audio

    _require_transformers()
    x, sr = load_audio(item)
    import numpy as np
    from scipy.signal import resample_poly

    x16 = resample_poly(x, 16000, sr).astype(np.float32)[: 16000 * 30]
    return _run_classifiers("audio-classification", config.AUDIO_MODELS, {"raw": x16, "sampling_rate": 16000},
                            "ml_audio_classifier", "voice")


@analyzer("ml_audio_watermark", (AUDIO,), "Meta AudioSeal watermark detector.")
def audioseal_check(item: Item) -> list[Signal]:
    if not config.ML_ENABLED:
        raise SkipAnalyzer("local models turned off (SCANNER_ML=0)")
    try:
        import torch
        from audioseal import AudioSeal
    except ImportError:
        raise SkipAnalyzer("audioseal not installed (pip install audioseal)")
    from .audio import load_audio
    from scipy.signal import resample_poly

    x, sr = load_audio(item)
    x16 = torch.tensor(resample_poly(x, 16000, sr), dtype=torch.float32)[None, None, : 16000 * 60]
    try:
        detector = _audioseal_detector()
        prob, _ = detector.detect_watermark(x16, 16000)
    except Exception as e:
        raise SkipAnalyzer(f"AudioSeal failed: {type(e).__name__}: {str(e)[:160]}")
    prob = float(prob)
    if prob > 0.9:
        return [Signal("ml_audio_watermark", "AudioSeal watermark found", 0.97, "strong",
                       f"Meta's AudioSeal watermark detected ({prob:.0%}).", {"prob": prob}, verdict="verified_ai")]
    return [Signal("ml_audio_watermark", "No AudioSeal watermark", 0.5, "weak",
                   "No AudioSeal watermark. Only some Meta and open-source voice tools add it.", {"prob": prob})]


@lru_cache(maxsize=1)
def _audioseal_detector():
    from audioseal import AudioSeal

    return AudioSeal.load_detector("audioseal_detector_16bits")
