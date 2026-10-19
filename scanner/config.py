"""Settings, read from environment variables (or a .env file next to the project)."""

from __future__ import annotations

import os
from pathlib import Path


def _load_dotenv() -> None:
    env = Path(__file__).resolve().parent.parent / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


def _list(name: str, default: str) -> list[str]:
    return [x.strip() for x in os.environ.get(name, default).split(",") if x.strip()]


# Local Hugging Face models (only used when requirements-ml.txt is installed).
ML_ENABLED = os.environ.get("SCANNER_ML", "1") != "0"
BINOCULARS_OBSERVER = os.environ.get("SCANNER_BINOCULARS_OBSERVER", "tiiuae/falcon-7b")
BINOCULARS_PERFORMER = os.environ.get("SCANNER_BINOCULARS_PERFORMER", "tiiuae/falcon-7b-instruct")
# Published accuracy-optimised threshold for the Falcon-7B pair. Recalibrate if you change models.
BINOCULARS_THRESHOLD = float(os.environ.get("SCANNER_BINOCULARS_THRESHOLD", "0.9015"))
TEXT_MODELS = _list("SCANNER_TEXT_MODELS", "Hello-SimpleAI/chatgpt-detector-roberta")
IMAGE_MODELS = _list("SCANNER_IMAGE_MODELS", "Organika/sdxl-detector,umm-maybe/AI-image-detector")
AUDIO_MODELS = _list("SCANNER_AUDIO_MODELS", "MelodyMachine/Deepfake-audio-detection-V2")

# Paid detection APIs: each runs only when its key is set.
SIGHTENGINE_USER = os.environ.get("SIGHTENGINE_API_USER")
SIGHTENGINE_SECRET = os.environ.get("SIGHTENGINE_API_SECRET")
HIVE_API_KEY = os.environ.get("HIVE_API_KEY")
GPTZERO_API_KEY = os.environ.get("GPTZERO_API_KEY")

MAX_UPLOAD_MB = int(os.environ.get("SCANNER_MAX_UPLOAD_MB", "500"))
