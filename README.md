# AI Scanner

Drop in any file (text, image, audio, video, PDF, Word, PowerPoint) and it checks for signs it was made by AI, then shows every piece of evidence behind the verdict.

**Try it: https://bigboy3900.github.io/ai-scanner/** (runs entirely in your browser; files are never uploaded)

Pasting text works without an account. Scanning files needs a free account (email sign-in link, Apple, Google, GitHub or Discord); see [AUTH_SETUP.md](AUTH_SETUP.md) to switch accounts on.

## Two versions

- **Website (GitHub Pages):** the same Python checks, run in the browser with [Pyodide](https://pyodide.org). The browser decodes audio and video and reads Content Credentials with [c2pa-web](https://github.com/contentauth/c2pa-js). It can't run the trained AI models or paid APIs. Source in `web/`, assembled by `tools/build_site.py`, deployed by `.github/workflows/pages.yml` on every push to `main`.
- **Full version (your computer):** adds trained detector models and paid APIs. Instructions below.

## Run the full version locally

Needs Python 3.10+ and `ffmpeg` (for audio and video).

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install --no-deps invisible-watermark

python -m scanner serve            # then open http://127.0.0.1:8000
python -m scanner photo.jpg        # command line
python -m scanner --text "paste text here"
python -m scanner -v --json file.pdf   # every signal, as JSON
python -m pytest tests             # tests
python tools/build_site.py && python -m http.server -d _site 8080   # preview the website version
```

Optional extras, for stronger detection:

```bash
pip install -r requirements-ml.txt   # local trained detectors (PyTorch; models download on first use)
cp .env.example .env                 # add paid API keys (Hive, Sightengine, GPTZero) if you have them
```

## What it checks

| Layer | Checks | Media |
|---|---|---|
| Provenance | C2PA Content Credentials (signature validated), IPTC "AI-generated" source type, generator names and prompts in metadata, camera EXIF | all files |
| Watermarks | Stable Diffusion invisible watermark, Meta AudioSeal (with extras) | image, audio |
| Text | Trained writing-style model (function words, sentence rhythm, punctuation habits; runs everywhere, including the website), AI-typical vocabulary, chatbot boilerplate, formatting habits, hidden Unicode characters; Binoculars and a Hugging Face classifier (with extras); GPTZero (with key) | text, and text inside documents |
| Image forensics | Fourier spectrum peaks from upsampling, high-frequency falloff, sensor noise vs. brightness, JPEG quantization tables, generator-typical sizes; trained classifiers (with extras); Hive / Sightengine (with keys) | images, document images, video frames |
| Audio forensics | Hard bandwidth cutoffs from 16-24 kHz generators, digitally clean pauses, pause timing; deepfake-voice classifier (with extras) | audio, video soundtracks |
| Video | Texture "swimming" after motion compensation, detail flicker, generator-typical lengths, phone camera tags, missing audio; plus all image checks on frames and audio checks on the soundtrack | video |
| Documents | Authoring app, editing time vs. word count, Word revision sessions; plus text and embedded-image checks | PDF, DOCX, PPTX, XLSX, ODT |

Each check returns a score (0 = looks human, 1 = looks AI) and a strength:

- **strong**: signed provenance, embedded generation settings, decoded watermark. Can decide the verdict alone.
- **moderate**: trained detectors and paid APIs.
- **weak**: heuristics. Combined, they can move the verdict only so far, so heuristics alone never produce "Likely AI".

Verdicts: Verified AI, Likely AI, Possibly AI, Inconclusive, Possibly human-made, Likely human-made, Verified capture. A container (document, video) is flagged if any part of it is.

## Honest limits

- The heuristic thresholds are set from published research and quick tests, not calibrated on a labeled dataset yet. Treat their scores as hints.
- The local models and paid APIs are wired up but were not run while this was built: the build machine couldn't reach Hugging Face, and no API keys were available. They skip themselves with a reason if they can't load.
- Text detection is easy to beat (paraphrasing, editing) and over-flags non-native and formal writers.
- Screenshots, resizing and recompression erase most image and audio traces.
- A missing watermark or missing metadata proves nothing.

## The writing-style model

`scanner/models/text_style.json` is a small logistic-regression model trained by `tools/train_text_style.py` on public human-written corpora (news, fiction, speeches, reviews, forum posts from nltk_data) and the AI-written samples in `tools/data/ai_texts/`. In cross-validation it catches about 82% of AI-written 200-word sections while flagging about 1.7% of human ones; whole documents do better because sections are combined. Its human data has no modern student essays and its AI data comes mainly from one model family, so add samples from other chatbots (and real student writing) to `tools/data/` and re-run the script to improve it.

## Adding a detector

Write a function in `scanner/analyzers/`, decorate it, and it runs automatically:

```python
@analyzer("my_check", (IMAGE,), "What it looks at.")
def my_check(item: Item) -> list[Signal]:
    ...
    return [Signal("my_check", "Name shown to users", score, MODERATE, "Plain-language explanation.")]
```

Raise `SkipAnalyzer("reason")` when it can't run.
