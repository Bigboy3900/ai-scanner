"""Model-free audio checks: bandwidth cutoffs, digital silence and pause rhythm.

Voice cloning and text-to-speech models usually generate at 16-24 kHz, so their output
has no content above 8-12 kHz even when saved at 44.1/48 kHz, and they tend to produce
perfectly clean pauses. Phone calls and low-bitrate files share some of these traits.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import numpy as np

from ..core import AUDIO, WEAK, Item, Signal, SkipAnalyzer, analyzer, clamp
from .image import to_png_data_uri

MAX_SECONDS = 120


def load_audio(item: Item) -> tuple[np.ndarray, int]:
    if "samples" in item.meta:
        return item.meta["samples"], item.meta["sr"]
    if not shutil.which("ffmpeg"):
        raise SkipAnalyzer("ffmpeg not installed")
    src = item.path
    if not src:
        fd, src = tempfile.mkstemp()
        os.write(fd, item.data)
        os.close(fd)
    probe_sr = 48000
    try:
        out = subprocess.run(["ffprobe", "-v", "quiet", "-select_streams", "a:0", "-show_entries", "stream=sample_rate",
                              "-of", "csv=p=0", src], capture_output=True, text=True, timeout=30).stdout.strip()
        probe_sr = min(int(out.split()[0]), 48000) if out else 48000
    except Exception:
        pass
    raw = subprocess.run(["ffmpeg", "-v", "quiet", "-i", src, "-t", str(MAX_SECONDS), "-ac", "1", "-ar", str(probe_sr),
                          "-f", "f32le", "-"], capture_output=True, timeout=300).stdout
    x = np.frombuffer(raw, dtype=np.float32).astype(np.float64)
    if len(x) < probe_sr:
        raise SkipAnalyzer("audio shorter than one second or could not be decoded")
    item.meta["samples"], item.meta["sr"] = x, probe_sr
    return x, probe_sr


def stft_db(x: np.ndarray, sr: int, n: int = 2048, hop: int = 512) -> np.ndarray:
    # Build the hopped view directly; a full sliding window view is too large for 32-bit (browser) builds.
    x = np.ascontiguousarray(x)
    count = 1 + (len(x) - n) // hop
    frames = np.lib.stride_tricks.as_strided(x, shape=(count, n), strides=(hop * x.strides[0], x.strides[0]),
                                             writeable=False) * np.hanning(n)
    S = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    return 10 * np.log10(S + 1e-12)


@analyzer("audio_bandwidth", (AUDIO,), "Finds hard frequency cutoffs left by speech and music generators.")
def bandwidth(item: Item) -> list[Signal]:
    x, sr = load_audio(item)
    S = stft_db(x, sr)
    rms = 10 * np.log10(np.mean(np.power(10, S / 10), axis=1) + 1e-12)
    loud = S[rms > np.percentile(rms, 50)]
    spec = 10 * np.log10(np.mean(np.power(10, loud / 10), axis=0) + 1e-12)
    freqs = np.fft.rfftfreq(2048, 1 / sr)
    # Look for the steepest edge: the biggest drop in level across ~1 kHz above 3 kHz.
    step = max(1, int(round(1000 / (freqs[1] - freqs[0]))))
    smooth = np.convolve(spec, np.ones(5) / 5, mode="same")
    drops = smooth[:-step] - smooth[step:]
    drops[freqs[:-step] < 3000] = 0
    i = int(np.argmax(drops))
    drop = float(drops[i])
    # The cutoff is where the level first falls 12 dB below the band just before the edge.
    pre = float(np.median(smooth[max(0, i - step):i + 1]))
    j0 = max(0, i - step)
    below = np.where(smooth[j0:i + step + 1] < pre - 12)[0]
    cutoff = float(freqs[j0 + below[0]]) if len(below) else float(freqs[i])
    if drop < 25:
        cutoff = float(freqs[-1])
    nyq = sr / 2
    vis = {"visual": to_png_data_uri(np.flipud(S[:: max(1, len(S) // 600)].T)),
           "visual_caption": "Spectrogram (time left to right, frequency bottom to top)"}
    detail = {"cutoff_hz": round(cutoff), "sample_rate": sr, "edge_drop_db": round(drop, 1), **vis}
    typical_gen = any(abs(cutoff - f) < 700 for f in (8000, 11025, 12000))
    if cutoff < nyq * 0.9 and typical_gen and drop > 30 and sr >= 32000:
        return [Signal("audio_bandwidth", "Hard cutoff at a generator rate", 0.72, WEAK,
                       f"Nothing above {cutoff / 1000:.1f} kHz despite a {sr / 1000:.1f} kHz file, with a sharp edge. "
                       "Speech generators commonly output 16-24 kHz audio, which leaves exactly this cutoff. "
                       "Upsampled phone or video-call audio can look the same.", detail)]
    if cutoff < 5000 and sr >= 16000:
        return [Signal("audio_bandwidth", "Narrowband audio", 0.5, WEAK,
                       f"Content stops at {cutoff / 1000:.1f} kHz, typical of phone recordings; it hides most other traces.", detail)]
    return [Signal("audio_bandwidth", "Full or encoder-typical bandwidth", 0.45, WEAK,
                   f"Content extends to {cutoff / 1000:.1f} kHz, consistent with a normal recording or MP3/AAC encoding.", detail)]


@analyzer("audio_silence", (AUDIO,), "Checks pauses for a natural room-noise floor and natural timing.")
def silence(item: Item) -> list[Signal]:
    x, sr = load_audio(item)
    frame = int(sr * 0.02)
    n = len(x) // frame
    rms = np.sqrt(np.mean(x[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-20)
    db = 20 * np.log10(rms)
    floor = float(np.percentile(db, 5))
    peak = float(np.percentile(db, 95))
    out = []
    exact_zero = float(np.mean(np.abs(x) < 1e-7))
    if floor < -85 or exact_zero > 0.05:
        out.append(Signal("audio_silence", "Digitally clean pauses", 0.65, WEAK,
                          f"The quietest moments sit at {floor:.0f} dBFS ({exact_zero:.0%} of samples are exactly zero). Real "
                          "microphones always pick up some room noise; generated speech and heavily gated studio audio don't.",
                          {"noise_floor_db": round(floor, 1), "exact_zero_fraction": round(exact_zero, 3)}))
    else:
        out.append(Signal("audio_silence", "Natural noise floor", 0.4, WEAK,
                          f"Background noise floor of {floor:.0f} dBFS, as a real microphone would record.",
                          {"noise_floor_db": round(floor, 1)}))
    # Pause rhythm: TTS pauses are strikingly uniform in length.
    thresh = floor + (peak - floor) * 0.25
    quiet = db < thresh
    runs, cur = [], 0
    for q in quiet:
        if q:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    pauses = np.array([r for r in runs if r >= 8]) * 0.02  # pauses of at least 160 ms
    if len(pauses) >= 6:
        cv = float(pauses.std() / pauses.mean())
        out.append(Signal("audio_silence", "Pause timing", clamp(0.5 + (0.55 - cv) * 0.5, 0.35, 0.65), WEAK,
                          f"{len(pauses)} pauses with timing variation {cv:.2f}. People pause irregularly; synthetic speech "
                          "is often more even.", {"n_pauses": len(pauses), "pause_cv": round(cv, 3)}))
    return out
