"""Video-only checks: container details and frame-to-frame consistency.

Frames and the audio track are also scanned by the image and audio analyzers
(see ingest._expand_video); this module covers what only exists over time.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ..core import VIDEO, WEAK, Item, Signal, SkipAnalyzer, analyzer, clamp

GENERATOR_DURATIONS = (4, 5, 6, 8, 10)
PHONE_TAGS = ("com.apple.quicktime.make", "com.apple.quicktime.model", "com.android.version",
              "com.android.manufacturer", "location", "com.apple.quicktime.location.ISO6709")


@analyzer("video_container", (VIDEO,), "Checks duration, camera tags and stream layout against generator output.")
def container(item: Item) -> list[Signal]:
    probe = item.meta.get("probe") or {}
    if not probe:
        raise SkipAnalyzer("ffprobe unavailable or file unreadable")
    fmt = probe.get("format", {})
    tags = {k.lower(): v for k, v in (fmt.get("tags") or {}).items()}
    streams = probe.get("streams", [])
    if not streams:
        raise SkipAnalyzer("the video's streams couldn't be read")
    v = next((s for s in streams if s.get("codec_type") == "video"), {})
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    dur = float(fmt.get("duration") or 0)
    out = []
    phone = [t for t in PHONE_TAGS if t.lower() in tags]
    if phone:
        out.append(Signal("video_container", "Phone camera tags", 0.3, WEAK,
                          f"Carries phone recording tags ({', '.join(phone)}). These can be copied, but generators don't write them.",
                          {"tags": {t: tags[t.lower()] for t in phone}}))
    if dur and dur <= 20 and any(abs(dur - d) < 0.15 for d in GENERATOR_DURATIONS):
        out.append(Signal("video_container", "Generator-typical length", 0.58, WEAK,
                          f"{dur:.2f} seconds long, one of the fixed clip lengths video generators produce.", {"duration": dur}))
    if not has_audio and dur:
        out.append(Signal("video_container", "No audio track", 0.55, WEAK,
                          "The video has no audio at all. Phone and camera recordings almost always include sound; many generated clips don't."))
    if not out:
        out.append(Signal("video_container", "Ordinary container", 0.48, WEAK,
                          f"{v.get('width')}×{v.get('height')}, {dur:.1f}s, encoder {tags.get('encoder', 'unknown')}: nothing distinctive."))
    return out


@analyzer("video_temporal", (VIDEO,), "Measures texture 'swimming' and detail flicker between consecutive frames.")
def temporal(item: Item) -> list[Signal]:
    try:
        import cv2
    except ImportError:
        raise SkipAnalyzer("opencv not installed")
    if item.meta.get("burst_pngs"):
        frames = [cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_GRAYSCALE) for b in item.meta["burst_pngs"]]
    elif shutil.which("ffmpeg") and item.path:
        frames = _ffmpeg_burst(item, cv2)
    else:
        raise SkipAnalyzer("no consecutive frames could be decoded (ffmpeg not installed, or the browser can't play this format)")
    frames = [f for f in frames if f is not None]
    if len(frames) < 8:
        raise SkipAnalyzer("could not extract consecutive frames")
    warp_err, motion, detail = [], [], []
    for a, b in zip(frames, frames[1:]):
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        h, w = a.shape
        gx, gy = np.meshgrid(np.arange(w), np.arange(h))
        warped = cv2.remap(a, (gx + flow[..., 0]).astype(np.float32), (gy + flow[..., 1]).astype(np.float32),
                           cv2.INTER_LINEAR)
        edges = cv2.Laplacian(b, cv2.CV_64F)
        textured = np.abs(edges) > np.percentile(np.abs(edges), 70)
        warp_err.append(float(np.abs(warped.astype(float) - b.astype(float))[textured].mean()))
        motion.append(float(np.hypot(flow[..., 0], flow[..., 1]).mean()))
        detail.append(float(np.abs(edges).mean()))
    err = float(np.median(warp_err))
    mot = float(np.median(motion))
    flicker = float(np.std(detail) / (np.mean(detail) + 1e-9))
    # Texture that changes even after following the motion ("swimming") is a common generator artifact.
    swim = err / (1 + mot)
    out = [Signal("video_temporal", "Texture consistency", clamp(0.5 + (swim - 6) / 30, 0.35, 0.7), WEAK,
                  f"After following the motion between frames, textured areas still change by {err:.1f} levels on average "
                  f"(motion {mot:.1f} px/frame). Generated video often has textures that shift or morph on their own.",
                  {"warp_error": round(err, 2), "motion": round(mot, 2), "swim": round(swim, 2)}),
           Signal("video_temporal", "Detail flicker", clamp(0.5 + (flicker - 0.05) * 3, 0.4, 0.68), WEAK,
                  f"Fine detail varies by {flicker:.1%} from frame to frame.", {"flicker": round(flicker, 4)})]
    return out


def _ffmpeg_burst(item: Item, cv2) -> list:
    dur = float((item.meta.get("probe") or {}).get("format", {}).get("duration") or 0)
    d = tempfile.mkdtemp(prefix="burst_")
    start = max(0.0, dur / 2 - 1)
    subprocess.run(["ffmpeg", "-v", "quiet", "-ss", f"{start:.2f}", "-i", item.path, "-frames:v", "24",
                    "-vf", "scale='min(640,iw)':-2", os.path.join(d, "b%03d.png")], timeout=120)
    frames = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in sorted(Path(d).glob("b*.png"))]
    shutil.rmtree(d, ignore_errors=True)
    return frames
