"""Model-free image forensics: frequency spectrum, sensor noise, JPEG tables and dimensions.

These look for traces that generators leave (periodic upsampling artifacts, missing
camera sensor noise) and that cameras leave (signal-dependent noise, camera JPEG tables).
They are heuristics: resizing, screenshots and heavy compression weaken all of them.
"""

from __future__ import annotations

import base64
import io

import numpy as np

from ..core import IMAGE, WEAK, Item, Signal, SkipAnalyzer, analyzer, clamp

# Output sizes common to popular generators (width, height), either orientation.
GENERATOR_SIZES = {
    (512, 512), (768, 768), (1024, 1024), (2048, 2048), (1024, 1792), (1024, 1536), (1536, 1024),
    (832, 1216), (896, 1152), (768, 1344), (640, 1536), (1456, 816), (1232, 928), (1344, 896), (1360, 768),
    (1248, 832), (1184, 864), (1152, 896), (1408, 768), (1792, 1024), (1216, 832), (1344, 768),
}

# Standard IJG luminance table at quality 50; libjpeg/Pillow/most software scale this.
IJG_LUMA = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99])


def load_gray(item: Item, max_side: int = 1024) -> tuple[np.ndarray, "object"]:
    from PIL import Image, ImageOps

    try:
        im = Image.open(io.BytesIO(item.data))
        im.load()
    except Exception as e:
        raise SkipAnalyzer(f"could not decode image: {e}")
    im = ImageOps.exif_transpose(im)
    if min(im.size) < 64:
        raise SkipAnalyzer("image too small")
    g = im.convert("L")
    if max(g.size) > max_side:
        # Center crop rather than resize: resizing would destroy the high-frequency traces we look for.
        w, h = g.size
        cw, ch = min(w, max_side), min(h, max_side)
        g = g.crop(((w - cw) // 2, (h - ch) // 2, (w - cw) // 2 + cw, (h - ch) // 2 + ch))
    return np.asarray(g, dtype=np.float64), im


def residual(gray: np.ndarray) -> np.ndarray:
    from scipy.ndimage import median_filter

    return gray - median_filter(gray, size=3)


def to_png_data_uri(arr: np.ndarray) -> str:
    from PIL import Image

    a = arr - arr.min()
    a = (255 * a / (a.max() or 1)).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@analyzer("image_spectrum", (IMAGE,), "Looks for periodic upsampling peaks and unnatural high-frequency falloff in the Fourier spectrum.")
def spectrum(item: Item) -> list[Signal]:
    gray, im = load_gray(item)
    n = min(gray.shape)
    n -= n % 16
    g = gray[:n, :n]
    r = residual(g)
    win = np.outer(np.hanning(n), np.hanning(n))
    F = np.fft.fftshift(np.abs(np.fft.fft2(r * win)))
    logF = np.log1p(F)

    # Periodic artifacts: bright isolated peaks away from the low-frequency centre.
    c = n // 2
    yy, xx = np.mgrid[:n, :n]
    rad = np.hypot(yy - c, xx - c)
    mask = rad > n * 0.12
    is_jpeg = item.data[:3] == b"\xff\xd8\xff"
    if is_jpeg:
        # JPEG's 8x8 blocks put peaks at multiples of n/8; ignore those positions.
        step = n // 8
        for k in range(-4, 5):
            for axis in (yy, xx):
                mask &= np.abs(axis - (c + k * step)) > 2
    vals = logF[mask]
    med, mad = np.median(vals), np.median(np.abs(vals - np.median(vals))) + 1e-9
    z = (vals - med) / mad
    n_peaks = int((z > 9).sum())
    peak_score = clamp(0.5 + min(n_peaks, 40) / 80, 0.5, 0.85)

    # Radially averaged power spectrum slope at high frequencies.
    P = np.abs(np.fft.fftshift(np.fft.fft2(g * win))) ** 2
    rbin = rad.astype(int)
    radial = np.bincount(rbin.ravel(), P.ravel()) / np.maximum(np.bincount(rbin.ravel()), 1)
    f = np.arange(len(radial))
    sel = (f > n * 0.1) & (f < n * 0.45)
    slope = float(np.polyfit(np.log(f[sel]), np.log(radial[sel] + 1e-9), 1)[0])
    # Natural photos fall off around -2 to -3. Much flatter suggests synthetic texture;
    # much steeper suggests an over-smoothed (or heavily downscaled / denoised) image.
    dev = abs(slope + 2.6)
    slope_score = clamp(0.5 + (dev - 0.8) * 0.25, 0.4, 0.7)

    return [
        Signal("image_spectrum", "Periodic frequency peaks", peak_score, WEAK,
               f"{n_peaks} strong isolated peaks in the noise spectrum. Generator upsampling layers tend to leave a "
               "grid of such peaks; camera photos usually have none." + (" JPEG block positions were excluded." if is_jpeg else ""),
               {"n_peaks": n_peaks, "visual": to_png_data_uri(logF), "visual_caption": "Noise spectrum (bright dots away from the centre are periodic artifacts)"}),
        Signal("image_spectrum", "High-frequency falloff", slope_score, WEAK,
               f"Power spectrum slope {slope:.2f} (typical photos: about -2 to -3).", {"slope": round(slope, 2)}),
    ]


@analyzer("image_noise", (IMAGE,), "Checks for camera-like sensor noise that rises with brightness.")
def noise(item: Item) -> list[Signal]:
    gray, im = load_gray(item)
    r = residual(gray)
    b = 32
    h, w = (gray.shape[0] // b) * b, (gray.shape[1] // b) * b
    blocks_mean = gray[:h, :w].reshape(h // b, b, w // b, b).mean(axis=(1, 3)).ravel()
    blocks_std = r[:h, :w].reshape(h // b, b, w // b, b).std(axis=(1, 3)).ravel()
    # Keep only fairly flat blocks, where residual is mostly noise rather than edges.
    texture = np.array([np.abs(np.diff(gray[i:i + b, j:j + b], axis=0)).mean()
                        for i in range(0, h, b) for j in range(0, w, b)])
    flat = (texture < np.percentile(texture, 40)) & (blocks_mean > 15) & (blocks_mean < 240)
    if flat.sum() < 12:
        raise SkipAnalyzer("not enough flat regions to measure noise")
    m, s = blocks_mean[flat], blocks_std[flat]
    corr = float(np.corrcoef(m, s)[0, 1]) if s.std() > 1e-6 else 0.0
    level = float(np.median(s))
    out = []
    # Real sensors: shot noise grows with brightness (positive correlation) and is never zero.
    if level < 0.35:
        out.append(Signal("image_noise", "Almost no sensor noise", 0.65, WEAK,
                          f"Smooth areas are nearly noise-free (median {level:.2f}). Generated images and heavily "
                          "denoised or compressed photos look like this.", {"noise_level": round(level, 3)}))
    else:
        score = clamp(0.5 - corr * 0.3, 0.35, 0.65)
        out.append(Signal("image_noise", "Noise vs. brightness", score, WEAK,
                          f"Noise level {level:.2f}; correlation with brightness {corr:+.2f}. Camera sensor noise usually "
                          "rises with brightness (positive), generator noise usually doesn't.",
                          {"noise_level": round(level, 3), "brightness_corr": round(corr, 3)}))
    return out


@analyzer("image_jpeg", (IMAGE,), "Compares JPEG quantization tables with standard software tables.")
def jpeg_tables(item: Item) -> list[Signal]:
    if item.data[:3] != b"\xff\xd8\xff":
        fmt = "PNG" if item.data[:4] == b"\x89PNG" else "WebP" if item.data[8:12] == b"WEBP" else "non-JPEG"
        if fmt == "PNG" and not item.meta.get("from_video"):
            return [Signal("image_jpeg", "Lossless PNG", 0.55, WEAK,
                           "Saved as PNG. Many generators output PNG, while cameras and phones save JPEG or HEIC.")]
        raise SkipAnalyzer(f"{fmt} image has no JPEG tables")
    from PIL import Image

    im = Image.open(io.BytesIO(item.data))
    q = getattr(im, "quantization", None) or {}
    if 0 not in q:
        raise SkipAnalyzer("no quantization table")
    luma = np.array(q[0][:64], dtype=float)
    best_q, best_err = None, 1e9
    for quality in range(1, 101):
        scale = 5000 / quality if quality < 50 else 200 - 2 * quality
        t = np.clip(np.floor((IJG_LUMA * scale + 50) / 100), 1, 255)
        # Pillow stores tables in zigzag order on some versions and natural order on others; compare sorted too.
        err = min(np.abs(np.sort(t) - np.sort(luma)).mean(), np.abs(t - luma).mean())
        if err < best_err:
            best_q, best_err = quality, err
    standard = best_err < 0.5
    detail = {"closest_ijg_quality": best_q, "mean_table_difference": round(float(best_err), 2)}
    if standard:
        return [Signal("image_jpeg", "Standard software JPEG table", 0.55, WEAK,
                       f"Uses the standard libjpeg table at quality {best_q}: saved by software (an editor, a website or a "
                       "generator) rather than straight from most cameras.", detail)]
    return [Signal("image_jpeg", "Custom JPEG table", 0.4, WEAK,
                   "Uses a non-standard quantization table, which is typical of camera and phone firmware.", detail)]


@analyzer("image_dimensions", (IMAGE,), "Checks whether the size matches common generator output sizes.")
def dimensions(item: Item) -> list[Signal]:
    from PIL import Image

    try:
        w, h = Image.open(io.BytesIO(item.data)).size
    except Exception as e:
        raise SkipAnalyzer(f"could not read size: {e}")
    if item.meta.get("from_video"):
        raise SkipAnalyzer("video frame")
    if (w, h) in GENERATOR_SIZES or (h, w) in GENERATOR_SIZES:
        return [Signal("image_dimensions", "Generator-typical size", 0.62, WEAK,
                       f"{w}×{h} is a standard output size for popular image generators.", {"size": [w, h]})]
    if w % 64 == 0 and h % 64 == 0 and max(w, h) <= 2048:
        return [Signal("image_dimensions", "Multiple-of-64 size", 0.56, WEAK,
                       f"{w}×{h}: both sides are multiples of 64, as diffusion models require.", {"size": [w, h]})]
    return [Signal("image_dimensions", "Ordinary size", 0.45, WEAK, f"{w}×{h} is not a typical generator size.", {"size": [w, h]})]
