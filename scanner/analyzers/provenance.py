"""Provenance checks: C2PA content credentials, watermarks, and metadata left by generators."""

from __future__ import annotations

import io
import json
import re

from ..core import AUDIO, DOCUMENT, IMAGE, MODERATE, STRONG, VIDEO, WEAK, Item, Signal, SkipAnalyzer, analyzer
from ..ingest import ffprobe

BINARY_KINDS = (IMAGE, AUDIO, VIDEO, DOCUMENT)

# Names of generators and AI tools that show up in metadata fields.
GENERATOR_PATTERNS = [
    r"stable[ _-]?diffusion", r"\bsdxl\b", r"comfyui", r"automatic1111", r"invokeai", r"fooocus", r"novelai",
    r"midjourney", r"dall[·\-\s]?e", r"\bopenai\b", r"chatgpt", r"gpt-4o", r"gpt-image", r"\bsora\b",
    r"adobe firefly", r"\bfirefly\b", r"imagen", r"gemini", r"\bveo\b", r"nano banana", r"synthid",
    r"leonardo\.ai", r"ideogram", r"\bflux(\.1)?\b", r"black forest labs", r"playground ?ai", r"\bgrok\b", r"aurora",
    r"bing image creator", r"microsoft designer", r"copilot", r"\brunway(ml)?\b", r"\bpika\b", r"\bkling\b",
    r"\bluma\b", r"dream machine", r"hailuo", r"minimax", r"\bsuno\b", r"\budio\b", r"elevenlabs", r"\bmurf\b",
    r"\bheygen\b", r"synthesia", r"d-id", r"deepfacelab", r"faceswap", r"\breplicate\b", r"civitai", r"tensor\.art",
    r"krea", r"artbreeder", r"nightcafe", r"craiyon", r"wombo", r"\bdream by wombo\b", r"lensa", r"canva magic",
]
GEN_RE = re.compile("|".join(GENERATOR_PATTERNS), re.I)
SD_PARAMS_RE = re.compile(r"(Steps: \d+|Sampler: |CFG scale: |Seed: \d+|Negative prompt:|\"class_type\"|KSampler)")
AI_SOURCE_TYPES = ("trainedAlgorithmicMedia", "compositeWithTrainedAlgorithmicMedia", "algorithmicMedia")


def collect_metadata(item: Item) -> dict[str, str]:
    """Pull every textual metadata field we can find into one flat dict."""
    fields: dict[str, str] = {}
    if item.kind == IMAGE and item.data:
        try:
            from PIL import ExifTags, Image

            im = Image.open(io.BytesIO(item.data))
            for k, v in (im.info or {}).items():
                if isinstance(v, (str, bytes)) and k not in ("icc_profile", "exif"):
                    fields[f"info:{k}"] = v.decode("utf-8", "replace") if isinstance(v, bytes) else v
            exif = im.getexif()
            for tag, val in exif.items():
                fields[f"exif:{ExifTags.TAGS.get(tag, tag)}"] = _s(val)
            for ifd_id in (ExifTags.IFD.Exif, ExifTags.IFD.GPSInfo):
                try:
                    for tag, val in exif.get_ifd(ifd_id).items():
                        fields[f"exif:{ExifTags.TAGS.get(tag, tag)}"] = _s(val)
                except Exception:
                    pass
        except Exception:
            pass
    xmp = _find_xmp(item.data)
    if xmp:
        fields["xmp"] = xmp
    if item.meta.get("xmp_raw"):
        fields["xmp"] = item.meta["xmp_raw"]
    for k, v in (item.meta.get("doc_info") or {}).items():
        fields[f"doc:{k}"] = str(v)
    if item.kind in (AUDIO, VIDEO) and item.path:
        probe = item.meta.get("probe") or ffprobe(item.path)
        item.meta["probe"] = probe
        for k, v in (probe.get("format", {}).get("tags") or {}).items():
            fields[f"tag:{k}"] = str(v)
        for i, st in enumerate(probe.get("streams", [])):
            for k, v in (st.get("tags") or {}).items():
                fields[f"stream{i}:{k}"] = str(v)
    return fields


def _s(v) -> str:
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace").strip("\x00")
    return str(v)


def _find_xmp(data: bytes) -> str | None:
    start = data.find(b"<x:xmpmeta")
    if start < 0:
        start = data.find(b"<?xpacket begin")
    if start < 0:
        return None
    end = data.find(b"</x:xmpmeta>", start)
    if end < 0:
        end = start + 65536
    return data[start:end + 12].decode("utf-8", "replace")


@analyzer("c2pa", BINARY_KINDS, "Reads and validates C2PA Content Credentials (signed provenance manifests).")
def c2pa_check(item: Item) -> list[Signal]:
    if item.meta.get("from_video"):
        raise SkipAnalyzer("checked on the whole video instead")
    has_marker = b"c2pa" in item.data or b"jumb" in item.data[:2_000_000]
    if "c2pa" in item.meta:
        # Already read by the caller (the browser version uses the c2pa-web library).
        pre = item.meta["c2pa"] or {}
        store, error = pre.get("store"), pre.get("error")
    else:
        store, error = _read_c2pa(item, has_marker)
    if not store:
        if has_marker and error and "notfound" not in error.lower().replace(" ", ""):
            return [Signal("c2pa", "C2PA data unreadable", 0.5, WEAK,
                           f"Something that looks like Content Credentials is present but could not be read ({error}).")]
        return [Signal("c2pa", "No Content Credentials", 0.5, WEAK,
                       "No C2PA manifest. This says nothing either way: most tools don't add one and it is easily stripped.")]

    active = store.get("manifests", {}).get(store.get("active_manifest"), {})
    validation = store.get("validation_status") or []
    errors = [v for v in validation if "untrusted" not in v.get("code", "") and v.get("code", "").endswith(
        ("mismatch", "invalid", "missing", "revoked", "expired", "malformed"))]
    untrusted = any("untrusted" in v.get("code", "") for v in validation)
    blob = json.dumps(store)
    ai_types = [t for t in AI_SOURCE_TYPES if t in blob]
    generator = active.get("claim_generator") or ", ".join(
        g.get("name", "") for g in active.get("claim_generator_info", []) if isinstance(g, dict))
    issuer = (active.get("signature_info") or {}).get("issuer", "unknown issuer")
    details = {"claim_generator": generator, "issuer": issuer, "ai_source_types": ai_types,
               "validation_errors": [v.get("code") for v in errors], "untrusted_signer": untrusted}

    if errors:
        return [Signal("c2pa", "Content Credentials failed validation", 0.6 if ai_types else 0.55, MODERATE,
                       f"A C2PA manifest is present but invalid ({', '.join(details['validation_errors'])}); "
                       "the file was changed after signing or the manifest was forged.", details)]
    if ai_types:
        note = " (signer is not on the trust list)" if untrusted else ""
        return [Signal("c2pa", "Content Credentials say AI-generated", 0.99 if not untrusted else 0.9, STRONG,
                       f"Signed by {issuer}{note}; the manifest declares {', '.join(ai_types)}. Generator: {generator or 'not named'}.",
                       details, verdict=None if untrusted else "verified_ai")]
    capture = "digitalCapture" in blob
    if capture and not untrusted:
        return [Signal("c2pa", "Content Credentials say camera capture", 0.05, STRONG,
                       f"Signed capture by {issuer} ({generator}). The manifest records a real camera capture with no AI steps.",
                       details, verdict="verified_capture")]
    return [Signal("c2pa", "Content Credentials present", 0.4, MODERATE,
                   f"Signed by {issuer} using {generator or 'an unnamed tool'}; no AI generation step is declared.", details)]


def _read_c2pa(item: Item, has_marker: bool) -> tuple[dict | None, str | None]:
    try:
        import c2pa
    except ImportError:
        if has_marker:
            return None, "no C2PA reader is available for this part"
        raise SkipAnalyzer("no C2PA reader available for this part (c2pa-python not installed)")
    if not item.path:
        raise SkipAnalyzer("no file on disk")
    try:
        with c2pa.Reader(item.path) as reader:
            return json.loads(reader.json()), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


@analyzer("metadata", BINARY_KINDS, "Looks for generator names, prompts and IPTC AI markers in metadata.")
def metadata_check(item: Item) -> list[Signal]:
    fields = collect_metadata(item)
    item.meta["metadata_fields"] = {k: v[:300] for k, v in fields.items()}
    out: list[Signal] = []
    blob = "\n".join(f"{k}={v}" for k, v in fields.items())

    src = [t for t in AI_SOURCE_TYPES if t in blob]
    if src:
        out.append(Signal("metadata", "IPTC: AI-generated source type", 0.92, MODERATE,
                          f"The IPTC digital source type is set to {src[0]}. Generators set this field; it can be removed but is rarely added falsely."))
    if SD_PARAMS_RE.search(blob):
        m = SD_PARAMS_RE.search(blob)
        out.append(Signal("metadata", "Generation parameters embedded", 0.97, STRONG,
                          "The file contains image-generation settings (prompt, sampler, seed, steps) written by Stable Diffusion-style tools.",
                          {"match": m.group(0)}, verdict="verified_ai"))
    hits = {}
    for k, v in fields.items():
        if k == "xmp":
            # Only look at XMP values that describe tools, not arbitrary descriptions.
            for tool in re.findall(r"(?:CreatorTool|softwareAgent|Software|Producer)[\"'=>\s]+([^\"'<]{1,120})", v):
                for m in GEN_RE.finditer(tool):
                    hits[m.group(0)] = f"xmp tool: {tool.strip()}"
            continue
        for m in GEN_RE.finditer(v):
            hits.setdefault(m.group(0), f"{k}: {v[:120]}")
    if hits:
        out.append(Signal("metadata", "AI tool named in metadata", 0.85, MODERATE,
                          f"Metadata mentions {', '.join(sorted(set(h.lower() for h in hits)))}.", {"where": hits}))

    if item.kind == IMAGE:
        make = fields.get("exif:Make", "").strip()
        model = fields.get("exif:Model", "").strip()
        camera_fields = [f for f in ("exif:ExposureTime", "exif:FNumber", "exif:ISOSpeedRatings", "exif:FocalLength",
                                     "exif:LensModel", "exif:GPSLatitude") if f in fields]
        if make and len(camera_fields) >= 2:
            out.append(Signal("metadata", "Camera EXIF present", 0.3, WEAK,
                              f"EXIF says it came from a {make} {model} with exposure settings. Real cameras write this, but it can be copied or faked.",
                              {"make": make, "model": model}))
        elif not fields and not item.meta.get("from_video"):
            out.append(Signal("metadata", "No metadata at all", 0.55, WEAK,
                              "The image has no metadata. Generators and social media uploads both strip it, so this is only a slight hint."))
        sw = fields.get("exif:Software") or fields.get("info:Software")
        if sw and not hits:
            out.append(Signal("metadata", "Editing software noted", 0.5, WEAK, f"Software field: {sw}.", {"software": sw}))
    if not out:
        out.append(Signal("metadata", "No AI markers in metadata", 0.45, WEAK,
                          "No generator names, prompts or AI source tags found. Metadata is easy to strip, so this means little."))
    return out


def _import_watermark_decoder():
    """invisible-watermark imports torch at module level, only for a decoder we don't use.
    Without PyTorch (e.g. in the browser) a placeholder module lets the rest load."""
    import importlib.util
    import sys
    import types

    stub = importlib.util.find_spec("torch") is None
    if stub:
        sys.modules["torch"] = types.ModuleType("torch")
    try:
        from imwatermark import WatermarkDecoder
    finally:
        if stub:
            del sys.modules["torch"]
    return WatermarkDecoder


# SDXL / SD 2 invisible watermark (dwtDct, 48 bits) and SD 1.x ("StableDiffusionV1", 136 bits).
SDXL_BITS = [int(b) for b in bin(0b101100111110110010010000011110111011000110011110)[2:]]


@analyzer("invisible_watermark", (IMAGE,), "Decodes the open-source invisible watermarks Stable Diffusion adds by default.")
def invisible_watermark_check(item: Item) -> list[Signal]:
    try:
        import cv2
        import numpy as np
        WatermarkDecoder = _import_watermark_decoder()
    except ImportError:
        raise SkipAnalyzer("invisible-watermark not installed")
    arr = cv2.imdecode(np.frombuffer(item.data, np.uint8), cv2.IMREAD_COLOR)
    if arr is None or min(arr.shape[:2]) < 256:
        raise SkipAnalyzer("image too small for watermark decoding")
    results = {}
    try:
        bits = WatermarkDecoder("bits", 48).decode(arr, "dwtDct")
        results["sdxl"] = sum(int(a) == b for a, b in zip(bits, SDXL_BITS)) / 48
    except Exception:
        pass
    try:
        raw = WatermarkDecoder("bytes", 136).decode(arr, "dwtDct")
        target = b"StableDiffusionV1"
        results["sd1"] = sum(a == b for a, b in zip(raw, target)) / len(target)
    except Exception:
        pass
    best = max(results.items(), key=lambda kv: kv[1], default=(None, 0))
    if best[1] >= 0.9:
        return [Signal("invisible_watermark", "Stable Diffusion watermark found", 0.98, STRONG,
                       f"Decoded the {best[0].upper()} invisible watermark ({best[1]:.0%} of bits match).", results,
                       verdict="verified_ai")]
    return [Signal("invisible_watermark", "No open-source watermark", 0.5, WEAK,
                   "No Stable Diffusion watermark detected. Most generators don't use this one, and resizing or recompression removes it.",
                   results)]
