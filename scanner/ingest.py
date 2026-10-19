"""Work out what a file is and break containers into analyzable parts."""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

from .core import AUDIO, DOCUMENT, IMAGE, TEXT, UNKNOWN, VIDEO, Item

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".heic", ".heif", ".avif"}
AUDIO_EXT = {".mp3", ".wav", ".flac", ".ogg", ".oga", ".m4a", ".aac", ".opus", ".wma", ".aiff", ".aif"}
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi", ".wmv", ".flv", ".3gp"}
TEXT_EXT = {".txt", ".md", ".markdown", ".rst", ".csv", ".json", ".html", ".htm", ".xml", ".rtf", ".tex", ".srt", ".vtt"}
DOC_EXT = {".pdf", ".docx", ".pptx", ".xlsx", ".odt", ".odp", ".epub"}

MAX_VIDEO_FRAMES = 16


def sniff(data: bytes, name: str) -> str:
    head = data[:64]
    ext = Path(name).suffix.lower()
    if head.startswith(b"%PDF"):
        return DOCUMENT
    if head.startswith((b"\xff\xd8\xff", b"\x89PNG", b"GIF8", b"BM")) or head[8:12] in (b"WEBP",):
        return IMAGE
    if head[4:8] == b"ftyp":
        brand = head[8:12]
        if brand in (b"heic", b"heix", b"mif1", b"avif", b"msf1"):
            return IMAGE
        if brand in (b"M4A ", b"M4B ") or ext in AUDIO_EXT:
            return AUDIO
        return VIDEO
    if head.startswith((b"ID3", b"fLaC", b"OggS")) or head[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return VIDEO if ext in VIDEO_EXT else AUDIO
    if head.startswith(b"RIFF"):
        return {b"WAVE": AUDIO, b"AVI ": VIDEO, b"WEBP": IMAGE}.get(head[8:12], UNKNOWN)
    if head.startswith(b"\x1a\x45\xdf\xa3"):  # Matroska / WebM
        return AUDIO if ext in AUDIO_EXT else VIDEO
    if head.startswith(b"PK"):
        return DOCUMENT
    for kind, exts in ((IMAGE, IMAGE_EXT), (AUDIO, AUDIO_EXT), (VIDEO, VIDEO_EXT), (DOCUMENT, DOC_EXT), (TEXT, TEXT_EXT)):
        if ext in exts:
            return kind
    try:
        data[:4096].decode("utf-8")
        return TEXT
    except UnicodeDecodeError:
        return UNKNOWN


def html_to_text(s: str) -> str:
    s = re.sub(r"(?is)<(script|style).*?</\1>", " ", s)
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    return re.sub(r"[ \t]+", " ", s)


def build_item(data: bytes, name: str, workdir: str, decoded: dict | None = None) -> Item:
    """decoded: media already decoded by the caller (the browser version has no ffmpeg).
    Keys: samples/sr for audio; probe, frames [(t, png)], burst [png], samples/sr for video."""
    kind = sniff(data, name)
    path = os.path.join(workdir, re.sub(r"[^\w.\-]", "_", Path(name).name) or "upload")
    with open(path, "wb") as f:
        f.write(data)
    item = Item(kind=kind, name=name, data=data, path=path)
    if decoded and "c2pa" in decoded:
        item.meta["c2pa"] = decoded["c2pa"]
    if kind == TEXT:
        text = data.decode("utf-8", errors="replace")
        if Path(name).suffix.lower() in (".html", ".htm") or text.lstrip().lower().startswith(("<!doctype html", "<html")):
            text = html_to_text(text)
        item.text = text
    elif kind == DOCUMENT:
        _expand_document(item, workdir)
    elif kind == AUDIO and decoded and decoded.get("samples") is not None:
        item.meta["samples"], item.meta["sr"] = decoded["samples"], decoded["sr"]
    elif kind == VIDEO and decoded:
        _expand_decoded_video(item, decoded)
    elif kind == VIDEO:
        _expand_video(item, workdir)
    return item


# ---------------------------------------------------------------- documents

def _expand_document(item: Item, workdir: str) -> None:
    if item.data.startswith(b"%PDF"):
        _expand_pdf(item)
    elif item.data.startswith(b"PK"):
        _expand_office_zip(item)


def _expand_pdf(item: Item) -> None:
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(item.data))
    except Exception as e:  # corrupt or encrypted
        item.meta["error"] = f"Could not open PDF: {e}"
        return
    info = {}
    try:
        for k, v in (reader.metadata or {}).items():
            info[str(k).lstrip("/")] = str(v)
    except Exception:
        pass
    item.meta["doc_info"] = info
    item.meta["pages"] = len(reader.pages)
    try:
        xmp = reader.xmp_metadata
        if xmp is not None:
            item.meta["xmp_raw"] = xmp.stream.get_data().decode("utf-8", "replace")
    except Exception:
        pass
    texts = []
    n_img = 0
    for i, page in enumerate(reader.pages):
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            pass
        try:
            for img in page.images:
                if n_img >= 20:
                    break
                n_img += 1
                item.children.append(Item(kind=IMAGE, name=f"{item.name} › page {i + 1} › {img.name}", data=img.data))
        except Exception:
            pass
    text = "\n".join(texts).strip()
    if text:
        item.children.insert(0, Item(kind=TEXT, name=f"{item.name} › text", text=text))


def _expand_office_zip(item: Item) -> None:
    try:
        z = zipfile.ZipFile(io.BytesIO(item.data))
    except zipfile.BadZipFile:
        return
    names = z.namelist()
    props = {}
    for meta_file in ("docProps/core.xml", "docProps/app.xml", "meta.xml"):
        if meta_file in names:
            xml = z.read(meta_file).decode("utf-8", "replace")
            for tag, val in re.findall(r"<(?:\w+:)?(\w+)[^>]*>([^<]{1,200})</", xml):
                props[tag] = val
    item.meta["doc_info"] = props
    if "word/document.xml" in names:
        xml = z.read("word/document.xml").decode("utf-8", "replace")
        item.meta["word_revisions"] = len(set(re.findall(r'w:rsidR="([0-9A-F]+)"', xml)))
        paras = re.findall(r"<w:p[ >].*?</w:p>", xml, flags=re.S)
        text = "\n".join("".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", p)) for p in paras)
    else:
        parts = sorted(n for n in names if re.match(r"(ppt/slides/slide\d+|xl/sharedStrings|content|OEBPS/.*)\.x?html?|.*\.xml$", n)
                       and not n.startswith(("docProps", "_rels", "[Content")))
        chunks = []
        for n in parts:
            xml = z.read(n).decode("utf-8", "replace")
            chunks.append(" ".join(re.findall(r"<(?:a:t|t|text:p|text:span|p)[^>]*>([^<]+)<", xml)))
        text = "\n".join(c for c in chunks if c.strip())
    text = text.strip()
    if text:
        item.children.append(Item(kind=TEXT, name=f"{item.name} › text", text=text))
    media = [n for n in names if re.search(r"(media|Pictures|images?)/.+\.(png|jpe?g|webp|gif|bmp|tiff?)$", n, re.I)]
    for n in media[:20]:
        item.children.append(Item(kind=IMAGE, name=f"{item.name} › {n}", data=z.read(n)))


# ---------------------------------------------------------------- video

def ffprobe(path: str) -> dict:
    if not shutil.which("ffprobe"):
        return {}
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", path],
            capture_output=True, timeout=60, check=True,
        ).stdout
        return json.loads(out)
    except Exception:
        return {}


def _expand_video(item: Item, workdir: str) -> None:
    probe = ffprobe(item.path)
    item.meta["probe"] = probe
    if not shutil.which("ffmpeg"):
        item.meta["error"] = "ffmpeg is not installed, so frames and audio could not be extracted."
        return
    duration = float(probe.get("format", {}).get("duration") or 0) or 10.0
    fps = min(2.0, MAX_VIDEO_FRAMES / max(duration, 0.1))
    frame_dir = tempfile.mkdtemp(dir=workdir, prefix="frames_")
    subprocess.run(
        ["ffmpeg", "-v", "quiet", "-i", item.path, "-vf", f"fps={fps:.4f}", "-frames:v", str(MAX_VIDEO_FRAMES),
         os.path.join(frame_dir, "f%03d.png")],
        timeout=300,
    )
    frames = sorted(Path(frame_dir).glob("f*.png"))
    item.meta["frame_paths"] = [str(p) for p in frames]
    item.meta["frame_times"] = [i / fps for i in range(len(frames))]
    # A few evenly spaced frames also go through the full image analyzers.
    picks = frames[:: max(1, len(frames) // 4)][:4]
    for p in picks:
        t = frames.index(p) / fps
        item.children.append(Item(kind=IMAGE, name=f"{item.name} › frame at {t:.1f}s", data=p.read_bytes(),
                                  meta={"from_video": True}))
    if any(s.get("codec_type") == "audio" for s in probe.get("streams", [])):
        wav = os.path.join(workdir, "audio.wav")
        subprocess.run(["ffmpeg", "-v", "quiet", "-y", "-i", item.path, "-vn", "-ac", "1", "-t", "120", wav], timeout=300)
        if os.path.exists(wav):
            item.children.append(Item(kind=AUDIO, name=f"{item.name} › audio track", data=Path(wav).read_bytes(),
                                      path=wav, meta={"from_video": True}))


def _expand_decoded_video(item: Item, decoded: dict) -> None:
    item.meta["probe"] = decoded.get("probe") or {}
    if decoded.get("error"):
        item.meta["error"] = decoded["error"]
    item.meta["burst_pngs"] = decoded.get("burst") or []
    frames = decoded.get("frames") or []
    for t, png in frames[:: max(1, len(frames) // 4)][:4]:
        item.children.append(Item(kind=IMAGE, name=f"{item.name} › frame at {t:.1f}s", data=png, meta={"from_video": True}))
    if decoded.get("samples") is not None:
        item.children.append(Item(kind=AUDIO, name=f"{item.name} › audio track",
                                  meta={"from_video": True, "samples": decoded["samples"], "sr": decoded["sr"]}))
