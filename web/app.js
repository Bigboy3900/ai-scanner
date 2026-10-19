// Runs the Python scanner in the browser with Pyodide. Files never leave the device.
// The browser does the media decoding that ffmpeg does in the server version:
// audio samples, video frames and C2PA manifests are handed to Python ready to analyze.

const PYODIDE_VERSION = "0.27.7";
const PYODIDE_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
const VIDEO_FRAMES = 16;
const BURST_FRAMES = 24;
const MAX_AUDIO_SECONDS = 120;

let pyodide = null;
let engineReady = null;
let c2paPromise = null;

export function startEngine(onProgress) {
  engineReady ??= (async () => {
    onProgress("Loading the Python engine…");
    const { loadPyodide } = await import(`${PYODIDE_URL}pyodide.mjs`);
    pyodide = await loadPyodide({ indexURL: PYODIDE_URL });
    onProgress("Loading maths and image libraries…");
    await pyodide.loadPackage(["numpy", "scipy", "pillow", "micropip"]);
    const manifest = await (await fetch("py-files.json")).json();
    const micropip = pyodide.pyimport("micropip");
    for (const wheel of manifest.wheels) {
      await micropip.install.callKwargs(new URL(wheel, location.href).href, { deps: false });
    }
    onProgress("Loading the scanner…");
    for (const path of manifest.files) {
      const text = await (await fetch(`py/${path}`)).text();
      const dir = `/home/pyodide/${path}`.split("/").slice(0, -1).join("/");
      pyodide.FS.mkdirTree(dir);
      pyodide.FS.writeFile(`/home/pyodide/${path}`, text);
    }
    pyodide.runPython("import sys; sys.path.insert(0, '/home/pyodide')");
    pyodide.runPython("import glue");
    onProgress("");
  })();
  return engineReady;
}

async function needOpenCV() {
  await pyodide.loadPackage(["opencv-python", "pywavelets"]);
}

async function readC2pa(file) {
  try {
    c2paPromise ??= import("./vendor/c2pa.js").then(({ createC2pa }) =>
      createC2pa({ wasmSrc: new URL("vendor/c2pa_bg.wasm", location.href).href }).then((c2pa) => ({ c2pa })));
    const { c2pa } = await c2paPromise;
    const { Reader } = await import("./vendor/c2pa.js");
    const reader = await Reader.fromBlob(c2pa, file.type || guessMime(file.name), file);
    if (!reader) return { store: null, error: "ManifestNotFound" };
    const store = await reader.manifestStore();
    await reader.free?.();
    return { store, error: null };
  } catch (e) {
    return { store: null, error: String(e?.name || "") + ": " + String(e?.message || e) };
  }
}

function guessMime(name) {
  const ext = name.split(".").pop().toLowerCase();
  return {
    jpg: "image/jpeg", jpeg: "image/jpeg", png: "image/png", webp: "image/webp", gif: "image/gif", heic: "image/heic",
    avif: "image/avif", tif: "image/tiff", tiff: "image/tiff", mp4: "video/mp4", mov: "video/quicktime",
    m4a: "audio/mp4", mp3: "audio/mpeg", wav: "audio/wav", pdf: "application/pdf",
  }[ext] || "application/octet-stream";
}

async function decodeAudio(arrayBuffer) {
  // Decode at 48 kHz so high-frequency cutoffs stay visible whatever the file's own rate.
  const ctx = new OfflineAudioContext(1, 1, 48000);
  const buf = await ctx.decodeAudioData(arrayBuffer.slice(0));
  const n = Math.min(buf.length, buf.sampleRate * MAX_AUDIO_SECONDS);
  const mono = new Float32Array(n);
  for (let c = 0; c < buf.numberOfChannels; c++) {
    const ch = buf.getChannelData(c);
    for (let i = 0; i < n; i++) mono[i] += ch[i] / buf.numberOfChannels;
  }
  return { samples: mono, sr: buf.sampleRate };
}

function seek(video, t) {
  return new Promise((resolve, reject) => {
    const done = () => { video.removeEventListener("seeked", done); resolve(); };
    video.addEventListener("seeked", done);
    video.onerror = () => reject(new Error("could not decode this video in the browser"));
    video.currentTime = Math.min(t, Math.max(0, video.duration - 0.01));
  });
}

async function grab(video, maxWidth) {
  const scale = Math.min(1, maxWidth / video.videoWidth);
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(video.videoWidth * scale);
  canvas.height = Math.round(video.videoHeight * scale);
  canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
  const blob = await new Promise((r) => canvas.toBlob(r, "image/png"));
  return new Uint8Array(await blob.arrayBuffer());
}

async function decodeVideo(file, arrayBuffer, onProgress) {
  const video = document.createElement("video");
  video.muted = true;
  video.preload = "auto";
  video.src = URL.createObjectURL(file);
  await new Promise((resolve, reject) => {
    video.onloadeddata = resolve;
    video.onerror = () => reject(new Error("this browser can't decode this video format"));
  });
  const duration = video.duration;
  const out = { frames: [], burst: [] };
  const step = duration / VIDEO_FRAMES;
  for (let i = 0; i < VIDEO_FRAMES; i += 4) {
    onProgress(`Extracting video frames (${i / 4 + 1}/4)…`);
    const t = i * step;
    await seek(video, t);
    out.frames.push([t, await grab(video, 1920)]);
  }
  onProgress("Extracting consecutive frames…");
  const start = Math.max(0, duration / 2 - 1);
  for (let i = 0; i < BURST_FRAMES; i++) {
    await seek(video, start + i / 30);
    out.burst.push(await grab(video, 640));
  }
  let audio = null;
  try { audio = await decodeAudio(arrayBuffer); } catch { /* no audio track */ }
  if (audio) Object.assign(out, audio);
  out.probe = JSON.stringify({
    format: { duration: String(duration), tags: containerStrings(new Uint8Array(arrayBuffer)) },
    streams: [{ codec_type: "video", width: video.videoWidth, height: video.videoHeight }, ...(audio ? [{ codec_type: "audio" }] : [])],
  });
  URL.revokeObjectURL(video.src);
  return out;
}

// Pull readable text out of MP4/MOV metadata boxes (udta/meta/ilst), standing in for ffprobe's tags.
function containerStrings(bytes) {
  const tags = {};
  const find = (needle, from = 0) => {
    outer: for (let i = from; i < bytes.length - needle.length; i++) {
      for (let j = 0; j < needle.length; j++) if (bytes[i + j] !== needle.charCodeAt(j)) continue outer;
      return i;
    }
    return -1;
  };
  const udta = find("udta");
  if (udta < 0) return tags;
  const size = (bytes[udta - 4] << 24 | bytes[udta - 3] << 16 | bytes[udta - 2] << 8 | bytes[udta - 1]) >>> 0;
  const region = bytes.subarray(udta, Math.min(bytes.length, udta + Math.min(size, 1 << 20)));
  let cur = "", n = 0;
  for (const b of region) {
    if (b >= 32 && b < 127) cur += String.fromCharCode(b);
    else { if (cur.length >= 6) tags[`udta_${n++}`] = cur; cur = ""; }
  }
  if (cur.length >= 6) tags[`udta_${n++}`] = cur;
  for (const k of ["com.apple.quicktime.make", "com.apple.quicktime.model", "com.android.version"]) {
    if (Object.values(tags).some((v) => v.includes(k))) tags[k] = "present";
  }
  return tags;
}

export async function scanFile(file, onProgress) {
  await startEngine(onProgress);
  const arrayBuffer = await file.arrayBuffer();
  const bytes = new Uint8Array(arrayBuffer);
  const glue = pyodide.pyimport("glue");
  const kind = glue.kind_of(bytes.subarray(0, 64), file.name);
  let decoded = null;
  if (kind !== "text") {
    decoded = {};
    onProgress("Checking Content Credentials…");
    decoded.c2pa = JSON.stringify(await readC2pa(file));
    if (kind === "image" || kind === "video" || kind === "document") {
      onProgress("Loading image forensics…");
      await needOpenCV();
    }
    if (kind === "audio") {
      onProgress("Decoding audio…");
      try { Object.assign(decoded, await decodeAudio(arrayBuffer)); } catch { /* reported as skipped */ }
    }
    if (kind === "video") {
      try { Object.assign(decoded, await decodeVideo(file, arrayBuffer, onProgress)); }
      catch (e) {
        decoded.probe = JSON.stringify({ format: { tags: containerStrings(bytes) }, streams: [] });
        decoded.error = `Your browser couldn't decode this video (${e.message}), so frame and soundtrack checks didn't run. Try Chrome or Edge, or convert it to MP4.`;
      }
    }
  }
  onProgress("Analyzing…");
  await new Promise((r) => setTimeout(r, 30)); // let the status repaint before Python blocks the page
  return JSON.parse(glue.scan_for_js(bytes, file.name, decoded));
}

export async function scanText(text, onProgress) {
  await startEngine(onProgress);
  const glue = pyodide.pyimport("glue");
  onProgress("Analyzing…");
  await new Promise((r) => setTimeout(r, 30));
  return JSON.parse(glue.scan_for_js(new TextEncoder().encode(text), "pasted text.txt", null));
}
