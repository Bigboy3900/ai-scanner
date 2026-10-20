"""Bridge between the browser page (app.js) and the scanner package, run inside Pyodide."""

import json

import numpy as np

from scanner import scan_bytes
from scanner.ingest import sniff


def _bytes(js):
    return bytes(js.to_py()) if hasattr(js, "to_py") else bytes(js)


def kind_of(head_js, name):
    return sniff(_bytes(head_js), name)


def scan_for_js(data_js, name, decoded_js=None):
    decoded = None
    if decoded_js is not None:
        d = decoded_js.to_py()
        decoded = {}
        if "c2pa" in d:
            decoded["c2pa"] = json.loads(d["c2pa"]) if isinstance(d["c2pa"], str) else d["c2pa"]
        if d.get("samples") is not None:
            decoded["samples"] = np.frombuffer(d["samples"], dtype=np.float32).astype(np.float64)
            decoded["sr"] = int(d["sr"])
        if d.get("probe"):
            decoded["probe"] = json.loads(d["probe"])
        if d.get("frames"):
            decoded["frames"] = [(float(t), bytes(png)) for t, png in d["frames"]]
        if d.get("error"):
            decoded["error"] = d["error"]
        if d.get("burst"):
            decoded["burst"] = [bytes(png) for png in d["burst"]]
    report = scan_bytes(_bytes(data_js), name, decoded=decoded)
    _hide_unavailable(report)
    return json.dumps(report, default=str)


def _hide_unavailable(report):
    """Local models and paid APIs can't run in the browser at all; don't list them as skipped checks."""
    report["skipped"] = [s for s in report["skipped"] if not s["analyzer"].startswith(("ml_", "api_"))]
    for part in report["parts"]:
        _hide_unavailable(part)
