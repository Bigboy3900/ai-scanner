"""Command line: python -m scanner FILE [FILE ...]  |  python -m scanner --text "..."  |  python -m scanner serve"""

from __future__ import annotations

import argparse
import json
import sys

ICON = {"verified_ai": "!!", "likely_ai": "! ", "possibly_ai": "? ", "inconclusive": "~ ",
        "possibly_human": "· ", "likely_human": "ok", "verified_capture": "OK"}


def print_report(r: dict, indent: int = 0, verbose: bool = False) -> None:
    pad = "  " * indent
    v = r["overall"] if indent == 0 else r["verdict"]
    print(f"{pad}[{ICON[v['band']]}] {r['name']} ({r['kind']}): {v['label']}, AI probability {v['ai_probability']:.0%}")
    print(f"{pad}     {v['reason']}")
    for s in sorted(r["signals"], key=lambda s: -abs(s["score"] - 0.5)):
        if verbose or abs(s["score"] - 0.5) > 0.04:
            print(f"{pad}     - {s['name']} [{s['strength']}, {s['score']:.2f}]: {s['explanation']}")
    if verbose:
        for s in r["skipped"]:
            print(f"{pad}     (skipped {s['analyzer']}: {s['reason']})")
    for p in r["parts"]:
        print_report(p, indent + 1, verbose)
    if indent == 0:
        for c in r["caveats"]:
            print(f"     Note: {c}")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["serve"]:
        import uvicorn

        sp = argparse.ArgumentParser(prog="python -m scanner serve")
        sp.add_argument("--host", default="127.0.0.1")
        sp.add_argument("--port", type=int, default=8000)
        a = sp.parse_args(argv[1:])
        uvicorn.run("scanner.web:app", host=a.host, port=a.port)
        return 0
    ap = argparse.ArgumentParser(prog="python -m scanner", description="Check files for signs they were made by AI.")
    ap.add_argument("files", nargs="*")
    ap.add_argument("--text", help="scan this text instead of a file")
    ap.add_argument("--json", action="store_true", help="print the full JSON report")
    ap.add_argument("-v", "--verbose", action="store_true", help="show every signal and skipped analyzer")
    ap.add_argument("--list", action="store_true", help="list analyzers")
    a = ap.parse_args(argv)

    from . import scan_path, scan_text
    from .core import REGISTRY

    if a.list:
        for spec in REGISTRY:
            print(f"{spec.name:24} {','.join(spec.kinds):30} {spec.description}")
        return 0
    reports = [scan_text(a.text)] if a.text else [scan_path(f) for f in a.files]
    if not reports:
        ap.print_help()
        return 1
    for r in reports:
        if a.json:
            print(json.dumps(r, indent=2, default=str))
        else:
            print_report(r, verbose=a.verbose)
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
