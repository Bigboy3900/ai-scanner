"""Assemble the static GitHub Pages site (browser version) into _site/.

    python tools/build_site.py [--out _site]

Copies web/, the scanner package's Python files, and the pure-Python wheels Pyodide
installs at startup. Pyodide itself and numpy/scipy/pillow/opencv load from its CDN.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WHEELS = ["pypdf==6.1.1", "invisible-watermark==0.2.0"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "_site"))
    out = Path(ap.parse_args().out)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(ROOT / "web", out, ignore=shutil.ignore_patterns("glue.py"))

    files = []
    for src in sorted((ROOT / "scanner").rglob("*.py")):
        rel = src.relative_to(ROOT).as_posix()
        if rel in ("scanner/web.py", "scanner/__main__.py"):
            continue  # server and CLI only
        files.append(rel)
    files += sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "scanner" / "models").glob("*.json"))
    files.append("glue.py")
    for rel in files:
        src = ROOT / "web" / rel if rel == "glue.py" else ROOT / rel
        dst = out / "py" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    wheel_dir = out / "wheels"
    wheel_dir.mkdir()
    subprocess.run([sys.executable, "-m", "pip", "download", "--no-deps", "--only-binary=:all:", "-q",
                    "-d", str(wheel_dir), *WHEELS], check=True)
    wheels = sorted(f"wheels/{w.name}" for w in wheel_dir.glob("*.whl"))
    (out / "py-files.json").write_text(json.dumps({"files": files, "wheels": wheels}, indent=1))
    (out / ".nojekyll").write_text("")
    print(f"Built {out} ({len(files)} Python files, {len(wheels)} wheels)")


if __name__ == "__main__":
    main()
