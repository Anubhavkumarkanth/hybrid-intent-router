"""Render build step: download the small ONNX model from its Hugging Face model repo and check every file's
SHA-256 against deploy/render/model_files.json (pinned to one commit), so the server runs exactly that model.
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = json.loads((Path(__file__).with_name("model_files.json")).read_text())


def fetch(name: str, sha256: str, out_dir: Path) -> None:
    target = out_dir / name
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == sha256:
        print(f"{name}: already present")
        return
    url = f"https://huggingface.co/{SPEC['repo']}/resolve/{SPEC['revision']}/{name}"
    tmp = target.with_suffix(target.suffix + ".part")
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as f:
        while chunk := resp.read(1 << 20):
            digest.update(chunk)
            f.write(chunk)
    if digest.hexdigest() != sha256:
        tmp.unlink()
        sys.exit(f"{name}: SHA-256 mismatch, refusing to use it")
    tmp.replace(target)
    print(f"{name}: {target.stat().st_size / 1e6:.1f} MB, SHA-256 ok")


def main() -> None:
    out_dir = ROOT / SPEC["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, sha256 in SPEC["files"].items():
        fetch(name, sha256, out_dir)


if __name__ == "__main__":
    main()
