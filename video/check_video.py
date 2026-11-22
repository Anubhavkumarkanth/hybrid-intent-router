"""Check video/tutorial.mp4 (make video-check). Exit 1 on any failure.

- ffprobe: duration 10-20 minutes, one video and one audio stream, 1920x1080, yuv420p
- audio not silent: mean volume above -35 dB (ffmpeg volumedetect)
- subtitles and chapters exist
- one frame from the middle of every scene -> video/build/frames/NN.png (960 px wide) for review
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "video"
MP4 = VIDEO / "tutorial.mp4"
FRAMES = VIDEO / "build" / "frames"


def sh(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def main() -> int:
    problems = []
    if not MP4.exists():
        print("FAIL: video/tutorial.mp4 missing (run make video)")
        return 1
    info = json.loads(sh(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(MP4)]).stdout)
    dur = float(info["format"]["duration"])
    kinds = [s["codec_type"] for s in info["streams"]]
    v = next((s for s in info["streams"] if s["codec_type"] == "video"), {})
    print(f"duration {dur / 60:.2f} min; streams {kinds}; video {v.get('codec_name')} {v.get('width')}x"
          f"{v.get('height')} {v.get('pix_fmt')} {v.get('r_frame_rate')}")
    if not 600 <= dur <= 1200:
        problems.append(f"duration {dur / 60:.2f} min is outside 10-20 min")
    if kinds.count("video") != 1 or kinds.count("audio") != 1:
        problems.append(f"expected 1 video + 1 audio stream, got {kinds}")
    if (v.get("width"), v.get("height"), v.get("pix_fmt")) != (1920, 1080, "yuv420p"):
        problems.append("video is not 1920x1080 yuv420p")

    vol = sh(["ffmpeg", "-hide_banner", "-i", str(MP4), "-vn", "-af", "volumedetect", "-f", "null", "-"]).stderr
    m = re.search(r"mean_volume:\s*(-?[\d.]+) dB", vol)
    mean = float(m.group(1)) if m else -999.0
    print(f"audio mean volume {mean} dB")
    if mean <= -35:
        problems.append(f"audio too quiet / silent (mean {mean} dB)")

    for f in ("tutorial.srt", "chapters.md"):
        if not (VIDEO / f).exists() or not (VIDEO / f).read_text().strip():
            problems.append(f"video/{f} missing or empty")

    timings = json.loads((VIDEO / "build" / "timings.json").read_text())
    FRAMES.mkdir(parents=True, exist_ok=True)
    for old in FRAMES.glob("*.png"):
        old.unlink()
    for sc in timings:
        mid = (sc["start"] + sc["end"]) / 2
        out = FRAMES / f"{sc['scene']:02d}.png"
        r = sh(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{mid:.2f}", "-i", str(MP4), "-frames:v", "1",
                "-vf", "scale=960:-2", str(out)])
        if r.returncode != 0 or not out.exists():
            problems.append(f"could not extract frame for scene {sc['scene']}: {r.stderr[-300:]}")
    print(f"extracted {len(list(FRAMES.glob('*.png')))} frames to {FRAMES.relative_to(ROOT)} "
          f"({len(timings)} scenes)")

    if problems:
        print("FAIL:\n- " + "\n- ".join(problems))
        return 1
    print("video-check: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
