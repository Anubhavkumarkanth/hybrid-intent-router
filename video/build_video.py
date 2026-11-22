"""Build video/tutorial.mp4 from video/script.md.

Pipeline per scene: Pillow slide (1920x1080) + Piper narration (WAV) -> ffmpeg clip (H.264 + AAC,
yuv420p, 30 fps). Clips are joined with the ffmpeg concat demuxer. Also writes
video/tutorial.srt (subtitles) and video/chapters.md (time stamps).

script.md format, one scene per "## " heading:
    ## Slide title
    - bullet (up to 5)
    visual: image results/charts/x.png | code src/file.py 10-30 | output logs/x.log 12-20 | output logs/x.log tail 8
    > narration line (several "> " lines are joined)

Numbers: write {{file.json:key.path|fmt}} instead of a number. It is filled from results/, so every
number in the video comes from a results file. fmt: pct pp ms usd int f1 f2 f3 mb raw.
Pronunciation: narration is written normally; PRONOUNCE swaps words (PyTorch -> "Pie Torch") before speech.

    python video/build_video.py            # build (reuses unchanged audio and clips)
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import urllib.request
import wave
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from router.fonts import font_for, font_for_text  # noqa: E402

VIDEO = ROOT / "video"
BUILD = VIDEO / "build"
VOICES = VIDEO / "voices"
VOICE = "en_US-lessac-medium"
VOICE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/" + VOICE
W, H = 1920, 1080
BG, INK, MUTED, ACCENT, PANEL = "#f6f7f9", "#1d2330", "#5b6475", "#2a6fdb", "#ffffff"
TERM_BG, TERM_INK = "#1f2430", "#e3e7ee"
PAUSE_S = 0.7  # silence after each scene

PRONOUNCE = [  # (written, spoken), whole words, applied in order
    ("PyTorch", "Pie Torch"), ("NGINX", "engine X"), ("APIs", "A P Is"), ("API", "A P I"),
    ("LLMs", "L L Ms"), ("LLM", "L L M"), ("JSONL", "Jason L"), ("JSON", "Jason"), ("CPU", "C P U"),
    ("GPU", "G P U"), ("MuRIL", "Mew Rill"), ("MASSIVE", "Massive"), ("ECE", "E C E"), ("WSL2", "W S L 2"),
    ("WSL", "W S L"), ("HTTP", "H T T P"), ("p95", "P 95"), ("p50", "P 50"), ("gunicorn", "G unicorn"),
    ("XLM-R", "X L M R"), ("AdamW", "Adam W"), ("NLL", "N L L"), ("fp16", "F P 16"), ("RTX", "R T X"),
    ("SHA-256", "shaw 256"), ("README", "read me"), ("CSV", "C S V"), ("pytest", "pie test"),
    ("venv", "V env"), ("ffmpeg", "F F mpeg"), ("softmax", "soft max"), ("dev_sub", "dev sub"),
    ("test_sub", "test sub"), ("en-US", "English"), ("hi-IN", "Hindi"), ("kn-IN", "Kannada"),
    ("ruff", "ruff"), ("Makefile", "make file"), ("CLS", "C L S"), ("URL", "U R L"), ("HF", "Hugging Face"),
    ("Q&A", "Q and A"), ("e.g.", "for example"), ("vs", "versus"), ("ms", "milliseconds"),
]


# ------------------------------------------------------------------ script parsing

@dataclass
class Scene:
    num: int
    title: str
    bullets: list[str] = field(default_factory=list)
    visual: str = "none"
    narration: str = ""


def parse_script(text: str) -> list[Scene]:
    scenes: list[Scene] = []
    for block in re.split(r"^## ", text, flags=re.M)[1:]:
        lines = block.strip().splitlines()
        sc = Scene(num=len(scenes) + 1, title=lines[0].strip())
        narr = []
        for line in lines[1:]:
            s = line.strip()
            if s.startswith("- "):
                sc.bullets.append(s[2:].strip())
            elif s.startswith("visual:"):
                sc.visual = s[len("visual:"):].strip()
            elif s.startswith(">"):
                narr.append(s[1:].strip())
            elif s:
                raise ValueError(f"scene {sc.num} ({sc.title}): unexpected line {s!r}")
        if len(sc.bullets) > 5:
            raise ValueError(f"scene {sc.num}: more than 5 bullets")
        if not narr:
            raise ValueError(f"scene {sc.num}: no narration")
        sc.narration = " ".join(narr)
        scenes.append(sc)
    return scenes


# ------------------------------------------------------------------ numbers from results/

PLACEHOLDER = re.compile(r"\{\{([\w./-]+\.json):([\w.-]+)(?:\|(\w+))?\}\}")


def lookup(fname: str, path: str):
    obj = json.loads((ROOT / "results" / fname).read_text())
    for part in path.split("."):
        obj = obj[int(part)] if isinstance(obj, list) else obj[part]
    return obj


def fmt_value(x, fmt: str | None, spoken: bool) -> str:
    if fmt in (None, "raw"):
        return str(x)
    if fmt == "pct":
        return f"{100 * x:.1f} percent" if spoken else f"{100 * x:.1f}%"
    if fmt == "pp":
        if spoken:
            return f"{'minus' if x < 0 else 'plus'} {abs(x):.1f} percentage points"
        return f"{x:+.1f} pp"
    if fmt == "ms":
        return f"{x:.0f} milliseconds" if spoken else f"{x:.0f} ms"
    if fmt == "usd":
        if spoken:
            return f"{100 * x:.2f} cents" if x < 1 else f"{x:.2f} dollars"
        return f"${x:.4f}"
    if fmt == "int":
        return f"{int(x)}" if spoken else f"{int(x):,}"
    if fmt == "mb":
        return f"{x:.0f} megabytes" if spoken else f"{x:.0f} MB"
    if fmt in ("f1", "f2", "f3"):
        return f"{x:.{fmt[1]}f}"
    raise ValueError(f"unknown format {fmt!r}")


def fill(text: str, spoken: bool) -> str:
    return PLACEHOLDER.sub(lambda m: fmt_value(lookup(m.group(1), m.group(2)), m.group(3), spoken), text)


def to_speech(text: str) -> str:
    text = fill(text, spoken=True)
    for written, spoken in PRONOUNCE:
        text = re.sub(rf"(?<![\w-]){re.escape(written)}(?![\w-])", spoken, text)
    text = re.sub(r"(?<=[a-z])_(?=[a-z])", " ", text)  # alarm_set -> alarm set
    return text.replace("`", "")


# ------------------------------------------------------------------ drawing helpers

def wrap(draw: ImageDraw.ImageDraw, text: str, size: int, width: int, bold: bool = False) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        cand = f"{cur} {w}".strip()
        if draw.textlength(cand, font=font_for_text(cand, size, bold)) <= width or not cur:
            cur = cand
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def fit_text(draw, text: str, width: int, max_h: int, start: int, min_size: int, bold=False, gap=1.25):
    """Largest font size (<= start) at which the wrapped text fits in width x max_h."""
    for size in range(start, min_size - 1, -2):
        lines = wrap(draw, text, size, width, bold)
        if len(lines) * size * gap <= max_h and all(
                draw.textlength(ln, font=font_for_text(ln, size, bold)) <= width for ln in lines):
            return size, lines
    raise ValueError(f"text does not fit even at {min_size}px: {text[:60]!r}")


def strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", s)


def read_lines(path: str, spec: list[str]) -> tuple[list[str], int]:
    """spec: ["10-30"] or ["tail", "8"] -> (lines, first line number)."""
    lines = strip_ansi((ROOT / path).read_text(encoding="utf-8")).splitlines()
    if spec and spec[0] == "tail":
        n = int(spec[1])
        return lines[-n:], max(1, len(lines) - n + 1)
    if spec:
        a, b = (int(v) for v in spec[0].split("-"))
        return lines[a - 1:b], a
    return lines, 1


def code_image(path: str, spec: list[str]) -> Image.Image:
    from pygments import highlight
    from pygments.formatters import ImageFormatter
    from pygments.lexers import get_lexer_for_filename

    lines, start = read_lines(path, spec)
    src = "\n".join(lines)
    fmt = ImageFormatter(font_name="DejaVu Sans Mono", font_size=30, line_numbers=True, line_number_start=start,
                         style="default", line_pad=6, image_pad=24, line_number_bg="#eef1f5",
                         line_number_fg="#8a93a3")
    img = Image.open(BytesIO(highlight(src, get_lexer_for_filename(path), fmt))).convert("RGB")
    return add_caption(img, f"{path}  (lines {start}-{start + len(lines) - 1})")


def output_image(path: str, spec: list[str], max_chars: int = 92) -> Image.Image:
    lines, _ = read_lines(path, spec)
    out = []
    for ln in lines:
        ln = ln.rstrip().replace("\t", "    ")
        while len(ln) > max_chars:
            out.append(ln[:max_chars])
            ln = "    " + ln[max_chars:]
        out.append(ln)
    size, lh, pad = 28, 38, 28
    font = font_for("mono", size)
    probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    width = int(max([probe.textlength(ln, font=font_for_text(ln, size) if not ln.isascii() else font)
                     for ln in out] + [400])) + 2 * pad
    img = Image.new("RGB", (width, len(out) * lh + 2 * pad + 44), TERM_BG)
    d = ImageDraw.Draw(img)
    for i, c in enumerate(("#8a93a3", "#a5adbb", "#c3c9d4")):  # window dots
        d.ellipse([pad + i * 26, 14, pad + i * 26 + 14, 28], fill=c)
    y = 44 + pad
    for ln in out:
        f = font if ln.isascii() else font_for_text(ln, size)
        d.text((pad, y), ln, font=f, fill=TERM_INK)
        y += lh
    return add_caption(img, path)


def add_caption(img: Image.Image, caption: str) -> Image.Image:
    font = font_for("mono", 24)
    out = Image.new("RGB", (img.width, img.height + 44), BG)
    out.paste(img, (0, 44))
    ImageDraw.Draw(out).text((4, 8), caption, font=font, fill=MUTED)
    return out


def visual_image(visual: str) -> Image.Image | None:
    parts = visual.split()
    if not parts or parts[0] == "none":
        return None
    kind, args = parts[0], parts[1:]
    if kind == "image":
        return Image.open(ROOT / args[0]).convert("RGB")
    if kind == "code":
        return code_image(args[0], args[1:])
    if kind == "output":
        return output_image(args[0], args[1:])
    raise ValueError(f"unknown visual {visual!r}")


def render_slide(sc: Scene, total: int, out: Path) -> None:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    m = 80
    # title (up to 2 lines)
    size, lines = fit_text(d, fill(sc.title, False), W - 2 * m, 150, 64, 36, bold=True, gap=1.2)
    y = 56
    for ln in lines:
        d.text((m, y), ln, font=font_for_text(ln, size, True), fill=INK)
        y += int(size * 1.2)
    d.rectangle([m, y + 14, m + 140, y + 20], fill=ACCENT)
    top, bottom = y + 56, H - 70

    vis = visual_image(sc.visual)
    bullets = [fill(b, False) for b in sc.bullets]
    if vis is None:
        box = (m, top, W - 2 * m, bottom - top)
        start = 48
    elif vis.width / vis.height > 1.75:  # wide chart: bullets on top, visual below
        n = len(bullets)
        bh = 0 if n == 0 else min(70 * n + 20, 230)
        box = (m, top, W - 2 * m, bh)
        place_visual(img, vis, (m, top + bh + 10, W - 2 * m, bottom - top - bh - 10))
        start = 36
    else:  # bullets left, visual right
        bw = 640 if bullets else 0
        box = (m, top, bw, bottom - top)
        vx = m + bw + (50 if bullets else 0)
        place_visual(img, vis, (vx, top, W - m - vx, bottom - top))
        start = 40
    if bullets:
        draw_bullets(d, bullets, box, start)

    foot = font_for("latin", 22)
    d.text((m, H - 46), "Hybrid Intent Router - how it was built", font=foot, fill=MUTED)
    d.text((W - m, H - 46), f"{sc.num} / {total}", font=foot, fill=MUTED, anchor="ra")
    img.save(out)


def draw_bullets(d, bullets: list[str], box, start: int) -> None:
    x, y0, w, h = box
    text_w = w - 44
    for size in range(start, 21, -2):
        blocks = [wrap(d, b, size, text_w) for b in bullets]
        need = sum(len(bl) * size * 1.3 + size * 0.7 for bl in blocks)
        if need <= h and all(d.textlength(ln, font=font_for_text(ln, size)) <= text_w for bl in blocks
                             for ln in bl):
            break
    else:
        raise ValueError(f"bullets do not fit: {bullets}")
    y = y0
    for bl in blocks:
        r = size * 0.18
        cy = y + size * 0.62
        d.ellipse([x + 6, cy - r, x + 6 + 2 * r, cy + r], fill=ACCENT)
        for ln in bl:
            d.text((x + 44, y), ln, font=font_for_text(ln, size), fill=INK)
            y += int(size * 1.3)
        y += int(size * 0.7)


def place_visual(img: Image.Image, vis: Image.Image, box) -> None:
    x, y, w, h = box
    scale = min(w / vis.width, h / vis.height, 1.6)
    vis = vis.resize((max(1, int(vis.width * scale)), max(1, int(vis.height * scale))), Image.LANCZOS)
    px, py = x + (w - vis.width) // 2, y + (h - vis.height) // 2
    ImageDraw.Draw(img).rectangle([px - 3, py - 3, px + vis.width + 2, py + vis.height + 2], outline="#d5dae3",
                                  width=3)
    img.paste(vis, (px, py))


# ------------------------------------------------------------------ audio + video

def ensure_voice() -> Path:
    VOICES.mkdir(parents=True, exist_ok=True)
    onnx = VOICES / f"{VOICE}.onnx"
    for suffix in (".onnx", ".onnx.json"):
        p = VOICES / f"{VOICE}{suffix}"
        if not p.exists():
            print(f"downloading {p.name}")
            urllib.request.urlretrieve(VOICE_URL + suffix, p)
    return onnx


def synth(voice, text: str, out: Path) -> float:
    """Piper narration + PAUSE_S of silence. Returns the duration in seconds."""
    from piper import SynthesisConfig

    tmp = out.with_suffix(".raw.wav")
    with wave.open(str(tmp), "wb") as wf:
        voice.synthesize_wav(text, wf, syn_config=SynthesisConfig(length_scale=1.08))
    with wave.open(str(tmp), "rb") as r:
        params, frames = r.getparams(), r.readframes(r.getnframes())
    silence = b"\x00" * int(params.framerate * PAUSE_S) * params.sampwidth * params.nchannels
    with wave.open(str(out), "wb") as w:
        w.setparams(params)
        w.writeframes(frames + silence)
    tmp.unlink()
    return wav_seconds(out)


def wav_seconds(p: Path) -> float:
    with wave.open(str(p), "rb") as r:
        return r.getnframes() / r.getframerate()


def run(cmd: list[str]) -> None:
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{res.stderr[-2000:]}")


def make_clip(slide: Path, audio: Path, out: Path, seconds: float) -> None:
    run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-framerate", "30", "-i", str(slide), "-i", str(audio),
         "-c:v", "libx264", "-tune", "stillimage", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
         "-r", "30", "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2", "-t", f"{seconds:.3f}",
         "-movflags", "+faststart", str(out)])


def probe_seconds(p: Path) -> float:
    res = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                         capture_output=True, text=True, check=True)
    return float(res.stdout.strip())


def srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def subtitle_cues(text: str, start: float, end: float, max_chars: int = 84) -> list[tuple[float, float, str]]:
    """Split narration into short cues; time is shared out by character count."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    cues = []
    for s in sentences:
        while len(s) > max_chars:
            cut = s.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            cues.append(s[:cut])
            s = s[cut:].strip()
        if s:
            cues.append(s)
    total = sum(len(c) for c in cues) or 1
    out, t = [], start
    for c in cues:
        dt = (end - start) * len(c) / total
        out.append((t, t + dt, c))
        t += dt
    return out


def main() -> int:
    from piper import PiperVoice

    scenes = parse_script((VIDEO / "script.md").read_text(encoding="utf-8"))
    for sub in ("slides", "audio", "clips"):
        (BUILD / sub).mkdir(parents=True, exist_ok=True)
    voice = PiperVoice.load(str(ensure_voice()))

    clips, timings, t = [], [], 0.0
    srt, n_cue = [], 0
    words = 0
    for sc in scenes:
        slide = BUILD / "slides" / f"{sc.num:02d}.png"
        render_slide(sc, len(scenes), slide)
        speech = to_speech(sc.narration)
        words += len(speech.split())
        key = hashlib.sha256((VOICE + speech).encode()).hexdigest()[:12]
        audio = BUILD / "audio" / f"{sc.num:02d}-{key}.wav"
        seconds = wav_seconds(audio) if audio.exists() else synth(voice, speech, audio)
        clip_key = hashlib.sha256(slide.read_bytes() + audio.read_bytes()).hexdigest()[:12]
        clip = BUILD / "clips" / f"{sc.num:02d}-{clip_key}.mp4"
        if not clip.exists():
            make_clip(slide, audio, clip, seconds)
        real = probe_seconds(clip)
        clips.append(clip)
        timings.append({"scene": sc.num, "title": fill(sc.title, False), "start": round(t, 3),
                        "end": round(t + real, 3)})
        for a, b, text in subtitle_cues(fill(sc.narration, False), t, t + real - PAUSE_S):
            n_cue += 1
            srt.append(f"{n_cue}\n{srt_time(a)} --> {srt_time(b)}\n{text.replace('`', '')}\n")
        print(f"scene {sc.num:02d} {real:6.1f}s  {sc.title}")
        t += real

    lst = BUILD / "concat.txt"
    lst.write_text("".join(f"file '{c.resolve()}'\n" for c in clips))
    out = VIDEO / "tutorial.mp4"
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
         "-movflags", "+faststart", str(out)])
    (VIDEO / "tutorial.srt").write_text("\n".join(srt), encoding="utf-8")
    (BUILD / "timings.json").write_text(json.dumps(timings, indent=2))
    chapters = ["# Chapters\n", "| Time | Scene |", "|---|---|"]
    chapters += [f"| {int(x['start'] // 60):02d}:{int(x['start'] % 60):02d} | {x['title']} |" for x in timings]
    (VIDEO / "chapters.md").write_text("\n".join(chapters) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}: {len(scenes)} scenes, {words} spoken words, {t / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
