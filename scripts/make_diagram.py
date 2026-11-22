"""Draw the two architecture diagrams with Pillow (no numbers, so nothing to keep in sync).

results/charts/architecture.png  how one message is routed (small model first, LLM when unsure)
results/charts/serving.png       how a request travels: client -> NGINX -> gunicorn -> Flask -> model / LLM
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from router.fonts import font_for  # noqa: E402

OUT = ROOT / "results" / "charts"
INK, MUTED, LINE = "#1d2330", "#5b6475", "#6b7486"
BLUE, GREEN, PURPLE, GREY = ("#e3edfc", "#2a6fdb"), ("#e2f4ea", "#1a9e5c"), ("#efe9f7", "#7b5ea7"), \
    ("#eef0f3", "#8a93a3")


def box(d: ImageDraw.ImageDraw, x, y, w, h, lines, colors, diamond=False):
    fill, edge = colors
    if diamond:
        cx, cy = x + w / 2, y + h / 2
        d.polygon([(cx, y), (x + w, cy), (cx, y + h), (x, cy)], fill=fill, outline=edge, width=4)
    else:
        d.rounded_rectangle([x, y, x + w, y + h], radius=18, fill=fill, outline=edge, width=4)
    sizes = [30] + [24] * (len(lines) - 1)
    total = sum(s * 1.35 for s in sizes)
    ty = y + (h - total) / 2
    for i, (ln, s) in enumerate(zip(lines, sizes, strict=True)):
        f = font_for("latin", s, bold=i == 0)
        d.text((x + w / 2, ty), ln, font=f, fill=INK if i == 0 else MUTED, anchor="ma")
        ty += s * 1.35


def arrow(d: ImageDraw.ImageDraw, pts, label=None, label_at=None):
    d.line(pts, fill=LINE, width=4, joint="curve")
    (x1, y1), (x2, y2) = pts[-2], pts[-1]
    if x1 == x2:
        s = 1 if y2 > y1 else -1
        tri = [(x2, y2), (x2 - 11, y2 - 20 * s), (x2 + 11, y2 - 20 * s)]
    else:
        s = 1 if x2 > x1 else -1
        tri = [(x2, y2), (x2 - 20 * s, y2 - 11), (x2 - 20 * s, y2 + 11)]
    d.polygon(tri, fill=LINE)
    if label:
        d.text(label_at, label, font=font_for("latin", 26, True), fill=INK, anchor="mm")


def routing_diagram() -> None:
    img = Image.new("RGB", (1900, 820), "white")
    d = ImageDraw.Draw(img)
    d.text((40, 30), "How one message is routed", font=font_for("latin", 40, True), fill=INK)
    box(d, 40, 300, 370, 190, ["Customer message", "English, Hindi or Kannada", "(text, max 500 chars)"], GREY)
    box(d, 480, 290, 400, 210, ["Small model", "MuRIL encoder + linear head", "trained on English only",
                                "runs on CPU"], BLUE)
    box(d, 950, 245, 340, 300, ["Confident?", "calibrated confidence", ">= threshold t"], PURPLE, diamond=True)
    box(d, 1400, 110, 460, 200, ["Small model answers", "fast and free"], GREEN)
    box(d, 1400, 480, 460, 200, ["LLM answers (Gemini)", "slower, costs money per call"], BLUE)
    arrow(d, [(410, 395), (480, 395)])
    arrow(d, [(880, 395), (950, 395)])
    arrow(d, [(1120, 245), (1120, 210), (1400, 210)], "yes", (1220, 180))
    arrow(d, [(1120, 545), (1120, 580), (1400, 580)], "no", (1220, 610))
    d.text((40, 760), "If the LLM fails or times out, the app keeps the small model's answer "
                      "(routed_to = small_fallback).", font=font_for("latin", 24), fill=MUTED)
    img.save(OUT / "architecture.png")


def serving_diagram() -> None:
    img = Image.new("RGB", (1900, 720), "white")
    d = ImageDraw.Draw(img)
    d.text((40, 30), "How a request is served on Linux (WSL2 Ubuntu)", font=font_for("latin", 40, True), fill=INK)
    y, h = 200, 230
    box(d, 40, y, 280, h, ["Client", "browser, curl", "or Locust"], GREY)
    box(d, 400, y, 320, h, ["NGINX", "127.0.0.1:8080", "reverse proxy,", "runs as normal user"], PURPLE)
    box(d, 800, y, 340, h, ["gunicorn", "127.0.0.1:8000", "3 sync workers", "x 4 torch threads"], GREEN)
    box(d, 1220, y, 300, h, ["Flask app", "POST /predict", "GET /health"], BLUE)
    box(d, 1600, 110, 270, 170, ["Small model", "in each worker"], BLUE)
    box(d, 1600, 380, 270, 170, ["Gemini API", "only when unsure"], GREY)
    arrow(d, [(320, y + h / 2), (400, y + h / 2)])
    arrow(d, [(720, y + h / 2), (800, y + h / 2)])
    arrow(d, [(1140, y + h / 2), (1220, y + h / 2)])
    arrow(d, [(1520, 270), (1560, 270), (1560, 195), (1600, 195)])
    arrow(d, [(1520, 360), (1560, 360), (1560, 465), (1600, 465)])
    d.text((40, 620), "preload_app loads the model once before the workers fork; each worker answers one "
                      "request at a time.", font=font_for("latin", 24), fill=MUTED)
    img.save(OUT / "serving.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    routing_diagram()
    serving_diagram()
    print("wrote results/charts/architecture.png and results/charts/serving.png")
