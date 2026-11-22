"""Pick a font that can draw the text's script (Latin, Devanagari for Hindi, Kannada).

DejaVu has no Devanagari or Kannada glyphs, so Hindi/Kannada text uses Noto fonts
(SIL Open Font License) stored in assets/fonts/. Pillow shapes these scripts correctly
when its raqm layout engine is available (needs libfribidi + libharfbuzz, present on Ubuntu).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PIL import ImageFont

ROOT = Path(__file__).resolve().parents[2]
DEJAVU = Path("/usr/share/fonts/truetype/dejavu")
NOTO = ROOT / "assets" / "fonts"

FONT_FILES = {
    ("latin", False): DEJAVU / "DejaVuSans.ttf",
    ("latin", True): DEJAVU / "DejaVuSans-Bold.ttf",
    ("mono", False): DEJAVU / "DejaVuSansMono.ttf",
    ("mono", True): DEJAVU / "DejaVuSansMono-Bold.ttf",
    ("devanagari", False): NOTO / "NotoSansDevanagari-Regular.ttf",
    ("devanagari", True): NOTO / "NotoSansDevanagari-Bold.ttf",
    ("kannada", False): NOTO / "NotoSansKannada-Regular.ttf",
    ("kannada", True): NOTO / "NotoSansKannada-Bold.ttf",
}


def script_of(text: str) -> str:
    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F:
            return "devanagari"
        if 0x0C80 <= cp <= 0x0CFF:
            return "kannada"
    return "latin"


@lru_cache(maxsize=256)
def font_for(script: str, size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_FILES[(script, bold)]
    if not path.exists():
        raise FileNotFoundError(f"font missing: {path} (see assets/fonts/README.md)")
    layout = ImageFont.Layout.RAQM if script in ("devanagari", "kannada") else ImageFont.Layout.BASIC
    return ImageFont.truetype(str(path), size, layout_engine=layout)


def font_for_text(text: str, size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return font_for(script_of(text), size, bold)
