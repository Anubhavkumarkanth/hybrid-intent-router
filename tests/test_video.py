"""Video builder helpers (no audio, no ffmpeg): script parsing, number placeholders, mixed-script text runs."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "video"))
import build_video as B  # noqa: E402


def test_script_runs_split_latin_and_indic():
    runs = B.script_runs('-d \'{"text": "ಈ ವಾರ"}\'')
    assert [k for k, _ in runs] == ["mono", "kannada", "mono"]
    assert "".join(p for _, p in runs) == '-d \'{"text": "ಈ ವಾರ"}\''
    assert runs[1][1] == "ಈ ವಾರ"  # the space inside Kannada stays in the Kannada run


def test_zero_width_joiner_stays_with_indic_run():
    text = "क‍ष x"
    assert [k for k, _ in B.script_runs(text)] == ["devanagari", "mono"]


def test_read_lines_counts_like_sed_despite_progress_bar_carriage_returns(tmp_path):
    """Regression: bare \\r from progress bars shifted every log range by 4 lines."""
    p = tmp_path / "x.log"
    p.write_bytes(b"one\nLoading 0%\rLoading 100%\nthree\n    four\n")
    lines, start = B.read_lines(str(p), ["2-4"])
    assert start == 2 and lines == ["Loading 100%", "three", "    four"]


def test_placeholders_spoken_and_written():
    assert B.fmt_value(0.8407, "pct", spoken=False) == "84.1%"
    assert B.fmt_value(0.8407, "pct", spoken=True) == "84.1 percent"
    assert B.fmt_value(0.1333, "pp", spoken=True) == "plus 0.1 percentage points"
    assert B.fmt_value(0.0331, "usd", spoken=True) == "3.31 cents"
    with pytest.raises(ValueError):
        B.fmt_value(1, "nope", spoken=False)


def test_pronunciation():
    assert B.to_speech("PyTorch and NGINX call the LLM") == "Pie Torch and engine X call the L L M"


def test_real_script_parses_within_limits():
    scenes = B.parse_script((B.VIDEO / "script.md").read_text(encoding="utf-8"))
    assert 25 <= len(scenes) <= 35
    assert all(len(s.bullets) <= 5 and s.narration for s in scenes)
