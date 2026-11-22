"""Write results/references.json: published zero-shot MASSIVE intent accuracy (hi-IN, kn-IN).

These numbers are copied by hand from the papers (checked 2026-11-22, see RESEARCH.md section 1).
They are reference points only, not our results.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ENTRIES = [
    {"model": "XLM-R base", "source": "MASSIVE paper, App. D Table 8", "hi": 74.8, "kn": 63.5,
     "url": "https://arxiv.org/abs/2204.08582"},
    {"model": "XLM-R", "source": "IndicXTREME, Table 19", "hi": 74.9, "kn": 61.7,
     "url": "https://arxiv.org/abs/2212.05409"},
    {"model": "MuRIL", "source": "IndicXTREME, Table 19", "hi": 82.4, "kn": 77.5,
     "url": "https://arxiv.org/abs/2212.05409"},
]

if __name__ == "__main__":
    out = {"note": "Published zero-shot (train on English) intent accuracy on MASSIVE test, percent.",
           "checked": "2026-11-22", "entries": ENTRIES}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "references.json").write_text(json.dumps(out, indent=2))
    print("wrote results/references.json")
