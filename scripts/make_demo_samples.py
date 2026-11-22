"""Pick sentences for the demo page's "Random test sentence" button (make demo-samples).

25 seeded random test_sub sentences per language, with their true intent, so a visitor can see the model being
right and wrong on real data. Display only: nothing here feeds a decision.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from router.data import load_subset  # noqa: E402

out = {}
for lang in ("en", "hi", "kn"):
    rows = random.Random(f"demo-{lang}").sample(load_subset(lang, "test_sub"), 25)  # different picks per language
    out[lang] = [{"id": r["id"], "text": r["text"], "intent": r["intent"]} for r in rows]
path = ROOT / "results" / "demo_samples.json"
path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"wrote {path.relative_to(ROOT)}: {sum(len(v) for v in out.values())} sentences")
