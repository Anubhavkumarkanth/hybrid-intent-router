"""results/ must match the code: recompute T and t from the committed logits + LLM cache (offline, fast)."""

import json

import numpy as np
import pytest

from router import analysis as A
from router import calibrate as C
from router import route as R
from router.data import load_labels
from router.llm import Cache

ROUTING = A.RESULTS / "routing.json"
pytestmark = pytest.mark.skipif(not ROUTING.exists(), reason="run `make route` first")


def test_temperature_and_threshold_reproduce():
    saved = json.loads(ROUTING.read_text())
    en_dev = A.load_logits("en_dev")
    T = C.fit_temperature(en_dev["logits"], en_dev["labels"])
    assert T == pytest.approx(saved["temperature"], abs=1e-3)

    labels = load_labels()
    idx = {name: i for i, name in enumerate(labels)}
    model, pv = saved["llm"]["model"], saved["llm"]["prompt_version"]
    cache = Cache()
    parts = [A.load_part(lang, "dev_sub", T, idx, cache, model, pv) for lang in A.LANGS]
    dev = A.pool(parts)
    llm_only = float((dev["llm"] == dev["gold"]).mean())
    chosen = R.choose_threshold(R.sweep(dev["conf"], dev["small"], dev["llm"], dev["gold"]), llm_only, A.TOLERANCE)
    assert chosen["t"] == saved["chosen_threshold"]["t"]
    assert chosen["llm_share"] == pytest.approx(saved["chosen_threshold"]["llm_share"], abs=1e-4)

    test = A.pool([A.load_part(lang, "test_sub", T, idx, cache, model, pv) for lang in A.LANGS])
    res = R.evaluate_threshold(test["conf"], test["small"], test["llm"], test["gold"], chosen["t"])
    assert res["accuracy"] == pytest.approx(saved["test_sub"]["pooled"]["router_acc"], abs=1e-4)
    assert np.isclose(res["llm_share"], saved["test_sub"]["pooled"]["llm_share"], atol=1e-4)
