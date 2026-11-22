"""The two pieces of model shrinking that are our own code: vocabulary pruning and int8 embedding rows."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import quantize as Q  # noqa: E402


def tiny_tokenizer_json(tmp_path):
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors

    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "wake", "##s", "me", "ಎದ್ದೇಳು", "জাগাও", "जगा", "##ओ"]
    tok = Tokenizer(models.WordPiece({w: i for i, w in enumerate(vocab)}, unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=False)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2),
                                                                                                 ("[SEP]", 3)])
    tok.add_special_tokens(["[PAD]", "[UNK]", "[CLS]", "[SEP]"])
    p = tmp_path / "tok.json"
    tok.save(str(p))
    return json.loads(p.read_text(encoding="utf-8"))


def test_prune_keeps_in_scope_scripts_and_same_tokens(tmp_path):
    from tokenizers import Tokenizer

    old_json = tiny_tokenizer_json(tmp_path)
    new_json, kept = Q.prune_tokenizer(old_json)
    assert "জাগাও" not in new_json["model"]["vocab"]  # Bengali dropped
    assert {"wake", "##s", "ಎದ್ದೇಳು", "जगा", "##ओ", "[CLS]"} <= set(new_json["model"]["vocab"])
    assert sorted(new_json["model"]["vocab"].values()) == list(range(len(kept)))
    (tmp_path / "new.json").write_text(json.dumps(new_json, ensure_ascii=False), encoding="utf-8")
    old, new = Tokenizer.from_file(str(tmp_path / "tok.json")), Tokenizer.from_file(str(tmp_path / "new.json"))
    for text in ["wakes me", "ಎದ್ದೇಳು", "जगाओ me"]:
        assert old.encode(text).tokens == new.encode(text).tokens
        # the new id of each token is the position of its old id in `kept`
        assert [kept[i] for i in new.encode(text).ids] == old.encode(text).ids


def test_int8_embedding_rows_match_float_table():
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(0)
    table = rng.normal(size=(50, 8)).astype(np.float32)
    table[3] = 0.0  # an all-zero row must not divide by zero
    g = helper.make_graph([helper.make_node("Gather", ["table", "ids"], ["out"], axis=0)], "g",
                          [helper.make_tensor_value_info("ids", TensorProto.INT64, ["b", "s"])],
                          [helper.make_tensor_value_info("out", TensorProto.FLOAT, ["b", "s", 8])],
                          [numpy_helper.from_array(table, "table")])
    m = Q.quantize_embedding_rows(helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]), "table")
    onnx.checker.check_model(m)
    ids = np.array([[0, 3, 49], [7, 7, 1]], dtype=np.int64)
    out = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, {"ids": ids})[0]
    step = np.abs(table).max(axis=1) / 127  # one int8 step per row
    assert np.all(np.abs(out - table[ids]) <= step[ids][..., None] / 2 + 1e-7)
    assert np.all(out[0, 1] == 0)


@pytest.mark.skipif(not (ROOT / "models" / "onnx" / "model.onnx").exists(), reason="run `make quantize` first")
def test_shipped_onnx_model_agrees_with_original():
    from router.data import load_subset
    from router.evaluate import load_logits
    from router.onnx_model import OnnxIntentModel

    m = OnnxIntentModel(ROOT / "models" / "onnx")
    rows = load_subset("en", "test_sub")[:100]
    ref = load_logits("en_test_sub")["logits"][:100]
    assert (m.logits([r["text"] for r in rows]).argmax(1) == ref.argmax(1)).mean() >= 0.97
