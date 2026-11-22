"""Model tests with a tiny random BERT encoder (no download, CPU, a few seconds)."""

import pytest
import torch

from router.model import IntentClassifier, load_model, predict_logits, save_model

NUM_LABELS = 60


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    """Write a tiny random BERT + a char-level-ish WordPiece tokenizer to a temp folder."""
    from transformers import BertConfig, BertModel, BertTokenizerFast

    d = tmp_path_factory.mktemp("tiny")
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + list("abcdefghijklmnopqrstuvwxyz") + [
        "wake", "me", "up", "play", "music", "weather", "today", "set", "alarm"]
    # transformers 5.x: the argument is `vocab` (path or dict). The old `vocab_file=` keyword is
    # silently ignored and gives an empty vocabulary (every word -> [UNK]); this bit us once.
    tok = BertTokenizerFast(vocab={w: i for i, w in enumerate(vocab)})
    assert tok.unk_token_id not in tok("wake me up play music")["input_ids"]
    tok.save_pretrained(d)
    cfg = BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=64, max_position_embeddings=64)
    BertModel(cfg).save_pretrained(d)
    return d


def make(tiny_dir):
    from transformers import AutoTokenizer

    torch.manual_seed(0)
    return IntentClassifier(str(tiny_dir), NUM_LABELS), AutoTokenizer.from_pretrained(tiny_dir)


def test_output_shape(tiny_dir):
    model, tok = make(tiny_dir)
    enc = tok(["wake me up", "play music today", "a"], padding=True, return_tensors="pt")
    out = model(enc["input_ids"], enc["attention_mask"])
    assert out.shape == (3, NUM_LABELS)


def test_can_overfit_tiny_batch(tiny_dir):
    model, tok = make(tiny_dir)
    texts = ["wake me up", "play music", "weather today", "set alarm"]
    y = torch.tensor([3, 17, 42, 3])
    enc = tok(texts, padding=True, return_tensors="pt")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    model.train()
    first = None
    for _ in range(40):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(enc["input_ids"], enc["attention_mask"]), y)
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    model.eval()
    with torch.no_grad():
        pred = model(enc["input_ids"], enc["attention_mask"]).argmax(-1)
    assert loss.item() < first * 0.1
    assert (pred == y).all()


def test_save_load_identical(tiny_dir, tmp_path):
    model, tok = make(tiny_dir)
    model.eval()
    texts = ["wake me up", "weather today"]
    before = predict_logits(model, tok, texts)
    save_model(model, tok, tmp_path / "ckpt", extra={"max_len": 32})
    loaded, tok2, meta = load_model(tmp_path / "ckpt")
    after = predict_logits(loaded, tok2, texts)
    assert meta["max_len"] == 32 and meta["num_labels"] == NUM_LABELS
    assert torch.equal(before, after)


def test_warmup_schedule_shape():
    from router.train import linear_warmup_decay

    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=1.0)
    sched = linear_warmup_decay(opt, warmup_steps=10, total_steps=100)
    lrs = []
    for _ in range(100):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] == pytest.approx(0.1)
    assert max(lrs) == pytest.approx(1.0) and lrs.index(max(lrs)) == 9
    assert lrs[-1] < 0.02
