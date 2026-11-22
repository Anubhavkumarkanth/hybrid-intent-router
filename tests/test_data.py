"""Checks for the processed MASSIVE data (run `make data` first)."""

import json

import pytest

from router import data as D

LANGS = ["en", "hi", "kn"]
EXPECTED = {"train": 11514, "dev": 2033, "test": 2974}

pytestmark = pytest.mark.skipif(
    not (D.PROC_DIR / "en_train.jsonl").exists(), reason="run `make data` first"
)


@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize("split", list(EXPECTED))
def test_split_sizes(lang, split):
    assert len(D.load_split(lang, split)) == EXPECTED[split]


def test_sixty_intents_same_everywhere():
    labels = D.load_labels()
    assert len(labels) == 60
    assert labels == sorted(labels)
    for lang in LANGS:
        for split in EXPECTED:
            assert {r["intent"] for r in D.load_split(lang, split)} <= set(labels)
        assert {r["intent"] for r in D.load_split(lang, "train")} == set(labels)


def test_label_ids_match_names():
    labels = D.load_labels()
    for r in D.load_split("hi", "dev"):
        assert labels[r["label"]] == r["intent"]


@pytest.mark.parametrize("split", list(EXPECTED))
def test_ids_parallel_across_languages(split):
    en = {r["id"]: r["intent"] for r in D.load_split("en", split)}
    for lang in ("hi", "kn"):
        other = {r["id"]: r["intent"] for r in D.load_split(lang, split)}
        assert other == en


def test_known_example():
    kn = {r["id"]: r for r in D.load_split("kn", "test")}
    en = {r["id"]: r for r in D.load_split("en", "test")}
    assert en["0"]["text"] == "wake me up at five am this week"
    assert kn["0"]["intent"] == "alarm_set"


def test_subsets_sizes_and_no_overlap():
    dev_sub = D.load_subset_ids("dev_sub")
    test_sub = D.load_subset_ids("test_sub")
    assert len(dev_sub) == len(set(dev_sub)) == 300
    assert len(test_sub) == len(set(test_sub)) == 500
    train_ids = {r["id"] for r in D.load_split("en", "train")}
    dev_ids = {r["id"] for r in D.load_split("en", "dev")}
    test_ids = {r["id"] for r in D.load_split("en", "test")}
    assert set(dev_sub) <= dev_ids
    assert set(test_sub) <= test_ids
    assert not set(dev_sub) & set(test_sub)
    assert not (set(dev_sub) | set(test_sub)) & train_ids


def test_subsets_exist_in_all_languages():
    for name in ("dev_sub", "test_sub"):
        en = D.load_subset("en", name)
        for lang in ("hi", "kn"):
            other = D.load_subset(lang, name)
            assert [r["id"] for r in other] == [r["id"] for r in en]
            assert [r["intent"] for r in other] == [r["intent"] for r in en]


def test_subsets_reproducible():
    for split, size, name in (("dev", 300, "dev_sub"), ("test", 500, "test_sub")):
        by_intent = {}
        for r in D.load_split("en", split):
            by_intent.setdefault(r["intent"], []).append(r["id"])
        again = D.stratified_sample(by_intent, size, D.SEED)
        saved = json.loads((D.PROC_DIR / f"{name}_ids.json").read_text())
        assert again == saved


def test_subset_is_stratified():
    sub = D.load_subset("en", "test_sub")
    assert len({r["intent"] for r in sub}) >= 55
