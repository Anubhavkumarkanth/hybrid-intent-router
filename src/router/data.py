"""Data pipeline for Amazon MASSIVE 1.1 (en-US, hi-IN, kn-IN).

Steps (run with `python -m router.data` or `make data`):
1. Download the official tarball (skip if present and SHA-256 matches).
2. Extract only the three language files.
3. Write data/processed/{lang}_{split}.jsonl with id, text, intent, label.
4. Write data/processed/labels.json (60 intents, sorted, from English train).
5. Make fixed subsets with seed 42: dev_sub (300 dev ids), test_sub (500 test ids).
"""

from __future__ import annotations

import hashlib
import json
import random
import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"

TARBALL_URL = (
    "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
)
TARBALL_NAME = "amazon-massive-dataset-1.1.tar.gz"
TARBALL_SHA256 = "4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577"

LANGS = {"en": "en-US", "hi": "hi-IN", "kn": "kn-IN"}
SPLITS = ("train", "dev", "test")
SEED = 42
DEV_SUB_SIZE = 300
TEST_SUB_SIZE = 500


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_tarball() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / TARBALL_NAME
    if path.exists() and sha256_of(path) == TARBALL_SHA256:
        print(f"tarball present, sha256 ok: {path}")
        return path
    print(f"downloading {TARBALL_URL}")
    urllib.request.urlretrieve(TARBALL_URL, path)
    got = sha256_of(path)
    if got != TARBALL_SHA256:
        raise RuntimeError(f"sha256 mismatch: expected {TARBALL_SHA256}, got {got}")
    print(f"downloaded, sha256 ok: {got}")
    return path


def read_locale(tar_path: Path, locale: str) -> list[dict]:
    member = f"1.1/data/{locale}.jsonl"
    with tarfile.open(tar_path, "r:gz") as tar:
        f = tar.extractfile(member)
        if f is None:
            raise FileNotFoundError(member)
        return [json.loads(line) for line in f.read().decode("utf-8").splitlines() if line]


def stratified_sample(ids_by_intent: dict[str, list[str]], k: int, seed: int) -> list[str]:
    """Pick k ids spread over intents as evenly as the counts allow.

    Proportional allocation (largest remainder), then a seeded random pick
    inside each intent. Deterministic for a fixed seed.
    """
    rng = random.Random(seed)
    total = sum(len(v) for v in ids_by_intent.values())
    intents = sorted(ids_by_intent)
    quotas = {i: k * len(ids_by_intent[i]) / total for i in intents}
    alloc = {i: int(quotas[i]) for i in intents}
    remaining = k - sum(alloc.values())
    by_remainder = sorted(intents, key=lambda i: (-(quotas[i] - alloc[i]), i))
    for i in by_remainder[:remaining]:
        alloc[i] += 1
    chosen: list[str] = []
    for i in intents:
        pool = sorted(ids_by_intent[i], key=int)
        chosen.extend(rng.sample(pool, alloc[i]))
    return sorted(chosen, key=int)


def build() -> None:
    tar_path = download_tarball()
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    rows: dict[str, list[dict]] = {lang: read_locale(tar_path, loc) for lang, loc in LANGS.items()}

    labels = sorted({r["intent"] for r in rows["en"] if r["partition"] == "train"})
    label_to_id = {name: idx for idx, name in enumerate(labels)}
    (PROC_DIR / "labels.json").write_text(json.dumps(labels, indent=1) + "\n")

    for lang, items in rows.items():
        for split in SPLITS:
            out = PROC_DIR / f"{lang}_{split}.jsonl"
            with open(out, "w", encoding="utf-8") as f:
                for r in items:
                    if r["partition"] != split:
                        continue
                    rec = {
                        "id": r["id"],
                        "text": r["utt"],
                        "intent": r["intent"],
                        "label": label_to_id[r["intent"]],
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"wrote {out.relative_to(ROOT)}")

    # Subsets: sampled from English, same ids used for all languages.
    for split, size, name in (("dev", DEV_SUB_SIZE, "dev_sub"), ("test", TEST_SUB_SIZE, "test_sub")):
        by_intent: dict[str, list[str]] = defaultdict(list)
        for r in rows["en"]:
            if r["partition"] == split:
                by_intent[r["intent"]].append(r["id"])
        ids = stratified_sample(by_intent, size, SEED)
        (PROC_DIR / f"{name}_ids.json").write_text(json.dumps(ids) + "\n")
        print(f"wrote {name}_ids.json ({len(ids)} ids)")


# ---------- helpers used by other modules ----------

def load_labels() -> list[str]:
    return json.loads((PROC_DIR / "labels.json").read_text())


def load_split(lang: str, split: str) -> list[dict]:
    path = PROC_DIR / f"{lang}_{split}.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_subset_ids(name: str) -> list[str]:
    return json.loads((PROC_DIR / f"{name}_ids.json").read_text())


def load_subset(lang: str, name: str) -> list[dict]:
    """name is 'dev_sub' or 'test_sub'. Returns rows in id order."""
    split = {"dev_sub": "dev", "test_sub": "test"}[name]
    ids = load_subset_ids(name)
    by_id = {r["id"]: r for r in load_split(lang, split)}
    return [by_id[i] for i in ids]


if __name__ == "__main__":
    build()
