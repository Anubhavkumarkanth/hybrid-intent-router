# R1: Data, model and training recipe

Goal: train a small multilingual encoder in plain PyTorch on MASSIVE 1.1 **en-US train only**
(11,514 utterances, 60 intents). Then test it zero-shot on **hi-IN** and **kn-IN**.
Hardware: RTX 3050 laptop GPU with 4 GB VRAM, 12 CPU cores, 7.4 GB RAM.

Every fact below has a source and was checked 2026-11-22. Lines marked **ESTIMATE** are my
own reasoning, not a published number.

---

## 1. Dataset: MASSIVE 1.1

### Confirmed facts

| Fact | Value | Source (checked 2026-11-22) |
|---|---|---|
| Tarball URL | `https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz`. It returns HTTP 200, is 40,251,390 bytes, and was last modified 2022-11-07. | [GitHub README](https://github.com/alexa/massive), plus a HEAD request to the URL |
| What 1.1 adds | 1.1 = 1.0 plus Catalan. "Data for languages other than Catalan are unchanged versus MASSIVE 1.0", so the hi-IN and kn-IN data match the paper's data. | [README](https://github.com/alexa/massive) |
| Data license | CC BY 4.0. NOTICE.md says: "The MASSIVE dataset is licensed under CC BY 4.0". The repo **code** is Apache 2.0, which is a separate license. | [NOTICE.md](https://github.com/alexa/massive/blob/main/NOTICE.md), [LICENSE.txt](https://github.com/alexa/massive/blob/main/LICENSE.txt), [HF card](https://huggingface.co/datasets/AmazonScience/massive) |
| JSONL fields | `id, locale, partition, scenario, intent, utt, annot_utt, worker_id`. Non-English files also have `slot_method` and `judgments`. `partition` is one of `train`/`dev`/`test`. We only need `id, locale, partition, intent, utt`. | [README](https://github.com/alexa/massive), [HF card](https://huggingface.co/datasets/AmazonScience/massive) |
| Split sizes per locale | 11,514 train / 2,033 dev / 2,974 test, the same in every locale | [HF card](https://huggingface.co/datasets/AmazonScience/massive); the dev and test sizes also appear in Table 2 of [IndicXTREME](https://arxiv.org/abs/2212.05409) |
| Intents | 60 intents, 55 slot types, 18 domains | [paper abstract](https://arxiv.org/abs/2204.08582) |
| Parallel ids | Every locale is a translation of the same SLURP English utterance, and `id` "maps to the original ID in the SLURP collection". So the same `id` means the same utterance in en-US, hi-IN and kn-IN. | [README](https://github.com/alexa/massive), [paper](https://arxiv.org/abs/2204.08582) |
| Total size | About 1M utterances: 587k train, 104k dev, 152k test, 153k held out (51 languages in 1.0) | [paper §1](https://arxiv.org/abs/2204.08582) |
| Scripts | hi-IN is Hindi in Devanagari (Indic genus). kn-IN is Kannada in Kannada script (Southern Dravidian). | [paper Table 2](https://arxiv.org/html/2204.08582) |

Notes:
- `worker_id` values are specific to each locale and do not map across locales. Only `id` is parallel. ([README](https://github.com/alexa/massive), checked 2026-11-22)
- Practical tip: the S3 download was very slow from this machine (about 3 MB in 6 minutes).
  The Hugging Face mirror `AmazonScience/massive` is an alternative, but it uses a loading script.
  Plan for a slow first download and cache the file.

### Published zero-shot intent accuracy (train on en-US, test on other locales)

**Source: MASSIVE paper, FitzGerald et al. 2022, arXiv 2204.08582, Appendix D, Table 8
"Intent accuracy by language … full dataset and zero-shot setup"**
([HTML version](https://arxiv.org/html/2204.08582), checked 2026-11-22). The ± values are confidence intervals.

| Locale | mT5 T2T Full | mT5 Enc Full | XLM-R base Full | mT5 T2T **Zero** | mT5 Enc **Zero** | XLM-R base **Zero** |
|---|---|---|---|---|---|---|
| en-US | 87.9 ±1.2 | 89.0 ±1.1 | 88.3 ±1.2 | n/a | n/a | n/a |
| **hi-IN** | 85.6 ±1.3 | 86.2 ±1.2 | 85.8 ±1.3 | 62.4 ±1.7 | 59.3 ±1.8 | **74.8 ±1.6** |
| **kn-IN** | 84.9 ±1.3 | 84.9 ±1.3 | 84.0 ±1.3 | 56.7 ±1.8 | 51.8 ±1.8 | **63.5 ±1.7** |
| all (avg) | 85.3 ±0.2 | 86.1 ±0.2 | 85.1 ±0.2 | 62.9 ±0.2 | 61.2 ±0.2 | 70.6 ±0.2 |

- "Full" means trained on all 51 locales. "Zero" means trained on en-US only.
- **XLM-R large: the paper does not report it.** The paper only has XLM-R **base**, mT5 base
  text-to-text, and mT5 base encoder-only. I found no XLM-R large MASSIVE number in it.
- Caveats on the paper's setup ([paper Table 3 caption, Appendix C](https://arxiv.org/html/2204.08582)):
  1. The models are **joint intent + slot** models, not intent-only.
  2. In zero-shot, "validation was performed with all locales", so the checkpoint was chosen using
     non-English dev data. Our setup (English dev only) is stricter, so expect lower or noisier numbers.
  3. The paper's hyperparameters came from a 128-trial search.
- The paper's zero-shot XLM-R base settings (Table 6): lr **4.7e-6**, linear schedule,
  warmup 500 steps, weight decay 0.11, gradient accumulation 8, Adam β=(0.99, 0.9999),
  dropout 0.35 (attention) / 0.25 (FFN), max-pooling intent head, and **word embeddings frozen**
  (`xlmr.embeddings.word_embeddings.weight`). The "Tot Epochs, LR Sched" value is 850,
  with a search range of [50, 1500]. The paper does not explain what that unit is, so I do not read it as an epoch count.
  (checked 2026-11-22)

**Second source: IndicXTREME / IndicBERT v2 paper, Doddapaneni et al. 2023, arXiv 2212.05409, Table 19
"Results on MASSIVE Intent Classification task", zero-shot from English**
([ar5iv](https://ar5iv.labs.arxiv.org/html/2212.05409), checked 2026-11-22):

| Model | hi | kn | Avg (7 Indic: bn hi kn ml ta te ur) |
|---|---|---|---|
| mBERT | 20.6 | 10.8 | 13.2 |
| XLM-R (base) | 74.9 | 61.7 | 66.6 |
| **MuRIL** | **82.4** | **77.5** | 77.2 |
| IndicBERT v2 | 82.7 | 78.2 | 78.8 |

- The XLM-R numbers match the MASSIVE paper closely (hi 74.9 vs 74.8, kn 61.7 vs 63.5).
- For intent and slot tasks they used lr 1e-5, weight decay 0.1, batch size 256 (Appendix N).
- The mBERT row looks unusually low. Treat it with care.

**Takeaway:** for kn-IN, MuRIL is about **16 points** better than XLM-R base in zero-shot (77.5 vs 61.7).
For hi-IN it is about 7.5 points better (82.4 vs 74.9).

---

## 2. Model choice

| | google/muril-base-cased | FacebookAI/xlm-roberta-base | microsoft/Multilingual-MiniLM-L12-H384 | distilbert-base-multilingual-cased |
|---|---|---|---|---|
| Architecture | BERT base, 12 layers, hidden size 768 | RoBERTa base, 12 layers, hidden size 768 | BERT, 12 layers, hidden size 384 | DistilBERT, 6 layers, hidden size 768 |
| Parameters | **236M** ([paper §2](https://arxiv.org/abs/2103.10730)) | 278.9M (HF safetensors count); "270M" in the [XLM-R paper Table 7](https://ar5iv.labs.arxiv.org/html/1911.02116) | 21M transformer + 96M embeddings, about 117M ([card](https://huggingface.co/microsoft/Multilingual-MiniLM-L12-H384)) | 135.4M (HF safetensors count) |
| Vocab size | 197,285 (cased WordPiece) | 250,002 (SentencePiece) | 250,037 (uses the XLM-R tokenizer) | 119,547 |
| Hindi / Kannada | 17 languages: 16 Indian languages + English, including hi and kn. Each has extra **transliterated** and **translated** pairs ([paper](https://arxiv.org/abs/2103.10730)). | CC-100 data: hi 20.2 GiB (1,715M tokens), **kn 3.3 GiB (169M tokens)** ([XLM-R Table 6](https://ar5iv.labs.arxiv.org/html/1911.02116)) | Same as the XLM-R tokenizer; distilled from XLM-R | Wikipedia data for 104 languages |
| License | Apache-2.0 | MIT | MIT | Apache-2.0 |
| Zero-shot MASSIVE intent hi / kn | 82.4 / 77.5 (IndicXTREME T19) | 74.9 / 61.7 (IndicXTREME T19); 74.8 / 63.5 (MASSIVE T8) | not published that I found | not published that I found |

All of the above was checked 2026-11-22. I got vocab sizes and parameter counts from the HF API and the `config.json` of each repo.

More published cross-lingual results (all zero-shot from English):
- **MuRIL vs mBERT on XTREME** (from the [model card](https://huggingface.co/google/muril-base-cased) and [paper](https://arxiv.org/abs/2103.10730)):
  - Averages: XNLI 74.07 vs 66.81, PANX 77.60 vs 58.01, MLQA F1 73.81 vs 65.34.
  - Hindi: XNLI 70.66 vs 60.52.
  - XTREME has no Kannada result in those tables.
  - The MuRIL paper compares only against mBERT, not XLM-R.
  - MuRIL's fine-tuning setup for XNLI was batch 32, lr 2e-5, 5 epochs, warmup 0.1, max_len 128 (MuRIL paper Table 3).
- **XLM-R base on XNLI:** average 76.2, hi 72.4 ([XLM-R Table 1](https://ar5iv.labs.arxiv.org/html/1911.02116)).
- **Multilingual-MiniLM on XNLI:** average 71.1, hi 66.2. XLM-R base scores 74.5 / 71.1 and mBERT 66.3 / 60.0 in the same table ([card](https://huggingface.co/microsoft/Multilingual-MiniLM-L12-H384)).

CPU inference speed (**ESTIMATE**, based on architecture):
- MuRIL and XLM-R base have the **same 12×768 transformer**. Their speed per token is almost the same.
  The large vocabularies only cost a table lookup, not compute. MuRIL's smaller vocab saves about 40M parameters
  (≈160 MB fp32) of RAM and disk.
- Indic text often splits into fewer subword tokens with MuRIL. Its paper shows a lower fertility ratio than mBERT
  (Fig. 5). Fewer tokens means slightly faster inference on hi and kn.
- Multilingual-MiniLM (hidden 384) costs roughly 1/4 of the compute per token.
  DistilBERT-multilingual (6 layers) costs about 1/2.
- For short queries (under 32 tokens), a 12×768 encoder on a modern CPU usually takes about 10–30 ms per sentence in fp32.
  Dynamic int8 quantization (`torch.ao.quantization.quantize_dynamic` on Linear layers) usually gives 1.5–2.5× speed-up.
  These are **ESTIMATES**: measure them on this machine.

### Recommendation
- **Primary: `google/muril-base-cased`.**
  1. It has the best published zero-shot MASSIVE intent scores on our exact target locales:
     kn +15.8 and hi +7.5 points over XLM-R base (IndicXTREME Table 19).
  2. Its pretraining included hi and kn, plus translation and transliteration pairs that line up English with Indic text.
  3. It is Apache-2.0 licensed.
  4. It has fewer parameters than XLM-R base (236M vs 279M).
  - Watch-outs: it was released as a TF checkpoint converted to PyTorch, so load it with `BertModel`.
    It is cased, so do not lowercase the text.
- **Backup (small/fast): `microsoft/Multilingual-MiniLM-L12-H384`.**
  1. Its XNLI result is about 3 points behind XLM-R base while needing about 1/4 of the transformer compute (card).
  2. It is MIT licensed.
  3. It shares XLM-R's tokenizer, which covers Kannada better than mBERT's.
  - Load it with `BertModel` + `XLMRobertaTokenizer`. The card warns that AutoTokenizer does not work.
  - I chose it over distilbert-multilingual because that model descends from mBERT, and mBERT did very poorly
    on Indic MASSIVE zero-shot (IndicXTREME T19: hi 20.6, kn 10.8).
- **Reference baseline (optional):** run xlm-roberta-base once. Our numbers can then be compared directly with the
  published 74.8 / 63.5 (MASSIVE T8).
- Also worth noting: `ai4bharat/IndicBERTv2-MLM-only` (MIT, 12×768, vocab 250k) scored slightly higher than MuRIL
  (82.7 / 78.2). It is a valid alternative if MuRIL has problems. (checked 2026-11-22: [HF](https://huggingface.co/ai4bharat/IndicBERTv2-MLM-only))

---

## 3. Training recipe (≈236–280M encoder, 11.5k short sentences, 60 classes)

### The 4 GB VRAM problem (important)
Full fine-tuning with AdamW keeps about 16 bytes per trainable parameter: the fp32 weights, the gradients, and Adam's two moment estimates.
Mixed precision (AMP) does **not** shrink these.
- XLM-R base: 279M × 16 B ≈ **4.5 GB**, which **does not fit** in 4 GB.
- MuRIL: 236M × 16 B ≈ **3.8 GB**, too tight once activations and the CUDA context are added.
- **Fix: freeze the word-embedding matrix.** MuRIL's embeddings are 197,285 × 768 ≈ 151M parameters,
  so about 85M parameters remain trainable. Memory then comes to ≈ 0.95 GB of weights + 0.34 GB of gradients + 0.68 GB of Adam state ≈ 2 GB,
  plus small activations.
- This is also what the MASSIVE authors chose for their zero-shot XLM-R run: frozen word embeddings ([Table 6](https://arxiv.org/html/2204.08582)).
  It is commonly believed to protect cross-lingual transfer, because the Hindi and Kannada token embeddings are never
  updated in a way that drifts them from English. The MASSIVE choice is the cited evidence; the "why" is my reasoning.
  (checked 2026-11-22)
- (**ESTIMATE**) With frozen embeddings + AMP (fp16 autocast + GradScaler) + max_len 64 + batch 32,
  peak VRAM should be about 2.5–3 GB. If it runs out of memory, use batch 16 with gradient accumulation 2.

### Suggested hyperparameters (starting point)

| Setting | Value | Why / source |
|---|---|---|
| max_len | **64**, with dynamic padding per batch | MASSIVE utterances are short voice-assistant commands. I could not compute exact token lengths here because the download was too slow, so **check the 99th-percentile token length** on the train set. 64 should cover hi and kn, where subword counts run higher. |
| batch size | 32 | Fits in VRAM; MuRIL XTREME used 32 ([MuRIL Table 3](https://arxiv.org/abs/2103.10730)) |
| learning rate | **2e-5** (try 1e-5 and 3e-5) | MuRIL XNLI used 2e-5. IndicXTREME MASSIVE used 1e-5. The MASSIVE XLM-R zero-shot search picked an even lower 4.7e-6. A lower lr tends to be safer for transfer (published choices; the trend is my reading). |
| optimizer | AdamW, betas (0.9, 0.999), eps 1e-8 | standard |
| weight decay | 0.01 (no decay on bias/LayerNorm) | IndicXTREME grid {0, 0.01}; they used 0.1 for MASSIVE |
| warmup | 10% of steps, then linear decay | MuRIL Table 3 warmup 0.1 |
| epochs | **3–5** (max 5), keeping the best checkpoint | 11,514 / 32 ≈ 360 steps per epoch, so about 1,100–1,800 steps. MuRIL XNLI used 5 epochs. IndicXTREME best epochs were around 2–5 (Table 13). |
| gradient clipping | max norm 1.0 | standard |
| dropout | 0.1 default; classifier dropout 0.1–0.2 | |
| head | [CLS] (or mean-pool) → dropout → Linear(768, 60) | |
| seeds | run 3 seeds and report mean ± std | Zero-shot results vary a lot between runs ([Keung et al. 2020](https://aclanthology.org/2020.emnlp-main.40)) |
| early stopping | evaluate on **en-US dev** each epoch, patience 1–2 | See the warning below |

### Risk: too much English training can hurt zero-shot
- **Keung et al., EMNLP 2020, "Don't Use English Dev"** ([ACL Anthology](https://aclanthology.org/2020.emnlp-main.40), checked 2026-11-22):
  - English dev accuracy "often has little or negative correlation" with target-language accuracy.
  - Published mBERT zero-shot results differed by up to 17 points partly because of how checkpoints were chosen.
  - They suggest also reporting an "oracle" score, where the checkpoint is picked with target-language dev data, as an upper bound.
- **Gaschi et al., Findings of ACL 2023** ([arXiv 2306.02790](https://arxiv.org/abs/2306.02790), checked 2026-11-22):
  - Fine-tuning on English can damage the model's cross-lingual alignment.
  - Alignment correlates with transfer.
- I found **no paper that specifically measures "more epochs → worse zero-shot" on MASSIVE**.
  That remains a plausible risk, not a proven fact here.
- Practical rules:
  1. Keep epochs low (≤5) and the lr small.
  2. Freeze the embeddings.
  3. Pick the checkpoint by en-US dev, as the project rules require.
  4. Also *log* hi-IN and kn-IN **dev** accuracy every epoch for analysis only, and report it as an "oracle" upper bound.
     Never use it for selection if the claim is strict zero-shot.

### Time estimates (**ESTIMATES**, not measured; verify with a 50-step timing run)
- The work per epoch is ≈ 360 steps × 32 sentences × ~20–30 tokens.
- **GPU (RTX 3050 laptop, AMP, frozen embeddings):** about 6–12 steps/s, so **≈ 1 minute per epoch**.
  5 epochs plus evaluation on en/hi/kn dev and test comes to **≈ 5–10 minutes per run**.
  3 seeds × 3 learning rates takes under 1.5 hours.
- **CPU (12 cores, fp32, `torch.set_num_threads(12)`):** about 0.3–0.8 steps/s for a 12×768 model, so
  **≈ 8–20 minutes per epoch** and **≈ 1–1.5 hours per 5-epoch run**.
  Multilingual-MiniLM should be about 3–4× faster on CPU.
- **CPU inference:** scoring 2,974 test sentences takes ≈ 1–2 minutes for a base model at batch 64.
- **RAM (7.4 GB):** this is fine. Keep DataLoader `num_workers` at 2–4 to avoid duplicating memory.
  Pre-tokenize the data once.

### Minimal plain-PyTorch checklist
1. Read the 3 JSONL files (en-US, hi-IN, kn-IN). Keep `id, partition, intent, utt`. Build the intent→index map from **en-US train only** and check that all 60 intents are present.
2. Tokenize once and save the tensors. Use dynamic padding in `collate_fn`.
3. Build the model: `AutoModel` (MuRIL) + linear head. Set `embeddings.word_embeddings.weight.requires_grad=False`.
4. Use AdamW over trainable params only, a linear warmup/decay scheduler, `torch.autocast('cuda', dtype=torch.float16)` + `GradScaler`, and `clip_grad_norm_(…, 1.0)`.
5. Each epoch: compute en-US dev accuracy (for selection) and log hi/kn dev accuracy (oracle only). Save the best en-dev checkpoint.
6. Final step: report en-US, hi-IN and kn-IN **test** accuracy for 3 seeds. Compare with MASSIVE Table 8 (XLM-R zero: hi 74.8, kn 63.5) and IndicXTREME Table 19 (MuRIL: hi 82.4, kn 77.5).

## Sources (all checked 2026-11-22)
- MASSIVE repo/README/NOTICE: https://github.com/alexa/massive
- MASSIVE HF card: https://huggingface.co/datasets/AmazonScience/massive
- MASSIVE paper (Tables 3, 5, 6, 8): https://arxiv.org/abs/2204.08582 , HTML https://arxiv.org/html/2204.08582
- IndicXTREME / IndicBERT v2 (Table 19, Appendix N): https://arxiv.org/abs/2212.05409
- MuRIL paper: https://arxiv.org/abs/2103.10730 ; card https://huggingface.co/google/muril-base-cased
- XLM-R paper (Tables 1, 6, 7): https://arxiv.org/abs/1911.02116 ; card https://huggingface.co/FacebookAI/xlm-roberta-base
- Multilingual-MiniLM card: https://huggingface.co/microsoft/Multilingual-MiniLM-L12-H384
- DistilBERT multilingual: https://huggingface.co/distilbert/distilbert-base-multilingual-cased
- Keung et al. 2020: https://aclanthology.org/2020.emnlp-main.40
- Gaschi et al. 2023: https://arxiv.org/abs/2306.02790
