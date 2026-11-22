# Data

**Amazon MASSIVE 1.1** (FitzGerald et al., 2022), license **CC BY 4.0**.
Source: https://github.com/alexa/massive

- Tarball: https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz
- Size: 40,251,390 bytes
- SHA-256: `4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577` (downloaded 2026-11-22)

`make data` downloads it to `data/raw/` (not in git), checks the SHA-256, and writes:

| File | What |
|---|---|
| `processed/{en,hi,kn}_{train,dev,test}.jsonl` | `id`, `text`, `intent`, `label` (not in git; rebuilt by `make data`) |
| `processed/labels.json` | the 60 intent names, sorted (label id = position) |
| `processed/dev_sub_ids.json` | 300 dev ids (seed 42, stratified by intent). Used to choose prompt, temperature check, threshold |
| `processed/test_sub_ids.json` | 500 test ids (seed 42, stratified). Used only for final numbers |

The same ids are used for English, Hindi and Kannada (MASSIVE ids are parallel translations).
