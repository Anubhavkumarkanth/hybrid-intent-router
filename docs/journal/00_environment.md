# Phase 0: Environment check and project skeleton

## What we did
- Checked the machine and saved everything to `logs/00_env.log`.
- Created a Python virtual environment `.venv` and installed PyTorch with CUDA (the laptop has an NVIDIA GPU).
- Created the folder structure, the `.gitignore` and a first git commit.

## Why
- We must know what the machine can do before planning: CPU cores decide gunicorn workers,
  RAM decides how many model copies fit, the GPU decides how fast training is.
- `.gitignore` must contain `.env` **before** the first commit, so the API key can never reach git.

## Key commands
```bash
uname -a; cat /etc/os-release; nproc; free -h; nvidia-smi
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
git init -b main && git add .gitignore && git commit -m "Add .gitignore"
```

## Result (from logs/00_env.log)
- Ubuntu 24.04.5 LTS on WSL2, Python 3.12.3, 12 CPU cores, 7.4 GiB RAM, 954 GB free disk.
- GPU: NVIDIA GeForce RTX 3050 Laptop, 4 GB VRAM.
- nginx 1.24.0, ffmpeg 6.1.1, git 2.43.0, make 4.3, fontconfig and DejaVu fonts: all present.
- `.env` has `GEMINI_API_KEY` (only presence checked, never printed).

## Errors and fixes
- The CUDA PyTorch download (~3 GB of wheels) was very slow and did not finish within the 10-minute tool limit.
  Fix: let it continue in the background and write code (data pipeline, LLM client, tests) in parallel.

## How to explain this in an interview
1. I started by checking the hardware, because it decides the design: 12 cores and 7.4 GB RAM limit how many model copies a server can hold.
2. I used a virtual environment so the project's packages never mix with the system Python.
3. The very first commit was the `.gitignore`, so the secret API key file could never be committed by accident.
4. I logged every check to a file, so the results are evidence, not memory.
