# R3: Serving (gunicorn + nginx), load testing (Locust), video tooling

Machine: WSL2 Ubuntu 24.04, 12 cores, 7.4 GB RAM, nginx 1.24 (apt), ffmpeg 6.1.1, Python 3.12.
Every fact says where it came from and "checked 2026-11-22". "Tested locally" means I ran it on this machine in a scratch folder.

Current versions on PyPI (from `https://pypi.org/pypi/<name>/json`, checked 2026-11-22):
gunicorn 27.0.1, locust 2.47.0, piper-tts 1.8.0, Pillow 13.0.0, Pygments 2.21.0, torch 2.15.0.

---

## 1. Serving a ~280M-param encoder (~1.1 GB fp32) with Flask + gunicorn on CPU

### What the docs say
- `workers` default is 1 (or `$WEB_CONCURRENCY`). The docs suggest "2-4 x $(NUM_CORES)". Source: gunicorn `config.py` (`Workers`), https://github.com/benoitc/gunicorn/blob/master/gunicorn/config.py (checked 2026-11-22).
- The design page suggests `(2 x CPU cores) + 1` only as a *starting point*, says "Gunicorn typically needs only 4-12 workers", and warns too many workers "waste resources and can reduce throughput". It recommends the `sync` worker for "CPU-bound apps behind a proxy". https://gunicorn.org/design/ (checked 2026-11-22). (Old URL docs.gunicorn.org now redirects to gunicorn.org.)
- These rules are for I/O-bound web apps. Each worker is a separate process, so each one holds its **own copy of the model** unless shared. 13 workers x ~1.5 GB = ~20 GB: impossible on 7.4 GB. For ML, memory decides the worker count, not the formula.
- `worker_class` default `sync`. `threads` default 1; if you set `threads > 1` with `sync`, gunicorn silently switches to `gthread`. `config.py` (`WorkerClass`, `WorkerThreads`), checked 2026-11-22.
- `timeout` default 30 s: a worker silent longer than this is killed and restarted. For `sync` workers this is effectively the max time per request. `graceful_timeout` default 30 s (time to finish requests after a restart signal). `keepalive` default 2 s, and the `sync` worker ignores keep-alive entirely. `config.py` (`Timeout`, `GracefulTimeout`, `Keepalive`), checked 2026-11-22.
- `max_requests` default 0 (off): restart a worker after N requests to limit memory leaks; pair with `max_requests_jitter`. `config.py`, checked 2026-11-22.
- `preload_app` (`--preload`) default False: "Load application code before the worker processes are forked ... save some RAM ... speed up boot". Downside: code reload by restarting workers no longer picks up changes. `config.py` (`Preload`), checked 2026-11-22.

### PyTorch threads and oversubscription
- PyTorch docs: with N vCPUs and M processes, each process should use at most `floor(N/M)` threads, to avoid CPU oversubscription (more busy threads than cores -> contention and slowdowns). https://docs.pytorch.org/docs/2.15/notes/multiprocessing.html (checked 2026-11-22).
- By default every worker's torch uses all 12 cores. 3 workers x 12 threads = 36 busy threads on 12 cores: bad. So call `torch.set_num_threads(k)` in each worker (and/or set `OMP_NUM_THREADS=k`, `MKL_NUM_THREADS=k` in the environment before Python starts).

### preload_app + PyTorch caveats (copy-on-write)
- After `fork`, parent and child share memory pages until one of them writes to a page (copy-on-write). Tensor weight data is big buffers that inference does not write, so it *can* stay shared. But Python object refcounts live in object headers and get updated constantly, so some pages get copied. Real savings must be measured (check `PSS` with `smem` or `/proc/<pid>/smaps_rollup`), not assumed. Summary of search results incl. https://discuss.pytorch.org/t/infer-torch-model-via-gunicorn-wsgi/60437 (checked 2026-11-22).
- Fork + threads danger: "If there's any thread that holds a lock or imports a module, and fork is called, it's very likely that the subprocess will be in a corrupted state and will deadlock." PyTorch multiprocessing notes (link above), checked 2026-11-22. Torch's OpenMP/MKL thread pool started in the master (e.g. by running a warm-up inference before fork) is exactly this risk.
- Gunicorn issue: preloading PyTorch can hang; maintainer advice is to make sure no extra threads run at fork time. https://github.com/benoitc/gunicorn/issues/2478 (checked 2026-11-22).
- Haystack users saw preload-related hangs/timeouts go away with `torch.set_num_threads(1)` before forking. https://github.com/deepset-ai/haystack/issues/2409 (checked 2026-11-22).
- Never touch CUDA in the master before fork (not relevant on CPU, but good to know). https://github.com/pytorch/pytorch/issues/49555 (checked 2026-11-22).

Simple safe rule: either (a) **no preload**, load the model inside each worker (e.g. lazily, or in a `post_fork` hook), or (b) preload but **do not run any inference in the master**, set threads in `post_fork`, and test for hangs.

### Concrete recommendation for 12 cores / 7.4 GB RAM
Memory budget per worker: ~1.1 GB weights + ~0.4-0.6 GB Python/torch/tokenizer/activations = ~1.5-1.7 GB. Leave ~1.5-2 GB for OS, nginx, Locust.

- **Start with 3 sync workers x 4 torch threads** (3 x 4 = 12 = cores). ~5 GB total without preload. Safe.
- Try 2 workers x 6 threads as well; pick by measured p95 under Locust. 4 workers x 3 threads only if PSS per worker shows it fits (~6+ GB is too tight).
- `worker_class = "sync"` (CPU-bound, behind nginx). Do not use gthread threads for more concurrency: extra requests in one process just fight for the same torch threads; nginx/backlog queues them instead.
- `timeout = 120` (model load in a fresh worker can take > 30 s on WSL2 disk; a single long batch should not get the worker killed). `graceful_timeout = 30`. `keepalive = 2` (ignored by sync anyway).
- `max_requests = 1000`, `max_requests_jitter = 100` as a leak guard (costs a model reload each time; raise if reloads hurt).
- Bind `127.0.0.1:8000` (only nginx talks to it).
- Optional speed win: `torch.inference_mode()` around forward passes; dynamic int8 quantization (`torch.ao.quantization.quantize_dynamic`) can roughly quarter the Linear weight memory and allow more workers (measure accuracy).

```python
# gunicorn.conf.py  (sketch)
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
bind = "127.0.0.1:8000"
workers = 3
worker_class = "sync"
timeout = 120
graceful_timeout = 30
keepalive = 2
max_requests = 1000
max_requests_jitter = 100
preload_app = False          # each worker loads its own model: simple and fork-safe

def post_fork(server, worker):
    import torch
    torch.set_num_threads(4)          # floor(12 cores / 3 workers)
    torch.set_num_interop_threads(1)
```
Run: `gunicorn -c gunicorn.conf.py app:app`. WSL2 note: RAM seen by WSL is capped by `.wslconfig` (`memory=`); 7.4 GB is what Linux sees, check with `free -h`.

---

## 2. nginx as a non-root user with a project-local config

### Facts
- Switches: `-p prefix` (path prefix, default `/usr/local/nginx`), `-c file`, `-e file` (alternative error log, `stderr` allowed, **added in 1.19.5**, so OK on 1.24), `-g directives`, `-t` (test), `-T` (test + dump), `-s stop|quit|reload|reopen`. https://nginx.org/en/docs/switches.html (checked 2026-11-22).
- Ubuntu's nginx 1.24 build (from `nginx -V`, tested locally 2026-11-22): `--prefix=/usr/share/nginx`, `--pid-path=/run/nginx.pid`, `--error-log-path=stderr`, `--http-log-path=/var/log/nginx/access.log`, temp paths under `/var/lib/nginx/{body,proxy,fastcgi,uwsgi,scgi}`. A normal user cannot write any of these, so the config must override **all** of them: `pid`, `error_log`, `access_log`, `client_body_temp_path`, `proxy_temp_path`, `fastcgi_temp_path`, `uwsgi_temp_path`, `scgi_temp_path`. nginx creates temp dirs itself at startup if the parent is writable (tested locally).
- Relative paths in the config are resolved against the `-p` prefix. Without `-p`, they resolve against `/usr/share/nginx` and fail: `open() "/usr/share/nginx/logs/nginx.pid" failed` (tested locally 2026-11-22). The `-p` value should end with `/`.
- A `user` directive while not root gives: `[warn] the "user" directive makes sense only if the master process runs with super-user privileges, ignored` (tested locally 2026-11-22). Just leave `user` out.
- Ports below 1024 need root; 8080 is fine.
- `-s stop` / `-s reload` find the master through the pid file, so they **must use the same `-p` and `-c`** (else nginx looks at `/run/nginx.pid`). Tested locally 2026-11-22.
- Defaults worth knowing: `client_max_body_size 1m` (larger body -> 413), `proxy_connect_timeout 60s`, `proxy_read_timeout 60s`, `proxy_send_timeout 60s`. https://nginx.org/en/docs/http/ngx_http_core_module.html#client_max_body_size and https://nginx.org/en/docs/http/ngx_http_proxy_module.html (checked 2026-11-22).
- Upstream keep-alive needs `keepalive N` in the `upstream` block plus `proxy_http_version 1.1` and an empty `Connection` header (proxy module docs, `keepalive` in https://nginx.org/en/docs/http/ngx_http_upstream_module.html#keepalive, checked 2026-11-22). Gunicorn sync workers close connections anyway, so this mainly matters if you switch worker type.

### Minimal working config (tested locally 2026-11-22: `-t` ok, start ok, request proxied, `-s stop` ok)
Layout: `deploy/nginx/{conf/nginx.conf, logs/, tmp/}` (make `logs/` and `tmp/` first).
```nginx
# conf/nginx.conf  -- no "user" line
worker_processes 2;
pid        logs/nginx.pid;
error_log  logs/error.log warn;
events { worker_connections 1024; }
http {
  access_log logs/access.log;
  client_body_temp_path tmp/client_body;
  proxy_temp_path       tmp/proxy;
  fastcgi_temp_path     tmp/fastcgi;
  uwsgi_temp_path       tmp/uwsgi;
  scgi_temp_path        tmp/scgi;
  upstream app { server 127.0.0.1:8000; keepalive 16; }
  server {
    listen 127.0.0.1:8080;
    client_max_body_size 1m;
    location / {
      proxy_pass http://app;
      proxy_http_version 1.1;
      proxy_set_header Connection "";
      proxy_set_header Host $host;
      proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
      proxy_connect_timeout 5s;
      proxy_send_timeout 60s;
      proxy_read_timeout 130s;   # a bit above gunicorn timeout=120
    }
  }
}
```
Commands (run from anywhere, P = absolute path of `deploy/nginx/`):
```bash
nginx -p "$P/" -c conf/nginx.conf -e logs/error.log -t      # test
nginx -p "$P/" -c conf/nginx.conf -e logs/error.log         # start (daemonizes)
nginx -p "$P/" -c conf/nginx.conf -e logs/error.log -s reload
nginx -p "$P/" -c conf/nginx.conf -e logs/error.log -s stop
```
`-e` matters because nginx opens the error log *before* it reads the config; on builds whose default is a file in `/var/log/nginx` you get a startup alert without it. (Ubuntu's default is `stderr`, so it is harmless there, but keep `-e` for portability.) With no backend running you get `502` and `connect() failed (111: Connection refused)` in `logs/error.log` (tested).

---

## 3. Locust headless mode

- Version: **locust 2.47.0** (PyPI, released 2026-10-21, checked 2026-11-22).
- Flags: `--headless` (no web UI, start at once), `-u/--users` (peak users), `-r/--spawn-rate` (users started per second), `-t/--run-time` (e.g. `60s`, `5m`, `1h30m`; default forever), `-H/--host`, `--csv PREFIX`, `--csv-full-history` (all rows per name into history, needs `--csv`), `--only-summary` (no periodic table, only final), `--exit-code-on-error` (default 1). https://docs.locust.io/en/stable/configuration.html (checked 2026-11-22).

```bash
locust -f loadtest/locustfile.py --headless -u 20 -r 5 -t 60s \
  --host http://127.0.0.1:8080 --csv results/run1 --only-summary
```

Files written (`--csv results/run1`), from locust source `locust/stats.py`, https://github.com/locustio/locust/blob/master/locust/stats.py (checked 2026-11-22):
- `run1_stats.csv` columns: `Type, Name, Request Count, Failure Count, Median Response Time, Average Response Time, Min Response Time, Max Response Time, Average Content Size, Requests/s, Failures/s, 50%, 66%, 75%, 80%, 90%, 95%, 98%, 99%, 99.9%, 99.99%, 100%`. One row per request name, then a final row with `Name == "Aggregated"` (Type empty) for totals.
- p50 = column `"50%"`, p95 = column `"95%"`, p99 = `"99%"` (all in milliseconds, integers; `N/A` if no requests). Throughput = `"Requests/s"`; errors = `"Failure Count"`.
- `run1_stats_history.csv`: one row per interval (`Timestamp, User Count, Type, Name, Requests/s, Failures/s, <percentiles>, Total Request Count, Total Failure Count, ...`). By default only Aggregated rows; all names with `--csv-full-history`.
- `run1_failures.csv`: `Method, Name, Error, Occurrences` (+ First/Last Seen in current source).
- `run1_exceptions.csv`: `Count, Message, Traceback, Nodes`.

Parsing sketch:
```python
import csv
row = next(r for r in csv.DictReader(open("results/run1_stats.csv")) if r["Name"] == "Aggregated")
p50, p95, rps, fails = row["50%"], row["95%"], float(row["Requests/s"]), int(row["Failure Count"])
```
Tip: pick column names by header, not position (column set grew over versions). Percentiles are approximate (bucketed response times).

---

## 4. Video tooling

### Piper TTS (piper-tts 1.8.0, OHF-Voice/piper1-gpl)
Source: https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/API_PYTHON.md and `src/piper/config.py` (checked 2026-11-22). Voices: `python3 -m piper.download_voices en_US-lessac-medium` (needs `.onnx` + `.onnx.json` side by side).
```python
import wave
from piper import PiperVoice, SynthesisConfig
voice = PiperVoice.load("voices/en_US-lessac-medium.onnx")
cfg = SynthesisConfig(length_scale=1.1)   # >1 slower, <1 faster
with wave.open("seg01.wav", "wb") as wf:
    voice.synthesize_wav("Hello from the router demo.", wf, syn_config=cfg)
```
`SynthesisConfig` fields: `speaker_id`, `length_scale`, `noise_scale`, `noise_w_scale`, `normalize_audio=True`, `volume=1.0`. Streaming: `for chunk in voice.synthesize(text): chunk.audio_int16_bytes` (plus `sample_rate`, `sample_width`, `sample_channels`). Note: this is the GPL 1.x rewrite; old `rhasspy/piper` CLI examples differ.

### Pillow text slides with wrapping + auto-shrink (Pillow 13.0.0)
- `ImageDraw.textlength(text, font)` -> float width (no newlines allowed). `ImageDraw.textbbox((x,y), text, font, spacing=4)` -> `(left, top, right, bottom)`, TrueType only, handles newlines via `multiline_textbbox`. https://pillow.readthedocs.io/en/stable/reference/ImageDraw.html (checked 2026-11-22).
- `textsize()`, `multiline_textsize()` and `ImageFont.getsize()` were **removed in Pillow 10.0.0**; old blog code using them breaks. https://pillow.readthedocs.io/en/stable/releasenotes/10.0.0.html (checked 2026-11-22).
- Font: `ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)` (file present on this machine, checked locally).

Approach: greedy word wrap with `textlength`, then measure the block with `multiline_textbbox`; if it is too tall or wide, shrink the font size and retry.
```python
from PIL import Image, ImageDraw, ImageFont
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
def wrap(d, text, font, max_w):
    lines = []
    for para in text.split("\n"):
        cur = ""
        for w in para.split():
            t = (cur + " " + w).strip()
            if d.textlength(t, font=font) <= max_w or not cur: cur = t
            else: lines.append(cur); cur = w
        lines.append(cur)
    return "\n".join(lines)
def slide(text, out, W=1280, H=720, pad=60, start=56, min_size=18):
    img = Image.new("RGB", (W, H), "#101820"); d = ImageDraw.Draw(img)
    for size in range(start, min_size - 1, -2):
        font = ImageFont.truetype(FONT, size)
        body = wrap(d, text, font, W - 2 * pad)
        l, t, r, b = d.multiline_textbbox((0, 0), body, font=font, spacing=size // 3)
        if r - l <= W - 2 * pad and b - t <= H - 2 * pad: break
    d.multiline_text((pad, pad), body, font=font, fill="white", spacing=size // 3)
    img.save(out)
```
Use even width/height (yuv420p needs them), e.g. 1280x720.

### Pygments code images
- `ImageFormatter` options: `image_format` (PNG), `font_name` (default "DejaVu Sans Mono" on *nix), `font_size` 14, `line_pad` 2, `line_numbers` True, `line_number_bg "#eed"`, `line_number_fg "#886"`, `image_pad` 10, `hl_lines` []; `style` is the general formatter option (default `"default"`, e.g. `style="monokai"`). https://pygments.org/docs/formatters/ (checked 2026-11-22).
- On Linux it finds the font by running `fc-list "<name>:style=<style>" file`, and raises `FontNotFound('No usable fonts named: "..."')` if nothing matches. So `font_name` must be a **fontconfig family name** ("DejaVu Sans Mono"), not a file path, and `fc-list` must exist. Source: `pygments/formatters/img.py`, https://github.com/pygments/pygments/blob/master/pygments/formatters/img.py (checked 2026-11-22). On this machine `fc-match "DejaVu Sans Mono"` -> `DejaVuSansMono.ttf` (checked locally). Needs Pillow installed.
```python
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import ImageFormatter
png = highlight(code, PythonLexer(), ImageFormatter(font_name="DejaVu Sans Mono",
      font_size=22, line_numbers=True, style="monokai", image_pad=20))
open("code.png", "wb").write(png)   # then paste onto a 1280x720 canvas with Pillow
```
Image size follows the code length, so paste/scale it onto a fixed canvas before ffmpeg.

### ffmpeg (6.1.1) – all commands tested locally 2026-11-22
Still image + audio -> clip:
```bash
ffmpeg -y -loop 1 -i slide01.png -i seg01.wav -c:v libx264 -tune stillimage \
  -pix_fmt yuv420p -r 30 -c:a aac -b:a 128k -shortest clip01.mp4
```
(`-loop 1` repeats the image forever, `-shortest` stops at the audio end, `yuv420p` for player compatibility.) Tune docs: https://trac.ffmpeg.org/wiki/Encode/H.264 (checked 2026-11-22; site sometimes blocks bots).

Concat demuxer (no re-encode):
```bash
printf "file 'clip01.mp4'\nfile 'clip02.mp4'\n" > list.txt
ffmpeg -y -f concat -safe 0 -i list.txt -c copy final.mp4
```
"All files must have the same streams (same codecs, same time base, etc.)"; `safe` defaults to 1 and rejects unsafe paths (absolute paths etc.), hence `-safe 0`. https://ffmpeg.org/ffmpeg-formats.html#concat-1 (checked 2026-11-22). So encode every clip with the same size, fps, pix_fmt, audio rate/channels. Observed: 2.0 s + 3.0 s clips gave a 5.046 s file (small AAC padding per clip), so do not expect exact sums.

Duration:
```bash
ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 final.mp4
# -> 5.046440
```
Loudness check (output goes to **stderr**):
```bash
ffmpeg -hide_banner -i final.mp4 -vn -af volumedetect -f null - 2>&1 | grep -E "mean_volume|max_volume"
# [Parsed_volumedetect_0 @ 0x...] mean_volume: -21.2 dB
# [Parsed_volumedetect_0 @ 0x...] max_volume: -17.6 dB
```
Parse with `re.search(r"mean_volume:\s*(-?[\d.]+) dB", stderr)`. Silence shows as `-91.0 dB` or lower; use that to catch an empty TTS track. Filter docs: https://ffmpeg.org/ffmpeg-filters.html#volumedetect (checked 2026-11-22).

### SRT subtitles
Blocks of: number (from 1), `HH:MM:SS,mmm --> HH:MM:SS,mmm` (comma before ms), one or more text lines, blank line. https://en.wikipedia.org/wiki/SubRip (checked 2026-11-22).
```
1
00:00:00,000 --> 00:00:04,250
First narrated sentence.

2
00:00:04,250 --> 00:00:09,100
Second sentence.
```
Build timings by adding each clip's ffprobe duration (use the clip duration, not the wav, to follow the AAC drift). Burn in: `ffmpeg -i final.mp4 -vf subtitles=subs.srt -c:a copy out.mp4` (needs libass), or mux soft subs into mp4 with `-c:s mov_text`.
