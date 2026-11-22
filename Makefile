# Hybrid Intent Router — main commands. Always uses the project venv.
SHELL := /bin/bash
.SHELLFLAGS := -o pipefail -c   # a failing command piped into tee must still fail
PY := .venv/bin/python
export PYTHONPATH := $(CURDIR)/src:$(CURDIR)
export PYTHONUNBUFFERED := 1   # so `| tee` logs show progress live
WORKERS ?= 3
export GUNICORN_WORKERS := $(WORKERS)
RUN := $(CURDIR)/deploy/nginx/run
NGINX_PREFIX := $(CURDIR)/deploy/nginx
NGINX := nginx -p "$(NGINX_PREFIX)/" -c nginx.conf -e "$(CURDIR)/logs/nginx_error.log"

.PHONY: help setup data llm-pilot llm llm-single smoke train evaluate route tables test lint verify \
        serve serve-bg nginx-start nginx-stop stack-up stack-down loadtest video video-check clean-run

help:
	@grep -E '^[a-z-]+:' Makefile | cut -d: -f1 | tr '\n' ' '; echo

setup:
	python3 -m venv .venv
	$(PY) -m pip install -r requirements.txt -r requirements-dev.txt

data:
	$(PY) -m router.data 2>&1 | tee logs/02_data.log

llm-pilot:
	$(PY) -m router.llm_baseline pilot 2>&1 | tee logs/03_llm_pilot.log

llm:
	$(PY) -m router.llm_baseline run 2>&1 | tee logs/03_llm_run.log

llm-single:
	$(PY) -m router.llm_baseline single 2>&1 | tee logs/03_llm_single.log

smoke:
	$(PY) -m router.train --smoke 30 2>&1 | tee logs/04_smoke.log

train:
	$(PY) -m router.train 2>&1 | tee logs/04_train.log

evaluate:
	$(PY) -m router.evaluate 2>&1 | tee logs/04_evaluate.log

route:
	$(PY) -m router.analysis 2>&1 | tee logs/05_route.log

tables:
	$(PY) scripts/render_tables.py

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .

verify:
	$(PY) -m ruff check .
	$(PY) -m pytest -q
	$(PY) scripts/check_numbers.py

smoke-api:
	bash scripts/smoke_api.sh 2>&1 | tee logs/06_smoke.log

# ---- serving: gunicorn on 127.0.0.1:8000, NGINX (no sudo) on 127.0.0.1:8080
serve:
	$(PY) -m gunicorn -c deploy/gunicorn.conf.py

serve-bg:
	mkdir -p $(RUN) logs
	$(PY) -m gunicorn -c deploy/gunicorn.conf.py --daemon --pid $(RUN)/gunicorn.pid \
		--access-logfile logs/gunicorn_access.log --error-logfile logs/gunicorn_error.log
	@for i in $$(seq 90); do curl -sf http://127.0.0.1:8000/health >/dev/null && exit 0; sleep 1; done; \
		echo "gunicorn did not become healthy, see logs/gunicorn_error.log"; exit 1

nginx-start:
	mkdir -p $(RUN) $(NGINX_PREFIX)/temp $(NGINX_PREFIX)/logs logs
	$(NGINX) -t
	$(NGINX)

nginx-stop:
	@if [ -f $(RUN)/nginx.pid ]; then $(NGINX) -s stop && echo "nginx stopped"; else echo "nginx not running"; fi

stack-up: serve-bg nginx-start
	curl -sf http://127.0.0.1:8080/health && echo "  <- /health through NGINX"

stack-down: nginx-stop
	@if [ -f $(RUN)/gunicorn.pid ]; then kill $$(cat $(RUN)/gunicorn.pid) && echo "gunicorn stopped"; \
		else echo "gunicorn not running"; fi
	@for i in $$(seq 30); do [ -f $(RUN)/gunicorn.pid ] || exit 0; sleep 1; done; echo "gunicorn still running"; exit 1

video:
	$(PY) scripts/make_diagram.py
	$(PY) video/build_video.py 2>&1 | tee logs/10_video.log

video-check:
	$(PY) video/check_video.py 2>&1 | tee logs/10_video_check.log

# Load test through NGINX. Start the stack first with: ROUTER_DISABLE_LLM=1 make stack-up
LT_USERS ?= 20
LT_RATE ?= 5
LT_TIME ?= 2m
loadtest:
	$(PY) -m locust -f deploy/locustfile.py --headless -u $(LT_USERS) -r $(LT_RATE) -t $(LT_TIME) \
		--host http://127.0.0.1:8080 --csv results/loadtest --only-summary 2>&1 | tee logs/07_loadtest.log
	$(PY) scripts/loadtest_summary.py '{"users": $(LT_USERS), "spawn_rate": $(LT_RATE), "duration": "$(LT_TIME)", "workers": $(WORKERS), "torch_threads": '$$(( $$(nproc) / $(WORKERS) ))'}' | tee logs/07_loadtest_summary.log
