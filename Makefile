# Hybrid Intent Router — main commands. Always uses the project venv.
SHELL := /bin/bash
PY := .venv/bin/python
export PYTHONPATH := $(CURDIR)/src:$(CURDIR)
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
