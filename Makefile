.PHONY: setup train train-offline train-url train-email train-online collect-online prepare-url prepare-email test lint api dashboard

PYTHON ?= python

setup:
	$(PYTHON) -m pip install -e ".[dev]"

train: train-url train-email train-online

train-offline: train-url train-email

train-url:
	$(PYTHON) -m phishlens.models.train_url --data data/urls.csv --output artifacts

prepare-url:
	$(PYTHON) -m phishlens.models.prepare_url

prepare-email:
	$(PYTHON) -m phishlens.models.prepare_email

train-email:
	$(PYTHON) -m phishlens.models.train_email --data data/emails.csv --output artifacts

collect-online:
	$(PYTHON) -m phishlens.models.precompute_online --max-domains 100

train-online:
	$(PYTHON) -m phishlens.models.train_url_online

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

api:
	$(PYTHON) -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --no-access-log --no-proxy-headers

dashboard:
	$(PYTHON) -m streamlit run dashboard/app.py
