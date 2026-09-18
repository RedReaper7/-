PYTHON ?= python3.11

.PHONY: install bootstrap train api test

install:
	$(PYTHON) -m pip install -e ".[api,geo,dev]"

bootstrap:
	$(PYTHON) scripts/bootstrap_data.py

train:
	$(PYTHON) scripts/train_model.py

api:
	$(PYTHON) scripts/run_api.py

test:
	pytest

