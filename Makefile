# Backend Makefile.

VENV_DIR := skin-lesion-env

ifeq ($(OS),Windows_NT)
	PYTHON      := py -3.13
	VENV_PYTHON := $(VENV_DIR)\Scripts\python.exe
	VENV_PIP    := $(VENV_DIR)\Scripts\pip.exe
else
	PYTHON      := python3.13
	VENV_PYTHON := $(VENV_DIR)/bin/python
	VENV_PIP    := $(VENV_DIR)/bin/pip
endif

.PHONY: help setup install install-dev run test lint typecheck clean

help:
	@echo "Available targets:"
	@echo "  setup        - Create skin-lesion-env/ and install dev packages"
	@echo "  install      - Install production packages only (requirements.txt)"
	@echo "  install-dev  - Re-sync dev packages (requirements-dev.txt)"
	@echo "  run          - Start FastAPI dev server on :8000"
	@echo "  test         - Run pytest"
	@echo "  lint         - Run ruff"
	@echo "  typecheck    - Run mypy"
	@echo "  clean        - Remove skin-lesion-env/ and __pycache__"

setup:
	@echo "Creating venv at $(VENV_DIR) ..."
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -r requirements-dev.txt
	@echo "Done."

install:
	$(VENV_PYTHON) -m pip install -r requirements.txt

install-dev:
	$(VENV_PYTHON) -m pip install -r requirements-dev.txt

run:
	$(VENV_PYTHON) -m uvicorn app.main:app --reload --port 8000

test:
	$(VENV_PYTHON) -m pytest

lint:
	$(VENV_PYTHON) -m ruff check .

typecheck:
	$(VENV_PYTHON) -m mypy app/

clean:
	cmd /c "rmdir /s /q $(VENV_DIR)" 2>nul || rm -rf $(VENV_DIR)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
