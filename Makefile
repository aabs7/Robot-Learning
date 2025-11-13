VENV := venv
VENV_PYTHON = $(VENV)/bin/python3
VENV_PIP = $(VENV)/bin/pip

.PHONY: build
build: requirements.txt
	@git submodule update --init --recursive
	@python3 -m venv $(VENV)
	@$(VENV_PIP) install -r requirements.txt
	@$(VENV_PIP) install -r robosuite/requirements.txt
	@$(VENV_PIP) install -r robosuite/requirements-extra.txt
	@$(VENV_PIP) install robosuite/