PYTHON_VERSION := 3.9

.PHONY: build
build: requirements.txt
	@git submodule update --init --recursive
	@uv venv --python ${PYTHON_VERSION}
	@uv sync
	@uv pip install -r robosuite/requirements-extra.txt
	@uv pip install -r requirements.txt