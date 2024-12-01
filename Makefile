#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = intel_img_classification
PYTHON_VERSION = 3.11.5
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python Dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Install Graphviz using Homebrew
.PHONY: install-graphviz
install-graphviz:
	@echo "Installing Graphviz..."
	brew install graphviz || echo "Graphviz is already installed."

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 intel_img_classification
	isort --check --diff --profile black intel_img_classification
	black --check --config pyproject.toml intel_img_classification

## Format source code with black
.PHONY: format
format:
	black --config pyproject.toml intel_img_classification

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Install all project requirements, including Graphviz and Python dependencies
.PHONY: install
install: install-graphviz requirements
	@echo "All dependencies installed successfully!"

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)