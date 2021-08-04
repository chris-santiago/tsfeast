.PHONY: lint
lint:
	pylint tsfeast --verbose

.PHONY: tests
tests:
	pytest --cov=tsfeast --cov-report=html --verbose

.PHONY: type
type:
	mypy -p tsfeast

.PHONY: docs
docs:
	sphinx-apidoc tsfeast -o docs/source/
	sphinx-build -b html docs/source/ docs/build/html

.PHONY: manifest
manifest:
	check-manifest

.PHONY: precommit
precommit:
	pre-commit run trailing-whitespace --all-files
	pre-commit run end-of-file-fixer --all-files
	pre-commit run check-yaml --all-files
	pre-commit run check-added-large-files --all-files
	pre-commit run isort --all-files

.PHONY: checks
checks: lint tests type docs manifest precommit