lint:
	ruff check rangebench
	ruff format --check rangebench
	mypy rangebench

fmt:
	ruff format rangebench
	ruff check --fix rangebench

check: lint
	python3 -m py_compile rangebench/*.py
	python3 -m rangebench list > /dev/null
	python3 -m unittest discover -s tests
