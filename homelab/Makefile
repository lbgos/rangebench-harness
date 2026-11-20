SRC = rangebench_homelab tests templates

lint:
	ruff check $(SRC)
	ruff format --check $(SRC)
	mypy rangebench_homelab

fmt:
	ruff format $(SRC)
	ruff check --fix $(SRC)

check: lint
	python3 -m py_compile rangebench_homelab/*.py
	python3 -m rangebench_homelab --help > /dev/null
	python3 -m unittest discover -s tests -t .
