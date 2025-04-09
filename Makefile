run:
	python -m app.main \
	--model-path mlx-community/gemma-3-4b-it-4bit \
	--max-concurrency 1 \
	--queue-timeout 300 \
	--queue-size 100

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"