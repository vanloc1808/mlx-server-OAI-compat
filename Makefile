run:
	mlx-server launch \
	--model-path mlx-community/gemma-3-4b-it-qat-4bit \
	--model-type vlm \
	--max-concurrency 1 \
	--queue-timeout 300 \
	--queue-size 100

install:
	pip install -e .