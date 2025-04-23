run:
	mlx-server launch \
	--model-path mlx-community/Qwen2.5-VL-3B-Instruct-4bit \
	--model-type vlm \
	--max-concurrency 1 \
	--queue-timeout 300 \
	--queue-size 100

install:
	pip install -e .