run:
	fastapi dev

build-image:
	docker build -t maguowei/qwen2-vl-api .
