# Makefile for cuBLASter

.PHONY: all build install clean test examples help cuda-check

# Default target
all: build

help:
	@echo "cuBLASter Makefile"
	@echo "=================="
	@echo ""
	@echo "Targets:"
	@echo "  all         - Build the project (default)"
	@echo "  build       - Build cuBLASter with CUDA support"
	@echo "  install     - Install cuBLASter in development mode"
	@echo "  clean       - Clean build artifacts"
	@echo "  test        - Run tests"
	@echo "  examples    - Run example programs"
	@echo "  cuda-check  - Check CUDA installation"
	@echo "  help        - Show this help message"

cuda-check:
	@echo "Checking CUDA installation..."
	@nvcc --version || (echo "Error: nvcc not found. Please install CUDA toolkit." && exit 1)
	@nvidia-smi || (echo "Warning: nvidia-smi not found. GPU may not be available.")
	@echo "CUDA check completed."

build: cuda-check
	@echo "Building cuBLASter..."
	python -m pip install -e .
	@echo "Build completed."

install: build
	@echo "cuBLASter installed in development mode."

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -name "*.so" -delete
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.c" -path "*/core/*" -delete
	find . -name "*.cpp" -path "*/core/*" -not -name "*.cu" -delete
	@echo "Clean completed."

test: build
	@echo "Running tests..."
	python examples/simple_example.py
	@echo "Tests completed."

examples: build
	@echo "Running examples..."
	@echo "1. Simple matrix multiplication example:"
	python examples/simple_example.py
	@echo ""
	@echo "2. Lattice reduction example:"
	cat examples/lattice_example.txt | python src/cublaster/app.py -pq
	@echo "Examples completed."

# Development targets
dev-install:
	python -m pip install -e ".[dev]"

lint:
	flake8 src/ examples/
	black --check src/ examples/

format:
	black src/ examples/

# Wheel building
wheel: cuda-check
	python -m build --wheel

# Docker targets (if needed)
docker-build:
	docker build -t cublaster .

docker-run:
	docker run --gpus all -it cublaster