# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

tritonBLAS is a lightweight Triton-based GEMM (General Matrix Multiplication) library that uses an analytical model (Origami) to predict optimal kernel configurations instead of autotuning. It targets AMD MI300X GPUs (ROCm/HIP platform).

## Development Environment

The project requires a Docker container with ROCm PyTorch and Triton installed:

```bash
# Start development container
docker compose up --build -d
docker attach tritonBLAS-dev

# Install package (run inside container)
pip3 install -e .
export PYTHONPATH=$(pwd)/include/:$PYTHONPATH
```

## Common Commands

### Build/Install
```bash
pip3 install -e .
export PYTHONPATH=$(pwd)/include/:$PYTHONPATH
```

The setup.py clones and builds a dependency from hipBLASLt (`origami` analytical model) automatically.

### Linting and Formatting
```bash
# Check code style
ruff check .

# Format code
ruff format .
```

Configuration is in `pyproject.toml` with line-length 120, double quotes, and specific ignored rules (E501, E701, E731, E741, F841, F401).

### Running Tests
```bash
cd tests

# Run all tests
pytest

# Run specific test file
pytest test_matmul.py

# Run specific test with verbose output
pytest test_matmul.py::test_matmul -v

# Run with specific parameters
pytest test_matmul_lt.py -v
```

### Running Examples
```bash
cd examples
python3 example_matmul.py
python3 example_matmul_lt.py
```

### Benchmarks
```bash
cd benchmarks
python3 tritonblas_matmul.py
python3 torch_matmul.py
```

## Architecture

### Core API Layer (`include/tritonblas/`)

The main exports are defined in `__init__.py`:
- `matmul()`: Drop-in replacement for `torch.matmul` (work-in-progress API)
- `matmul_lt()`: Peak performance API with precomputed selector (hipBLASLt-style)
- `matmul_a8w8()`, `matmul_a8w8_lt()`: A8W8 quantized matmul variants
- `matmul_fp4()`: FP4 quantized matmul
- `addmm()`: Matrix addition and multiplication
- `OrigamiMatmulSelector`: Configuration selector using analytical model

### Key Components

**`origami.py`**: Contains `OrigamiMatmulSelector` class that interfaces with the origami library (from hipBLASLt) to analytically determine optimal kernel configurations (block sizes, group sizes, number of SMs) based on matrix dimensions and hardware properties, avoiding the need for autotuning.

**`matmul.py`**: Implements the main matmul entry points:
- `persistent_matmul_lt()`: Uses persistent kernels with pre-allocated global buffers
- `streamk_matmul_lt()`: Uses Stream-K algorithm for better GPU utilization
- Both use `torch.library.triton_op` and `wrap_triton` for PyTorch integration

**`kernels/`**: Kernel implementations
- `persistent_gemm.py`: Persistent GEMM kernel for data-parallel execution
- `streamk_gemm.py`: Stream-K GEMM kernel for fixing wave quantization issues
- `fp4_matmul.py`: FP4 quantized matmul implementation
- `stages/`: Kernel stage abstractions including `gemm_context.py`, `tile.py`, `matrix_view.py`, `schedule.py`

**`utils.py`**: Shared utilities including `generate_matmul_inputs()` for test input generation with proper transpose handling.

### Design Patterns

1. **Peak Performance API Pattern**: Uses `OrigamiMatmulSelector` to precompute configurations, similar to hipBLASLt/cuBLASLt:
   ```python
   selector = tritonblas.OrigamiMatmulSelector(m, n, k, a_dtype, b_dtype, c_dtype, device)
   tritonblas.matmul_lt(A, B, C, selector=selector)
   ```

2. **Global Buffers**: The library pre-allocates global buffers (`_global_locks`, `_global_P`) for persistent kernels to avoid repeated allocations.

3. **Hardware-Aware Configuration**: Origami model queries hardware properties (CU count, memory bandwidth, instruction latency) to select optimal tile sizes (BLK_M, BLK_N, BLK_K) and launch parameters.

### Testing Structure

Tests use pytest with parameterized fixtures for:
- Matrix dimensions (m, n, k)
- Data types (float16, bfloat16, float32, float8)
- Transpose combinations (T/T, N/N, T/N, N/T)
- Stream-K enabled/disabled

Correctness is verified against `torch.matmul` with `torch.testing.assert_close()`.

### Dependencies

- `triton`: Core JIT compilation (installed from source in Docker)
- `origami`: Analytical model from hipBLASLt (cloned and built during `pip install`)
- `torch`: PyTorch with ROCm support
- `pandas`, `pytest`, `ruff`, `llnl-hatchet`: Development dependencies
