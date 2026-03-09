# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

tritonBLAS is a lightweight Triton-based GEMM (General Matrix Multiplication) library that uses an analytical model (Origami) to predict optimal kernel configurations instead of autotuning.

**Current Status**: AMD MI300X (CDNA3/ROCm) support is implemented.
**Porting Target**: NVIDIA Blackwell (tcgen05 Tensor Core) - IN PROGRESS.

## NVIDIA Blackwell (tcgen05) Porting Plan

### Hardware Differences Summary

| Feature | AMD MI300X (CDNA3) | NVIDIA Blackwell (tcgen05) |
|---------|-------------------|---------------------------|
| **Tensor Core** | MFMA (Matrix Fused Multiply-Add) | tcgen05.mma_async |
| **Thread Organization** | Wave64 (64 threads) | Warp32 (32 threads) |
| **Compute Units** | CUs with 4 SIMD units per CU | SMs with Tensor Cores |
| **Cache Hierarchy** | L1 → L2 → HBM | Shared Mem → L1 → L2 → HBM |
| **Async Operations** | Limited | Extensive (TMA, async mma) |
| **Software Stack** | ROCm/HIP | CUDA 12.8+ |

### Key Porting Tasks

#### 1. Hardware Model Updates (`include/tritonblas/origami.py`)
- Replace `origami.get_hardware_for_device()` with Blackwell hardware parameters
- Update `N_CU` → `num_SMs` ( Streaming Multiprocessors)
- Add tcgen05-specific matrix instruction latencies
- Update memory bandwidth and cache size constants

```python
# Example hardware parameters for Blackwell
blackwell_hw_params = {
    "num_sms": 128,  # GB200 GPU
    "sm_clock_ghz": 1.8,
    "tensor_core_throughput": "2x FP16/BF16 per SM per clock",
    "shared_mem_per_sm": 256 * 1024,  # 256 KB
    "l2_cache_size": 96 * 1024 * 1024,  # 96 MB
    "hbm_bandwidth_gbps": 8000,  # 8 TB/s
}
```

#### 2. Kernel Updates (`include/tritonblas/kernels/`)

**Persistent GEMM** (`persistent_gemm.py`):
- Replace `tl.dot()` with tcgen05-specific mma instructions (via Triton intrinsics)
- Update block sizes for Warp32 execution (128x128 or 256x256 → 128x64/64x128)
- Add TMA (Tensor Memory Accelerator) support for async loads
- Adjust `num_warps` from 8 (for Wave64) to 4/8 (for Warp32)

**Stream-K GEMM** (`streamk_gemm.py`):
- Same changes as persistent GEMM
- Update atomics for inter-SM synchronization (Blackwell has different memory ordering)

**FP4 Support** (`fp4_matmul.py`):
- tcgen05 has native FP4 support - leverage `tcgen05.mma` with FP4 input types
- Update scaling factors for Blackwell's FP4 format

#### 3. Build System Updates

**`setup.py`**:
- Remove hipBLASLt dependency (AMD-specific)
- Add cuBLASLt or cutlass dependency for reference implementations
- Update to CUDA 12.8+ toolchain

**`Dockerfile`**:
- Replace `rocm/pytorch:latest-release` with `nvidia/cuda:12.8-devel-ubuntu22.04`
- Install PyTorch with CUDA support
- Install Triton with NVIDIA backend

#### 4. Triton Intrinsic Mapping

| AMD (ROCm) | NVIDIA (CUDA) | Notes |
|------------|---------------|-------|
| `tl.dot()` | `tl.dot()` + `allow_tf32` | May need `trans_a/trans_b` hints |
| `maxntid` | `num_threads` | Already abstracted in Triton |
| `waveid` | `warp_id` | Triton provides abstraction |
| MFMA-specific | `nvidia_tc_gen5_mma` | Use Triton's target-specific intrinsics |

#### 5. Testing Updates

**New test requirements**:
- Blackwell GPU (GB200 or B100/B200)
- CUDA 12.8+ driver
- Triton with NV backend enabled

**Verification**:
- Performance parity with cuBLASLt
- Correctness against PyTorch CUDA matmul

### Development Environment (NVIDIA)

```bash
# Docker for Blackwell development
docker run --gpus all -it nvidia/cuda:12.8-devel-ubuntu22.04

# Install PyTorch with CUDA 12.8
pip install torch --index-url https://download.pytorch.org/whl/cu128

# Install Triton (latest with Blackwell support)
pip install triton

# Install package
pip install -e .
```

### Performance Targets (Blackwell)

| Data Type | Target Utilization |
|-----------|-------------------|
| FP16/BF16 | >90% of theoretical tcgen05 peak |
| FP8 | >85% of theoretical peak |
| FP4 | >80% of theoretical peak (new in Blackwell) |

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

**AMD (Current)**:
- `triton`: Core JIT compilation (installed from source in Docker)
- `origami`: Analytical model from hipBLASLt (cloned and built during `pip install`)
- `torch`: PyTorch with ROCm support
- `pandas`, `pytest`, `ruff`, `llnl-hatchet`: Development dependencies

**NVIDIA (Porting Target)**:
- `triton`: Core JIT compilation with NV backend
- `torch`: PyTorch with CUDA 12.8+ support
- Reference: cuBLASLt or CUTLASS for validation
- Same Python dev dependencies

## TCGen05 Analytical Model

A new analytical performance model (`tcgen05_model.py`) has been implemented for NVIDIA Blackwell GPUs. This model predicts kernel performance without running autotune, similar to the Origami model for AMD.

### Key Components

**`TCGen05PerformanceModel`**: Core analytical model that estimates:
- tcgen05.mma_async instruction throughput and latency
- TMA (Tensor Memory Accelerator) async load performance
- Memory bandwidth limitations
- Wave quantization effects
- Pipeline stall conditions

**`TCGen05MatmulSelector`**: High-level interface for selecting optimal configurations:
```python
from tritonblas import TCGen05MatmulSelector

selector = TCGen05MatmulSelector(m, n, k, a_dtype, b_dtype, c_dtype, "GB200")
print(selector.predicted_tflops)  # Predicted performance
print(selector.get_performance_report())  # Detailed analysis
```

### Model Architecture

The model considers:

1. **Compute Cycles**: Based on tcgen05 instruction count and issue rate
   - tcgen05 shape: 128x8x64 for FP16, scaled for FP8/FP4
   - Latency hiding through software pipelining (2-4 stages)

2. **Memory Cycles**: TMA async loads overlap with compute
   - Only 10% of memory traffic on critical path (estimated)
   - Accounts for HBM bandwidth

3. **Wave Quantization**: Penalty for partial waves
   - Analyzes total tiles vs SM count
   - Recommends Stream-K when last wave utilization < 50%

### Usage Example

```python
from tritonblas import TCGen05MatmulSelector, predict_gemm_performance
import torch

# Predict performance without running kernel
selector = TCGen05MatmulSelector(
    8192, 8192, 8192,
    torch.float16, torch.float16, torch.float16,
    gpu_name="GB200"
)

print(f"Optimal config: [{selector.block_m}, {selector.block_n}, {selector.block_k}]")
print(f"Predicted: {selector.predicted_tflops:.1f} TFLOPS ({selector.predicted_utilization:.1f}%)")
print(f"Bottleneck: {selector.bottleneck}")
```

See `examples/example_tcgen05_model.py` for comprehensive usage examples.
