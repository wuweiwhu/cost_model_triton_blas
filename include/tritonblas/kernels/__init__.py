"""
Kernel implementations for tritonblas.

This package contains specific GEMM kernel implementations:
- persistent_gemm: Persistent (data-parallel) GEMM kernel using composable stages (AMD)
- persistent_gemm_monolithic: Monolithic persistent GEMM kernel (legacy, for debugging)
- streamk_gemm: Stream-K GEMM kernel for load balancing (AMD)
- blackwell_gemm: NVIDIA Blackwell tcgen05-optimized kernels
- stages: Composable kernel building blocks

Environment Variables:
- TBLAS_USE_MONOLITHIC: Set to '1' or 'true' to use the monolithic persistent kernel instead of the composable stages version
- TBLAS_BACKEND: Set to 'blackwell' to use NVIDIA Blackwell kernels (default: auto-detect)
"""

import os

# Backend selection
_backend = os.environ.get('TBLAS_BACKEND', 'auto').lower()

# Check environment variable to determine which persistent kernel to use
_use_monolithic = os.environ.get('TBLAS_USE_MONOLITHIC', '').lower() in ('1', 'true', 'yes')

if _backend == 'blackwell':
    # Use NVIDIA Blackwell kernels
    from .blackwell_gemm import blackwell_matmul as persistent_matmul
    from .blackwell_gemm import blackwell_matmul as streamk_matmul
    from .blackwell_gemm import BlackwellGEMMKernel
elif _use_monolithic:
    # Use monolithic version (legacy implementation)
    from .persistent_gemm_monolithic import persistent_matmul
    from .streamk_gemm import streamk_matmul
else:
    # Use composable stages version (default, AMD)
    from .persistent_gemm import persistent_matmul
    from .streamk_gemm import streamk_matmul

# FP4 kernel
from .fp4_matmul import fp4_matmul

# Export stages submodule
from . import stages

__all__ = ['persistent_matmul', 'streamk_matmul', 'fp4_matmul', 'stages']

# Conditionally export Blackwell components
try:
    from .blackwell_gemm import blackwell_matmul, BlackwellGEMMKernel
    __all__.extend(['blackwell_matmul', 'BlackwellGEMMKernel'])
except ImportError:
    pass
