"""
NVIDIA Blackwell (tcgen05) Analytical Performance Model for GEMM

This module provides an Origami-style analytical model to predict GEMM kernel
performance on NVIDIA Blackwell GPUs (GB200, B100, B200) without autotuning.

Key Features:
- Models tcgen05.mma_async instruction throughput and latency
- Accounts for Tensor Memory Accelerator (TMA) async loads
- Supports FP16, BF16, FP8, and FP4 data types
- Predicts optimal block sizes, warps, and grid configuration
"""

import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import torch


@dataclass
class BlackwellHardwareSpecs:
    """Hardware specifications for NVIDIA Blackwell GPUs."""
    # GPU identification
    name: str
    sm_count: int  # Number of SMs (Streaming Multiprocessors)
    sm_clock_ghz: float  # SM clock frequency in GHz
    memory_clock_ghz: float  # HBM memory clock in GHz
    memory_bus_width: int  # Memory bus width in bits

    # Memory hierarchy
    l2_cache_size_mb: float  # L2 cache size in MB
    shared_mem_per_sm_kb: int  # Shared memory per SM in KB

    # tcgen05 Tensor Core specs
    tcgen05_mma_shape: Tuple[int, int, int] = (128, 8, 64)  # M, N, K for FP16
    tcgen05_issue_rate_per_sm: int = 1  # Instructions per cycle per SM
    tcgen05_latency_cycles: int = 32  # Pipeline latency

    # TMA (Tensor Memory Accelerator)
    tma_throughput_bytes_per_cycle: int = 128  # TMA load throughput
    tma_latency_cycles: int = 8

    def memory_bandwidth_gbps(self) -> float:
        """Calculate theoretical memory bandwidth in GB/s."""
        return (self.memory_clock_ghz * 1e9 * self.memory_bus_width * 2) / (8 * 1e9)

    def tcgen05_peak_tflops(self, dtype_bits: int) -> float:
        """Calculate peak Tensor Core throughput in TFLOPS."""
        # Each tcgen05.mma does M*N*K*2 FMA operations
        m, n, k = self.tcgen05_mma_shape
        ops_per_instruction = m * n * k * 2  # multiply-add = 2 ops

        # Total ops per cycle across all SMs
        ops_per_cycle = ops_per_instruction * self.tcgen05_issue_rate_per_sm * self.sm_count

        # Adjust for data type (e.g., FP8 does 2x ops of FP16)
        if dtype_bits == 8:
            ops_per_cycle *= 2
        elif dtype_bits == 4:
            ops_per_cycle *= 4

        return (ops_per_cycle * self.sm_clock_ghz) / 1e12


# Blackwell GPU specifications
BLACKWELL_GPUS = {
    "GB200": BlackwellHardwareSpecs(
        name="GB200",
        sm_count=128,
        sm_clock_ghz=1.8,
        memory_clock_ghz=2.1,
        memory_bus_width=8192,  # 8 HBM3e stacks
        l2_cache_size_mb=96.0,
        shared_mem_per_sm_kb=256,
        tcgen05_mma_shape=(128, 8, 64),
        tcgen05_issue_rate_per_sm=1,
        tcgen05_latency_cycles=32,
        tma_throughput_bytes_per_cycle=128,
        tma_latency_cycles=8,
    ),
    "B200": BlackwellHardwareSpecs(
        name="B200",
        sm_count=120,  # Slightly fewer SMs
        sm_clock_ghz=1.85,
        memory_clock_ghz=2.0,
        memory_bus_width=4096,
        l2_cache_size_mb=72.0,
        shared_mem_per_sm_kb=256,
        tcgen05_mma_shape=(128, 8, 64),
        tcgen05_issue_rate_per_sm=1,
        tcgen05_latency_cycles=32,
        tma_throughput_bytes_per_cycle=128,
        tma_latency_cycles=8,
    ),
    "B100": BlackwellHardwareSpecs(
        name="B100",
        sm_count=96,
        sm_clock_ghz=1.7,
        memory_clock_ghz=1.8,
        memory_bus_width=4096,
        l2_cache_size_mb=54.0,
        shared_mem_per_sm_kb=256,
        tcgen05_mma_shape=(128, 8, 64),
        tcgen05_issue_rate_per_sm=1,
        tcgen05_latency_cycles=32,
        tma_throughput_bytes_per_cycle=128,
        tma_latency_cycles=8,
    ),
}


@dataclass
class GEMMProblem:
    """Description of a GEMM problem."""
    m: int  # M dimension
    n: int  # N dimension
    k: int  # K dimension
    a_dtype: torch.dtype  # A matrix data type
    b_dtype: torch.dtype  # B matrix data type
    c_dtype: torch.dtype  # C matrix data type
    trans_a: bool = False  # Whether A is transposed
    trans_b: bool = False  # Whether B is transposed

    def total_flops(self) -> int:
        """Total floating point operations."""
        return 2 * self.m * self.n * self.k  # multiply-add = 2 ops

    def bytes_accessed(self) -> int:
        """Total bytes accessed from HBM."""
        a_size = self.m * self.k * self.a_dtype.itemsize
        b_size = self.k * self.n * self.b_dtype.itemsize
        c_size = self.m * self.n * self.c_dtype.itemsize
        return a_size + b_size + c_size  # C is read+write, counted once for estimate


@dataclass
class KernelConfig:
    """Kernel configuration parameters."""
    block_m: int  # Tile size in M dimension
    block_n: int  # Tile size in N dimension
    block_k: int  # Tile size in K dimension (per iteration)
    num_warps: int  # Warps per block (1 warp = 32 threads on Blackwell)
    num_stages: int  # Software pipeline stages
    use_tma: bool = True  # Use Tensor Memory Accelerator
    use_warp_specialization: bool = False  # Use warp specialization

    def threads_per_block(self) -> int:
        return self.num_warps * 32  # 32 threads per warp on Blackwell

    def shared_memory_bytes(self, dtype_size: int = 2) -> int:
        """Estimated shared memory usage per block."""
        # A buffer: block_m * block_k * dtype_size
        # B buffer: block_k * block_n * dtype_size
        # Double buffer if num_stages == 2
        a_buffer = self.block_m * self.block_k * dtype_size * self.num_stages
        b_buffer = self.block_k * self.block_n * dtype_size * self.num_stages
        # Add accumulator buffer (FP32)
        c_buffer = self.block_m * self.block_n * 4
        return a_buffer + b_buffer + c_buffer

    def total_tiles(self, m: int, n: int) -> int:
        """Total number of output tiles."""
        return math.ceil(m / self.block_m) * math.ceil(n / self.block_n)


@dataclass
class PerformanceEstimate:
    """Performance estimation results."""
    config: KernelConfig

    # Timing estimates (in microseconds)
    compute_time_us: float
    memory_time_us: float
    total_time_us: float

    # Throughput metrics
    achieved_tflops: float
    utilization_percent: float

    # Bottleneck analysis
    bottleneck: str  # "compute", "memory", or "latency"

    # Wave analysis
    total_waves: float
    last_wave_utilization: float


class TCGen05PerformanceModel:
    """
    Analytical performance model for NVIDIA Blackwell tcgen05 Tensor Core.

    Models the execution time of GEMM kernels based on:
    1. tcgen05.mma_async instruction throughput and latency
    2. TMA (Tensor Memory Accelerator) async load performance
    3. Memory bandwidth for non-TMA accesses
    4. Wave quantization effects
    5. Pipeline stall conditions
    """

    def __init__(self, hardware: BlackwellHardwareSpecs):
        self.hw = hardware

        # Valid block sizes for tcgen05 (constrained by instruction shape)
        # tcgen05 requires M to be multiple of 128 for FP16
        self.valid_block_m = [128, 256]
        self.valid_block_n = [64, 128, 256, 512]
        self.valid_block_k = [64, 128, 256]  # Must be multiple of 64 for tcgen05

        # Warp constraints (32 threads per warp on Blackwell)
        self.warps_per_sm_max = 64
        self.threads_per_warp = 32

    def _get_tcgen05_shape(self, dtype: torch.dtype) -> Tuple[int, int, int]:
        """Get tcgen05 MMA shape for given data type."""
        m, n, k = self.hw.tcgen05_mma_shape

        if dtype == torch.float8_e4m3fn or dtype == torch.float8_e5m2:
            # FP8: 2x throughput, same shape but K can be larger
            return (m, n, k * 2)
        elif str(dtype) == "torch.float4":  # FP4 not directly in torch yet
            # FP4: 4x throughput
            return (m, n, k * 4)
        else:
            return (m, n, k)

    def _estimate_tcgen05_instructions(self, config: KernelConfig, problem: GEMMProblem) -> int:
        """Estimate number of tcgen05.mma instructions per tile."""
        mi_m, mi_n, mi_k = self._get_tcgen05_shape(problem.a_dtype)

        # Number of instructions in M, N, K dimensions
        instrs_m = config.block_m // mi_m
        instrs_n = config.block_n // mi_n
        k_iters = math.ceil(problem.k / config.block_k)
        instrs_k = config.block_k // mi_k

        total_instrs_per_tile = instrs_m * instrs_n * k_iters * instrs_k
        return total_instrs_per_tile

    def _estimate_compute_cycles(self, config: KernelConfig, problem: GEMMProblem,
                                  num_active_sms: int) -> float:
        """
        Estimate compute cycles considering:
        1. tcgen05 instruction issue rate
        2. Pipeline depth (latency hiding)
        3. Wave quantization
        """
        total_tiles = config.total_tiles(problem.m, problem.n)
        instrs_per_tile = self._estimate_tcgen05_instructions(config, problem)
        total_instrs = total_tiles * instrs_per_tile

        # Instruction issue is the throughput bottleneck
        # Each SM can issue 1 tcgen05.mma per cycle
        cycles_issue = total_instrs / (self.hw.tcgen05_issue_rate_per_sm * num_active_sms)

        # Latency hiding: with sufficient pipeline depth, latency is hidden
        # Pipeline depth = num_stages * (latency of one stage)
        pipeline_depth = config.num_stages * self.hw.tcgen05_latency_cycles

        # If we don't have enough parallelism to hide latency
        instrs_per_sm = total_instrs / num_active_sms
        if instrs_per_sm < pipeline_depth:
            # Latency bound: each instruction waits for previous to complete
            cycles_latency = instrs_per_sm * self.hw.tcgen05_latency_cycles / num_active_sms
            return max(cycles_issue, cycles_latency)

        return cycles_issue

    def _estimate_memory_cycles(self, config: KernelConfig, problem: GEMMProblem,
                                 num_active_sms: int) -> float:
        """
        Estimate memory access cycles considering:
        1. TMA async loads (overlapped with compute)
        2. Memory bandwidth for initial loads
        """
        # Bytes needed per tile
        bytes_a = config.block_m * problem.k * problem.a_dtype.itemsize
        bytes_b = config.block_n * problem.k * problem.b_dtype.itemsize
        bytes_c = config.block_m * config.block_n * problem.c_dtype.itemsize

        total_tiles = config.total_tiles(problem.m, problem.n)
        total_bytes = total_tiles * (bytes_a + bytes_b + bytes_c)

        # TMA can overlap most memory access with compute
        # Only the first stage and last store are on critical path
        if config.use_tma:
            # Estimate exposed memory: initial loads + final stores
            bytes_exposed = total_tiles * (
                config.block_m * config.block_k * problem.a_dtype.itemsize * config.num_stages +
                config.block_n * config.block_k * problem.b_dtype.itemsize * config.num_stages +
                bytes_c  # Final write-back
            )
            # TMA is very efficient, most memory is overlapped
            bytes_exposed *= 0.1  # Only 10% on critical path
        else:
            bytes_exposed = total_bytes

        # Cycles at memory bandwidth
        bandwidth_bytes_per_cycle = (self.hw.memory_bandwidth_gbps() * 1e9) / (self.hw.sm_clock_ghz * 1e9)
        cycles_memory = bytes_exposed / (bandwidth_bytes_per_cycle * num_active_sms)

        return cycles_memory

    def _analyze_wave_quantization(self, config: KernelConfig, problem: GEMMProblem) -> Tuple[float, float]:
        """
        Analyze wave quantization effects.
        Returns (total_waves, last_wave_utilization)
        """
        total_tiles = config.total_tiles(problem.m, problem.n)
        max_active_blocks = min(
            self.hw.sm_count * self.warps_per_sm_max // config.num_warps,
            total_tiles
        )

        # How many waves needed
        total_waves = total_tiles / self.hw.sm_count

        # Last wave utilization
        full_waves = int(total_tiles // self.hw.sm_count)
        tiles_in_last_wave = total_tiles - (full_waves * self.hw.sm_count)

        if tiles_in_last_wave == 0:
            last_wave_util = 1.0
        else:
            last_wave_util = tiles_in_last_wave / self.hw.sm_count

        return total_waves, last_wave_util

    def estimate_config(self, config: KernelConfig, problem: GEMMProblem) -> PerformanceEstimate:
        """Estimate performance for a specific kernel configuration."""

        # Determine active SM count
        total_tiles = config.total_tiles(problem.m, problem.n)
        num_active_sms = min(self.hw.sm_count, total_tiles)

        # Compute and memory cycles
        compute_cycles = self._estimate_compute_cycles(config, problem, num_active_sms)
        memory_cycles = self._estimate_memory_cycles(config, problem, num_active_sms)

        # Total cycles (max of compute and memory due to async overlap)
        total_cycles = max(compute_cycles, memory_cycles * 0.8)  # 80% overlap assumed

        # Convert to time
        total_time_us = (total_cycles / (self.hw.sm_clock_ghz * 1e3))  # GHz to cycles/us

        # Wave quantization penalty
        total_waves, last_wave_util = self._analyze_wave_quantization(config, problem)
        if last_wave_util < 0.5:
            # Significant wave quantization penalty
            total_time_us /= (0.5 + 0.5 * last_wave_util)

        # Calculate achieved performance
        total_flops = problem.total_flops()
        achieved_tflops = (total_flops / total_time_us) / 1e6  # FLOPs/us to TFLOPS

        # Peak for this data type
        peak_tflops = self.hw.tcgen05_peak_tflops(problem.a_dtype.itemsize * 8)
        utilization = (achieved_tflops / peak_tflops) * 100

        # Determine bottleneck
        if compute_cycles > memory_cycles:
            bottleneck = "compute"
        elif memory_cycles > compute_cycles * 1.2:
            bottleneck = "memory"
        else:
            bottleneck = "balanced"

        return PerformanceEstimate(
            config=config,
            compute_time_us=(compute_cycles / (self.hw.sm_clock_ghz * 1e3)),
            memory_time_us=(memory_cycles / (self.hw.sm_clock_ghz * 1e3)),
            total_time_us=total_time_us,
            achieved_tflops=achieved_tflops,
            utilization_percent=utilization,
            bottleneck=bottleneck,
            total_waves=total_waves,
            last_wave_utilization=last_wave_util,
        )

    def select_optimal_config(self, problem: GEMMProblem,
                               constraints: Optional[Dict] = None) -> PerformanceEstimate:
        """
        Select optimal kernel configuration for given problem.

        Searches through valid configurations and returns the best performing one
        according to the analytical model.
        """
        constraints = constraints or {}
        max_shared_mem = constraints.get('max_shared_mem_kb', self.hw.shared_mem_per_sm_kb)

        best_estimate = None
        best_score = float('inf')

        for block_m, block_n, block_k, num_warps, num_stages in itertools.product(
            self.valid_block_m,
            self.valid_block_n,
            self.valid_block_k,
            [2, 4, 8],  # num_warps options
            [2, 3, 4],  # num_stages for pipeline
        ):
            config = KernelConfig(
                block_m=block_m,
                block_n=block_n,
                block_k=block_k,
                num_warps=num_warps,
                num_stages=num_stages,
                use_tma=True,
            )

            # Skip if shared memory exceeds limit
            if config.shared_memory_bytes() > max_shared_mem * 1024:
                continue

            # Skip if not enough warps to saturate TC
            if num_warps < 2:
                continue

            # Skip if block size doesn't divide problem well
            if problem.m < block_m and problem.m % block_m != 0:
                continue
            if problem.n < block_n and problem.n % block_n != 0:
                continue

            estimate = self.estimate_config(config, problem)

            # Score: lower is better, with penalty for low utilization
            score = estimate.total_time_us * (1 + (100 - estimate.utilization_percent) / 100)

            if score < best_score:
                best_score = score
                best_estimate = estimate

        return best_estimate


class TCGen05MatmulSelector:
    """
    High-level selector interface similar to OrigamiMatmulSelector.

    Usage:
        selector = TCGen05MatmulSelector(m, n, k, a_dtype, b_dtype, c_dtype, "GB200")
        config = selector.get_config()  # Returns optimal KernelConfig
    """

    def __init__(self, m: int, n: int, k: int,
                 a_dtype: torch.dtype, b_dtype: torch.dtype, c_dtype: torch.dtype,
                 gpu_name: str = "GB200",
                 trans_a: bool = False, trans_b: bool = False):

        if gpu_name not in BLACKWELL_GPUS:
            raise ValueError(f"Unknown GPU: {gpu_name}. Available: {list(BLACKWELL_GPUS.keys())}")

        self.problem = GEMMProblem(
            m=m, n=n, k=k,
            a_dtype=a_dtype, b_dtype=b_dtype, c_dtype=c_dtype,
            trans_a=trans_a, trans_b=trans_b
        )

        self.hw = BLACKWELL_GPUS[gpu_name]
        self.model = TCGen05PerformanceModel(self.hw)

        # Run analysis
        self._estimate = self.model.select_optimal_config(self.problem)
        self._config = self._estimate.config

    @property
    def block_m(self) -> int:
        return self._config.block_m

    @property
    def block_n(self) -> int:
        return self._config.block_n

    @property
    def block_k(self) -> int:
        return self._config.block_k

    @property
    def num_warps(self) -> int:
        return self._config.num_warps

    @property
    def num_stages(self) -> int:
        return self._config.num_stages

    @property
    def predicted_tflops(self) -> float:
        return self._estimate.achieved_tflops

    @property
    def predicted_utilization(self) -> float:
        return self._estimate.utilization_percent

    @property
    def bottleneck(self) -> str:
        return self._estimate.bottleneck

    def get_config(self) -> KernelConfig:
        """Get the optimal kernel configuration."""
        return self._config

    def get_performance_report(self) -> str:
        """Get detailed performance prediction report."""
        p = self.problem
        e = self._estimate
        c = self._config

        report = f"""
=== TCGen05 Performance Prediction ===
Problem: {p.m}x{p.n}x{p.k}, dtypes={p.a_dtype},{p.b_dtype},{p.c_dtype}
GPU: {self.hw.name} ({self.hw.sm_count} SMs @ {self.hw.sm_clock_ghz}GHz)

Optimal Configuration:
  Block: [{c.block_m}, {c.block_n}, {c.block_k}]
  Warps: {c.num_warps}, Stages: {c.num_stages}
  Shared Memory: {c.shared_memory_bytes() // 1024} KB

Performance Estimate:
  Total Time: {e.total_time_us:.2f} us
  Throughput: {e.achieved_tflops:.1f} TFLOPS
  Utilization: {e.utilization_percent:.1f}%
  Bottleneck: {e.bottleneck}

Wave Analysis:
  Total Waves: {e.total_waves:.2f}
  Last Wave Util: {e.last_wave_utilization*100:.1f}%

Peak Theoretical: {self.hw.tcgen05_peak_tflops(p.a_dtype.itemsize * 8):.1f} TFLOPS
        """
        return report


def predict_gemm_performance(m: int, n: int, k: int,
                              a_dtype: torch.dtype = torch.float16,
                              b_dtype: torch.dtype = torch.float16,
                              c_dtype: torch.dtype = torch.float16,
                              gpu_name: str = "GB200") -> PerformanceEstimate:
    """
    Convenience function to predict GEMM performance.

    Returns detailed performance estimate for given problem on specified GPU.
    """
    problem = GEMMProblem(m, n, k, a_dtype, b_dtype, c_dtype)
    hw = BLACKWELL_GPUS[gpu_name]
    model = TCGen05PerformanceModel(hw)
    return model.select_optimal_config(problem)


# Example usage and validation
if __name__ == "__main__":
    # Test different problem sizes
    test_cases = [
        (8192, 8192, 8192, torch.float16),
        (4096, 4096, 4096, torch.bfloat16),
        (16384, 16384, 16384, torch.float8_e4m3fn),
        (2048, 2048, 2048, torch.float16),
    ]

    for m, n, k, dtype in test_cases:
        print(f"\n{'='*60}")
        selector = TCGen05MatmulSelector(m, n, k, dtype, dtype, dtype, "GB200")
        print(selector.get_performance_report())
