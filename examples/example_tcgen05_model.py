"""
Example: Using TCGen05 Analytical Performance Model

This example demonstrates how to use the analytical model to predict
GEMM kernel performance on NVIDIA Blackwell (tcgen05) without running
the actual kernel.

Usage:
    python example_tcgen05_model.py --m 8192 --n 8192 --k 8192 --dtype fp16
"""

import argparse
import sys

# Add parent directory to path for imports
sys.path.insert(0, "../include")

try:
    import torch
    from tritonblas import TCGen05MatmulSelector, BLACKWELL_GPUS, predict_gemm_performance
except ImportError as e:
    print(f"Import error: {e}")
    print("Note: This example requires the tcgen05_model module.")
    print("The analytical model works without a GPU - it predicts performance mathematically.")
    sys.exit(1)


def print_comparison_table():
    """Print performance comparison across different GPUs."""
    print("\n" + "=" * 80)
    print("Blackwell GPU Specifications Comparison")
    print("=" * 80)

    for name, hw in BLACKWELL_GPUS.items():
        print(f"\n{name}:")
        print(f"  SMs: {hw.sm_count}")
        print(f"  SM Clock: {hw.sm_clock_ghz} GHz")
        print(f"  Memory Bandwidth: {hw.memory_bandwidth_gbps():.0f} GB/s")
        print(f"  L2 Cache: {hw.l2_cache_size_mb} MB")
        print(f"  Shared Mem/SM: {hw.shared_mem_per_sm_kb} KB")

        # Peak performance for different dtypes
        for bits in [16, 8, 4]:
            peak = hw.tcgen05_peak_tflops(bits)
            print(f"  Peak FP{bits}: {peak:.1f} TFLOPS")


def analyze_problem_size(m, n, k, dtype_str="fp16", gpu_name="GB200"):
    """Analyze a specific problem size and show detailed breakdown."""

    # Map dtype string to torch dtype
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp8": torch.float8_e4m3fn,
    }

    if dtype_str not in dtype_map:
        print(f"Unsupported dtype: {dtype_str}")
        return

    dtype = dtype_map[dtype_str]

    print(f"\n{'=' * 80}")
    print(f"Problem Analysis: {m}x{n}x{k}, dtype={dtype_str}, GPU={gpu_name}")
    print("=" * 80)

    # Create selector (runs analytical model)
    selector = TCGen05MatmulSelector(m, n, k, dtype, dtype, dtype, gpu_name)

    # Print detailed report
    print(selector.get_performance_report())

    # Additional analysis
    print("\nConfiguration Rationale:")
    print(f"  Block [{selector.block_m}, {selector.block_n}, {selector.block_k}]:")
    print(f"    - M={selector.block_m}: tcgen05 requires M multiple of 128")
    print(f"    - N={selector.block_n}: Balances parallelism vs overhead")
    print(f"    - K={selector.block_k}: Hides latency with {selector.num_stages} stages")

    print(f"\n  Warps={selector.num_warps}:")
    warp_coverage = (selector.block_m * selector.block_n) / (128 * 8)  # tcgen05 shape
    print(f"    - Covers {warp_coverage:.1f} tcgen05 instructions per iteration")

    print(f"\n  Stages={selector.num_stages}:")
    latency_cycles = 32  # tcgen05 latency
    sm_clock_ghz = 1.8
    latency_us = latency_cycles / (sm_clock_ghz * 1000)
    mem_latency_us = 0.5  # estimated
    print(f"    - tcgen05 latency: ~{latency_us:.2f} us")
    print(f"    - Memory latency: ~{mem_latency_us:.2f} us")
    print(f"    - Pipeline depth hides {selector.num_stages * latency_us:.2f} us")

    return selector


def sweep_problem_sizes(gpu_name="GB200"):
    """Sweep through different problem sizes and show predictions."""

    print("\n" + "=" * 80)
    print(f"Problem Size Sweep on {gpu_name}")
    print("=" * 80)
    print(f"{'M':>8} {'N':>8} {'K':>8} | {'Block':>20} | {'TFLOPS':>10} | {'Util%':>6}")
    print("-" * 80)

    test_cases = [
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
        (16384, 16384, 16384),
        (1024, 8192, 4096),   # Tall-skinny
        (8192, 1024, 4096),   # Short-wide
        (4096, 4096, 1024),   # Small K
    ]

    for m, n, k in test_cases:
        selector = TCGen05MatmulSelector(
            m, n, k,
            torch.float16, torch.float16, torch.float16,
            gpu_name
        )

        block_str = f"[{selector.block_m},{selector.block_n},{selector.block_k}]"
        print(f"{m:>8} {n:>8} {k:>8} | {block_str:>20} | "
              f"{selector.predicted_tflops:>10.1f} | {selector.predicted_utilization:>6.1f}")


def compare_streamk_vs_persistent(m, n, k, gpu_name="GB200"):
    """Compare Stream-K vs Persistent (data-parallel) approaches."""

    print("\n" + "=" * 80)
    print(f"Stream-K vs Persistent Comparison: {m}x{n}x{k}")
    print("=" * 80)

    from tritonblas.tcgen05_model import TCGen05PerformanceModel, GEMMProblem, KernelConfig

    hw = BLACKWELL_GPUS[gpu_name]
    model = TCGen05PerformanceModel(hw)
    problem = GEMMProblem(m, n, k, torch.float16, torch.float16, torch.float16)

    # Get optimal persistent config
    persistent_estimate = model.select_optimal_config(problem)
    p = persistent_estimate

    print("\nPersistent (Data-Parallel):")
    print(f"  Grid: {p.config.total_tiles(m, n)} tiles")
    print(f"  Waves: {p.total_waves:.2f}")
    print(f"  Last Wave Util: {p.last_wave_utilization*100:.1f}%")
    print(f"  Predicted Time: {p.total_time_us:.2f} us")
    print(f"  Predicted TFLOPS: {p.achieved_tflops:.1f}")

    # Analyze Stream-K
    # Stream-K uses different grid calculation
    total_iters = p.config.total_tiles(m, n) * (k // p.config.block_k)
    sk_grid = min(hw.sm_count, total_iters // 4)  # Aim for at least 4 iters per SM

    print("\nStream-K:")
    print(f"  Grid: {sk_grid} SMs")
    print(f"  Iters per SM: {total_iters / sk_grid:.1f}")
    print(f"  Note: Stream-K eliminates wave quantization")

    # Stream-K has overhead from partial tile reduction
    sk_overhead = 1.05  # 5% overhead estimate
    sk_time = p.total_time_us * sk_overhead / max(1, p.last_wave_utilization)

    print(f"  Estimated Time: {sk_time:.2f} us (with overhead)")

    if p.last_wave_utilization < 0.7:
        print(f"\n  => Stream-K recommended (low wave utilization: {p.last_wave_utilization*100:.1f}%)")
    else:
        print(f"\n  => Persistent sufficient (good wave utilization: {p.last_wave_utilization*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="TCGen05 Analytical Performance Model Demo"
    )
    parser.add_argument("--m", type=int, default=8192, help="M dimension")
    parser.add_argument("--n", type=int, default=8192, help="N dimension")
    parser.add_argument("--k", type=int, default=8192, help="K dimension")
    parser.add_argument("--dtype", type=str, default="fp16",
                        choices=["fp32", "fp16", "bf16", "fp8"],
                        help="Data type")
    parser.add_argument("--gpu", type=str, default="GB200",
                        choices=list(BLACKWELL_GPUS.keys()),
                        help="GPU model")
    parser.add_argument("--compare-gpus", action="store_true",
                        help="Show GPU comparison table")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep through problem sizes")
    parser.add_argument("--compare-streamk", action="store_true",
                        help="Compare Stream-K vs Persistent")

    args = parser.parse_args()

    if args.compare_gpus:
        print_comparison_table()

    if args.sweep:
        sweep_problem_sizes(args.gpu)
    elif args.compare_streamk:
        compare_streamk_vs_persistent(args.m, args.n, args.k, args.gpu)
    else:
        analyze_problem_size(args.m, args.n, args.k, args.dtype, args.gpu)

        # Also show Stream-K comparison if requested
        if args.m > 0:  # Always show in default mode
            compare_streamk_vs_persistent(args.m, args.n, args.k, args.gpu)


if __name__ == "__main__":
    main()
