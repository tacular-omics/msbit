"""
Benchmark msbit CPU and GPU backends on identical matrices and plot results.

Usage:
    python bench.py
"""

import itertools
import time

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from msbit import BitmapIndex
from msbit.encoding import random_bitmap_matrix

MAT_SIZES = [10_000, 50_000, 100_000, 250_000, 500_000]
ARRAY_SIZES = [10, 25, 50, 100, 200, 500]
TOP_K = 10
N_QUERIES = 10  # queries per timing trial — amortises GPU launch overhead
N_TRIALS = 3    # timing trials; best is kept

CUDA_AVAILABLE = False
try:
    from numba import cuda
    CUDA_AVAILABLE = cuda.is_available()
except Exception:
    pass


def _bench_backend(backend: str, bitmaps: np.ndarray, queries: np.ndarray) -> float:
    """Return best per-query time (seconds) over N_TRIALS trials."""
    index = BitmapIndex(bitmaps, backend=backend)
    index.search(queries[0], TOP_K)  # warmup this index

    times = []
    for _ in range(N_TRIALS):
        t0 = time.perf_counter()
        for q in queries:
            index.search(q, TOP_K)
        times.append((time.perf_counter() - t0) / N_QUERIES)
    return min(times)


def run_benchmarks() -> pl.DataFrame:
    backends = ["cpu"] + (["gpu"] if CUDA_AVAILABLE else [])
    if not CUDA_AVAILABLE:
        print("CUDA not available — skipping GPU benchmark", flush=True)

    print("Warming up JIT...", flush=True)
    _db = random_bitmap_matrix(10, 1_000)
    for backend in backends:
        BitmapIndex(_db, backend=backend).search(_db[0], TOP_K)

    results = []
    combos = list(itertools.product(ARRAY_SIZES, MAT_SIZES))
    for i, (array_size, mat_size) in enumerate(combos, 1):
        print(f"[{i}/{len(combos)}] array_size={array_size}  mat_size={mat_size}", flush=True)
        rng = np.random.default_rng(42)
        bitmaps = rng.integers(0, np.iinfo(np.uint64).max, size=(mat_size, array_size), dtype=np.uint64)
        queries = bitmaps[:N_QUERIES]
        matrix_mb = round(bitmaps.nbytes / 1024**2, 2)

        for backend in backends:
            best = _bench_backend(backend, bitmaps, queries)
            throughput = mat_size / best / 1_000_000
            print(f"  {backend:3s}  {best * 1000:.2f}ms/query  {throughput:.2f}M/s", flush=True)
            results.append({
                "backend": backend,
                "array_size": array_size,
                "mat_size": mat_size,
                "matrix_mb": matrix_mb,
                "time_per_query_s": best,
                "throughput_M_per_s": throughput,
            })

    return pl.DataFrame(results)


def _build_heatmap(df: pl.DataFrame, array_sizes: list, mat_sizes: list) -> np.ndarray:
    heatmap = np.zeros((len(array_sizes), len(mat_sizes)))
    for i, array_size in enumerate(array_sizes):
        for j, mat_size in enumerate(mat_sizes):
            val = df.filter(
                (pl.col("array_size") == array_size) & (pl.col("mat_size") == mat_size)
            )["throughput_M_per_s"]
            if len(val):
                heatmap[i, j] = val[0]
    return heatmap


def plot_results(df: pl.DataFrame) -> None:
    backends = sorted(df["backend"].unique().to_list())  # cpu before gpu
    array_sizes = sorted(df["array_size"].unique().to_list())
    mat_sizes = sorted(df["mat_size"].unique().to_list())

    colors = cm.viridis(np.linspace(0, 1, len(array_sizes)))
    style = {"cpu": "-", "gpu": "--"}
    marker = {"cpu": "o", "gpu": "s"}

    has_gpu = "gpu" in backends
    n_heatmaps = len(backends)  # one heatmap per backend + speedup if both present
    n_cols = 2 + n_heatmaps + (1 if has_gpu else 0)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))

    # --- throughput vs mat_size (overlaid) ---
    ax = axes[0]
    handles = []
    for color, array_size in zip(colors, array_sizes, strict=True):
        for backend in backends:
            sub = df.filter(
                (pl.col("backend") == backend) & (pl.col("array_size") == array_size)
            ).sort("mat_size")
            (line,) = ax.plot(
                sub["mat_size"].to_list(),
                sub["throughput_M_per_s"].to_list(),
                linestyle=style[backend], marker=marker[backend],
                color=color, alpha=0.85,
            )
            if array_size == array_sizes[0]:
                line.set_label(backend.upper())
                handles.append(line)
    # legend: backends by linestyle
    ax.legend(handles=handles, title="Backend", fontsize=9)
    # second legend for array sizes via proxy
    proxy = [
        plt.Line2D([0], [0], color=c, marker="o", linestyle="-", label=f"array={a}")
        for c, a in zip(colors, array_sizes, strict=True)
    ]
    ax.add_artist(ax.legend(handles=proxy, title="Array Size", fontsize=7, loc="upper right"))
    ax.legend(handles=handles, title="Backend", fontsize=9, loc="upper left")
    ax.set_xlabel("Matrix Size")
    ax.set_ylabel("Throughput (M rows/s)")
    ax.set_title("Throughput vs Matrix Size")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    # --- query time vs matrix_mb (overlaid) ---
    ax = axes[1]
    for color, array_size in zip(colors, array_sizes, strict=True):
        for backend in backends:
            sub = df.filter(
                (pl.col("backend") == backend) & (pl.col("array_size") == array_size)
            ).sort("matrix_mb")
            ax.plot(
                sub["matrix_mb"].to_list(),
                sub["time_per_query_s"].to_list(),
                linestyle=style[backend], marker=marker[backend],
                color=color, alpha=0.85,
                label=f"{backend} array={array_size}",
            )
    ax.set_xlabel("Matrix Size (MB)")
    ax.set_ylabel("Time per query (s)")
    ax.set_title("Query Time vs Matrix Memory")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    # --- per-backend throughput heatmaps ---
    for col_offset, backend in enumerate(backends):
        ax = axes[2 + col_offset]
        heatmap = _build_heatmap(df.filter(pl.col("backend") == backend), array_sizes, mat_sizes)
        im = ax.imshow(heatmap, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(mat_sizes)))
        ax.set_xticklabels([f"{m // 1000}k" for m in mat_sizes], rotation=45)
        ax.set_yticks(range(len(array_sizes)))
        ax.set_yticklabels(array_sizes)
        ax.set_xlabel("Matrix Size")
        ax.set_ylabel("Array Size")
        ax.set_title(f"{backend.upper()} Throughput (M rows/s)")
        plt.colorbar(im, ax=ax, label="M rows/s")
        for i in range(heatmap.shape[0]):
            for j in range(heatmap.shape[1]):
                ax.text(j, i, f"{heatmap[i, j]:.1f}", ha="center", va="center", color="w", fontsize=6)

    # --- GPU/CPU speedup heatmap ---
    if has_gpu:
        ax = axes[-1]
        cpu_hm = _build_heatmap(df.filter(pl.col("backend") == "cpu"), array_sizes, mat_sizes)
        gpu_hm = _build_heatmap(df.filter(pl.col("backend") == "gpu"), array_sizes, mat_sizes)
        with np.errstate(divide="ignore", invalid="ignore"):
            speedup = np.where(gpu_hm > 0, gpu_hm / cpu_hm, 0.0)
        im = ax.imshow(speedup, aspect="auto", cmap="RdYlGn", vmin=0)
        ax.set_xticks(range(len(mat_sizes)))
        ax.set_xticklabels([f"{m // 1000}k" for m in mat_sizes], rotation=45)
        ax.set_yticks(range(len(array_sizes)))
        ax.set_yticklabels(array_sizes)
        ax.set_xlabel("Matrix Size")
        ax.set_ylabel("Array Size")
        ax.set_title("GPU / CPU Speedup")
        plt.colorbar(im, ax=ax, label="× speedup")
        for i in range(speedup.shape[0]):
            for j in range(speedup.shape[1]):
                ax.text(j, i, f"{speedup[i, j]:.1f}×", ha="center", va="center", color="k", fontsize=6)

    plt.tight_layout()
    plt.savefig("benchmark_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved benchmark_results.png")


if __name__ == "__main__":
    df = run_benchmarks()
    print("\n", df)
    df.write_csv("benchmark_results.csv")
    plot_results(df)
