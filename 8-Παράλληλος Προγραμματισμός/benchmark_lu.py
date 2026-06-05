#!/usr/bin/env python3
"""Benchmark LU variants for multiple N and TILESIZE values and plot results."""

from __future__ import annotations

import os
import re
import subprocess
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


# ---- config -----------------------------------------------------------------
SOURCES = [
    ("serial", "0_lu_serial.c", []),
    ("serial-blocked", "1_lu_serial_cache_blocking.c", []),
    ("omp-for-blocked", "2_parallel_block_lu.c", ["-fopenmp"]),
    ("omp-tasks-blocked", "3_parallel_block_lu.c", ["-fopenmp"]),
]
N_VALUES = [512, 1024, 2048, 2496]
TILE_SIZES = [16, 32]
RUNS = 1
CC = "gcc"
CFLAGS = ["-Wall", "-O2"]
# -----------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
TIME_RE = re.compile(r"Computation time\s*=\s*([\d.eE+\-]+)\s*sec")


def compile_binary(label: str, src: str, extra: List[str], n: int, tilesize: int) -> str:
    safe = label.replace(" ", "_").replace("/", "_")
    binary = os.path.join(HERE, f"_bench_{safe}_N{n}_T{tilesize}")
    src_path = os.path.join(HERE, src)
    cmd = [CC] + CFLAGS + extra + [src_path, "-o", binary, f"-DN={n}", f"-DTILESIZE={tilesize}"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Compile failed ({label}, N={n}, T={tilesize}):\n{result.stderr}")
    return binary


def run_binary(binary: str) -> float:
    result = subprocess.run([binary], capture_output=True, text=True)
    match = TIME_RE.search(result.stdout)
    if not match:
        raise RuntimeError(
            f"Run failed ({binary}):\n{result.stderr}\n{result.stdout}"
        )
    return float(match.group(1))


def benchmark() -> Dict[int, Dict[str, List[float]]]:
    times: Dict[int, Dict[str, List[float]]] = {
        tile: {label: [] for label, _, _ in SOURCES} for tile in TILE_SIZES
    }

    for tile in TILE_SIZES:
        print(f"\n=== TILESIZE = {tile} ===")
        for n in N_VALUES:
            print(f"\n-- N = {n} --")
            for label, src, extra in SOURCES:
                binary = compile_binary(label, src, extra, n, tile)
                runs = [run_binary(binary) for _ in range(RUNS)]
                best = min(runs)
                times[tile][label].append(best)
                runs_fmt = ", ".join(f"{r:.4f}" for r in runs)
                print(f"  {label:18s}  best={best:.4f}s  runs=[{runs_fmt}]")
                os.remove(binary)

    return times


def plot(times: Dict[int, Dict[str, List[float]]]) -> None:
    labels = [label for label, _, _ in SOURCES]
    colors = ["steelblue", "tomato", "mediumseagreen", "goldenrod"]
    x = np.arange(len(N_VALUES))
    width = 0.18

    fig, axes = plt.subplots(1, len(TILE_SIZES), figsize=(14, 5), sharey=True)
    if len(TILE_SIZES) == 1:
        axes = [axes]

    for ax, tile in zip(axes, TILE_SIZES):
        for i, (label, color) in enumerate(zip(labels, colors)):
            offset = (i - (len(labels) - 1) / 2) * width
            vals = times[tile][label]
            ax.bar(x + offset, vals, width, label=label, color=color, edgecolor="black", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in N_VALUES])
        ax.set_xlabel("N")
        ax.set_title(f"TILESIZE = {tile}")
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    axes[0].set_ylabel("Execution time (s)")
    axes[0].legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    out = os.path.join(HERE, "lu_benchmark.png")
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved -> {out}")
    plt.show()


if __name__ == "__main__":
    results = benchmark()
    plot(results)
