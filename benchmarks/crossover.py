"""The time of simulate_fit per case on one device over batch sizes, to find the batch size from which the GPU is faster than the CPU.

usage: python benchmarks/crossover.py cpu|cuda OUTPUT.json   (TORCHINDUCTOR_CACHE_DIR should be on disk; the first calls of each size are warm-up)

Rule, written before any run (2026-11-24): the crossover is the smallest batch size from which the median time per case of the GPU is at least 20 percent below the CPU's at that size and at every larger size of the grid, in each of three runs of each device (benchmarks/crossover.sh). The default of simulate_fit for device=None (GPU_MIN_CASES in src/wobbler/gaiafit.py) is set to the crossover; if the GPU is never 20 percent faster at a size of the grid, the default is the CPU for every size. A device that is slower than the other at its default size is a defect.
What is timed: one call of wobbler.simulate_fit with n_orient equal to the batch size for one parallax, on a random scan of 191 epochs at G 13, the DR3 window, with the compiled code, from the call to the end of the device's work (torch.cuda.synchronize()). The results stay on the device. Three warm-up calls of each size come first (the compilation of a new size is not timed); then NINE timed calls. The median and the extremes of the nine are recorded.
"""
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

import wobbler as wb

SIZES = (100, 200, 500, 700, 1000, 2000, 5000, 10000, 20000)
WARMUP, TIMED, EPOCHS = 3, 9, 191
ROOT = Path(__file__).resolve().parents[1]


def relative_to_the_repository(path):
    """The path as written in the record: relative to the repository when it is inside it."""
    if not path:
        return path
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return path


def main():
    device, output = sys.argv[1], Path(sys.argv[2])
    rng = np.random.default_rng(0)
    jd = np.sort(rng.uniform(*wb.WINDOWS["DR3"], EPOCHS))
    scan = wb.Scan(jd=jd, psi=rng.uniform(0, np.pi, EPOCHS), plx_factor=rng.uniform(-0.9, 0.9, EPOCHS), device=device)
    orbit = wb.Orbit(period_days=300.0, ecc=0.3, t_peri_days=-50.0, omega=1.0)

    def call(size, seed):
        fit = wb.simulate_fit(scan, orbit, a1_au=1.0, parallax_mas=np.array([2.0]), G=13.0, n_orient=size, seed=seed, device=device, nu_eff=1.47)
        if device != "cpu":
            torch.cuda.synchronize()
        return fit

    rows = []
    for size in SIZES:
        for i in range(WARMUP):
            call(size, i)
        times = []
        for i in range(TIMED):
            started = time.perf_counter()
            call(size, 100 + i)
            times.append(time.perf_counter() - started)
        row = {"batch_size": size, "median_seconds": float(np.median(times)), "min_seconds": float(min(times)), "max_seconds": float(max(times)), "microseconds_per_case_median": float(np.median(times) / size * 1e6)}
        rows.append(row)
        print(f"{size:6d} cases: median {row['median_seconds'] * 1e3:8.2f} ms ({row['microseconds_per_case_median']:8.2f} us per case), range {row['min_seconds'] * 1e3:.2f} to {row['max_seconds'] * 1e3:.2f} ms", flush=True)
    cpu = next((line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo") if line.startswith("model name")), platform.processor())
    name = torch.cuda.get_device_name(0) if device != "cpu" else f"{cpu} ({torch.get_num_threads()} threads)"
    output.write_text(json.dumps({"device": device, "hardware": name, "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "compile": wb.compile_enabled(),
                                  "TORCHINDUCTOR_CACHE_DIR": relative_to_the_repository(os.environ.get("TORCHINDUCTOR_CACHE_DIR")), "load_1min_at_the_end": os.getloadavg()[0], "rows": rows}, indent=1) + "\n")


if __name__ == "__main__":
    main()
