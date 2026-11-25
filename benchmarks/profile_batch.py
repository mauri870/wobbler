"""Where the time of one call goes on the GPU at 1,000 cases: wall time against the time the kernels run, and the number of kernel launches.

usage: python benchmarks/profile_batch.py OUTPUT.json   (the GPU to itself; TORCHINDUCTOR_CACHE_DIR should be on disk)
Two operations of 1,000 cases of the 191-epoch random scan of benchmarks/crossover.py: wobbler.simulate_fit, and observe followed by fit_five_parameter on given noise (the operation of benchmarks/bench.py). Five warm-up calls, then five calls under torch.profiler; the wall time of a call is measured outside the profiler on 20 calls.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

import wobbler as wb

SIZE, EPOCHS = 1000, 191


def main():
    output = Path(sys.argv[1])
    rng = np.random.default_rng(0)
    jd = np.sort(rng.uniform(*wb.WINDOWS["DR3"], EPOCHS))
    scan = wb.Scan(jd=jd, psi=rng.uniform(0, np.pi, EPOCHS), plx_factor=rng.uniform(-0.9, 0.9, EPOCHS), device="cuda")
    orbit = wb.Orbit(period_days=300.0, ecc=0.3, t_peri_days=-50.0, omega=1.0)
    error = float(wb.epoch_error(13.0, device="cpu"))
    z_err, z_extra = rng.normal(size=(SIZE, EPOCHS)), rng.normal(size=(SIZE, EPOCHS))
    keep = rng.uniform(size=(SIZE, EPOCHS)) > 0.1
    extra = rng.uniform(0, 0.04, SIZE)
    t = scan.t_yr()
    parameters = dict(parallax=rng.uniform(0.5, 4, SIZE), pmra=rng.normal(0, 5, SIZE), pmdec=rng.normal(0, 5, SIZE), a0_mas=rng.uniform(0, 4, SIZE), inclination=np.arccos(rng.uniform(0, 1, SIZE)), node=rng.uniform(0, 2 * np.pi, SIZE),
                      period_days=rng.uniform(100, 900, SIZE), ecc=rng.uniform(0, 0.8, SIZE), t_peri_days=rng.uniform(-400, 400, SIZE), omega=rng.uniform(0, 2 * np.pi, SIZE))

    def simulate():
        wb.simulate_fit(scan, orbit, a1_au=1.0, parallax_mas=np.array([2.0]), G=13.0, n_orient=SIZE, seed=1, device="cuda", nu_eff=1.47)
        torch.cuda.synchronize()

    def observe_and_fit():
        observed = wb.observe(scan, error=error, z_err=z_err, extra=extra, z_extra=z_extra, device="cuda", **parameters)
        wb.fit_five_parameter(t, scan.psi, scan.plx_factor, observed, error, keep=keep, device="cuda")
        torch.cuda.synchronize()

    out = {"batch_size": SIZE, "device": torch.cuda.get_device_name(0), "torch": torch.__version__, "operations": {}}
    for name, function in (("simulate_fit", simulate), ("observe then fit_five_parameter", observe_and_fit)):
        for _ in range(5):
            function()
        started = time.perf_counter()
        for _ in range(20):
            function()
        wall = (time.perf_counter() - started) / 20
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(5):
                function()
        events = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
        kernel_seconds = sum(e.device_time for e in events) / 1e6 / 5
        names = {}
        for e in events:
            names[e.name] = names.get(e.name, 0.0) + e.device_time / 1e6 / 5
        top = sorted(names.items(), key=lambda kv: -kv[1])[:6]
        out["operations"][name] = {"wall_seconds_per_call": wall, "kernel_seconds_per_call": kernel_seconds, "kernel_launches_per_call": len(events) / 5, "kernel_share_of_wall": kernel_seconds / wall,
                                   "top_kernels_seconds_per_call": [[k, v] for k, v in top]}
        print(f"{name}: wall {wall * 1e3:.2f} ms per call, kernels {kernel_seconds * 1e3:.2f} ms ({kernel_seconds / wall * 100:.0f} percent of the wall), {len(events) / 5:.0f} kernel launches per call", flush=True)
        for k, v in top:
            print(f"    {v * 1e3:7.3f} ms  {k[:100]}")
    output.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
