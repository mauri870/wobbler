"""Benchmark of wobbler against gaiamock on generated inputs: 14,400 cases of a 191-epoch scan with random orbits and noise, RUWE, excess noise and the parallax of the DR3 window.
wobbler: observe and the batched fit in one call, in batches of 1,000 and in one batch, on the device named on the command line (cpu, or cuda where torch sees a GPU), the first call (which compiles) apart from the later ones.
gaiamock: fit_5par_solution_only and an excess-noise bisection, one case at a time on the same abscissae (gaiamock has no excess-noise function).
usage: python benchmarks/bench.py [cpu|cuda]   (needs gaiamock: install wobbler[gaiamock], see Installation in the README; TORCHINDUCTOR_CACHE_DIR should be on disk; WOBBLER_COMPILE=0 runs eagerly)"""
import os
import platform
import sys
import time

import gaiamock
import numpy as np
import torch

import wobbler as wb

DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cpu"
CASES, EPOCHS, G = 14400, 191, 13.0
rng = np.random.default_rng(0)
jd = np.sort(rng.uniform(*wb.WINDOWS["DR3"], EPOCHS))
scan = wb.Scan(jd=jd, psi=rng.uniform(0, np.pi, EPOCHS), plx_factor=rng.uniform(-0.9, 0.9, EPOCHS), device=DEVICE)
parameters = dict(parallax=rng.uniform(0.5, 4, CASES), pmra=rng.normal(0, 5, CASES), pmdec=rng.normal(0, 5, CASES), a0_mas=rng.uniform(0, 4, CASES), inclination=np.arccos(rng.uniform(0, 1, CASES)), node=rng.uniform(0, 2 * np.pi, CASES),
                  period_days=rng.uniform(100, 900, CASES), ecc=rng.uniform(0, 0.8, CASES), t_peri_days=rng.uniform(-400, 400, CASES), omega=rng.uniform(0, 2 * np.pi, CASES))
z_err, z_extra = rng.normal(size=(CASES, EPOCHS)), rng.normal(size=(CASES, EPOCHS))
extra, keep = rng.uniform(0, 0.04, CASES), rng.uniform(size=(CASES, EPOCHS)) > 0.1
error = float(wb.epoch_error(G, device="cpu"))
t = scan.t_yr()


def wobbler_run(part=slice(None)):
    observed = wb.observe(scan, error=error, z_err=z_err[part], extra=extra[part], z_extra=z_extra[part], device=DEVICE, **{k: v[part] for k, v in parameters.items()})
    result = wb.fit_five_parameter(t, scan.psi, scan.plx_factor, observed, error, keep=keep[part], device=DEVICE)
    if DEVICE != "cpu":
        torch.cuda.synchronize()
    return result


def timed(function, repeats):
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        times.append(time.perf_counter() - started)
    return float(np.median(times))


def in_batches():
    for start in range(0, CASES, 1000):
        wobbler_run(slice(start, start + 1000))


started = time.perf_counter()
in_batches()
first = time.perf_counter() - started
steady = timed(in_batches, 5)
wobbler_run()  # the first call of this shape compiles
whole = timed(wobbler_run, 5)
observed = wb.observe(scan, error=error, z_err=z_err, extra=extra, z_extra=z_extra, device="cpu", **parameters).numpy()
t_numpy, psi_numpy, plx_numpy = (x.cpu().numpy() for x in (t, scan.psi, scan.plx_factor))


def excess_noise(t_yr, psi, plx, obs, err):
    design = np.vstack([np.sin(psi), t_yr * np.sin(psi), np.cos(psi), t_yr * np.cos(psi), plx]).T
    weight = 1 / err**2
    mu = np.linalg.solve(design.T @ (weight[:, None] * design), design.T @ (weight * obs))
    r = obs - design @ mu
    nu = len(r) - 5
    if np.sum(r**2 * weight) / nu <= 1:
        return 0.0
    low, high = 0.0, 5.0
    for _ in range(60):
        mid = (low + high) / 2
        low, high = (mid, high) if np.sum(r**2 / (err**2 + mid**2)) > nu else (low, mid)
    return (low + high) / 2


def gaiamock_run():
    for i in range(CASES):
        k = keep[i]
        fit = gaiamock.fit_5par_solution_only(t_ast_yr=t_numpy[k], psi=psi_numpy[k], plx_factor=plx_numpy[k], ast_obs=observed[i, k], ast_err=error * np.ones(k.sum()))
        if fit[0] != 0 or fit[4] != 0:
            excess_noise(t_numpy[k], psi_numpy[k], plx_numpy[k], observed[i, k], error * np.ones(k.sum()))


reference = timed(gaiamock_run, 1)
cpu = next((line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo") if line.startswith("model name")), platform.processor())
device = torch.cuda.get_device_name(0) if DEVICE != "cpu" else f"{cpu} ({torch.get_num_threads()} threads)"
print(f"{CASES} cases of {EPOCHS} epochs; wobbler on {device}; python {platform.python_version()}, torch {torch.__version__}, numpy {np.__version__}, gaiamock 1.0.0; compile {wb.compile_enabled()}; TORCHINDUCTOR_CACHE_DIR={os.environ.get('TORCHINDUCTOR_CACHE_DIR')}")
print(f"wobbler, batches of 1,000: first call {first:.2f} s; later calls {steady:.3f} s ({CASES / steady:,.0f} cases per second)")
print(f"wobbler, one batch of {CASES:,}: {whole:.3f} s ({CASES / whole:,.0f} cases per second)")
print(f"gaiamock, one case at a time: {reference:.2f} s ({CASES / reference:,.0f} cases per second); wobbler is {reference / steady:.1f} times faster")
