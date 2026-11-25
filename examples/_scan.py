"""A synthetic scan for the examples: 70 epochs over the DR3 window with random scan angles and along-scan parallax factors. It stands for a real scan (Scan.from_gost, Scan.from_position, wobbler.sources) so that the examples run without data or a network."""
import numpy as np

from wobbler import Scan


def synthetic_scan(epochs=70, seed=3):
    rng = np.random.default_rng(seed)
    jd = np.sort(rng.uniform(2456891.5, 2457902.0, epochs))
    psi = rng.uniform(0, np.pi, epochs)
    plx_factor = rng.uniform(-1, 1, epochs)  # the along-scan parallax factor lies in [-1, 1]
    return Scan(jd=jd, psi=psi, plx_factor=plx_factor, device="cpu")
