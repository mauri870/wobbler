"""The orbit and mass functions in NumPy (wobbler.light), which need no torch: the mass function of a spectroscopic orbit and the companion mass at an inclination."""
import numpy as np

from wobbler import light

f = light.mass_function(period_days=185.59, k_kms=66.7, ecc=0.45)
print(f"mass function {f:.2f} solar masses")
for inclination_deg in (30, 60, 90):
    print(f"inclination {inclination_deg} deg: companion mass {light.companion_mass(f, 0.93, np.radians(inclination_deg)):.2f} solar masses")
assert light.min_companion_mass(f, 0.93) > 0
