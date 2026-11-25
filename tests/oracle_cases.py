"""The inputs and the outputs of the frozen NumPy implementation (oracle_numpy.py) for the cases the torch package is checked on: the orbit functions, the mass functions, the five-parameter fit with constant and varying epoch errors and masks, and a seeded simulate_fit, all from fixed seeds. build() runs the oracle; the digest in golden/oracle_digest.json (written by golden/record_oracle_digest.py) fixes what it returned when it was frozen, so that an edit of the oracle is seen."""
from functools import lru_cache

import numpy as np

import oracle_numpy as o


@lru_cache(maxsize=1)
def build():

    rng = np.random.default_rng(20261120)
    out = {}
    # orbit functions
    mean = rng.uniform(-10, 10, 2000)
    ecc = rng.uniform(0, 0.95, 2000)
    out.update(kepler_mean=mean, kepler_ecc=ecc, kepler_E=o.kepler(mean, ecc))
    a, node, omega, incl = rng.uniform(0.1, 5, 500), rng.uniform(0, 2 * np.pi, 500), rng.uniform(0, 2 * np.pi, 500), np.arccos(rng.uniform(-1, 1, 500))
    A, B, F, G = o.thiele_innes(a, node, omega, incl)
    camp = o.campbell(A, B, F, G)
    out.update(ti_a=a, ti_node=node, ti_omega=omega, ti_incl=incl, ti_ABFG=np.stack([A, B, F, G]), camp=np.stack([camp["a"], camp["node"], camp["omega"], camp["inclination"]]))
    period, e2, tp, om = rng.uniform(50, 900, 500), rng.uniform(0, 0.9, 500), rng.uniform(-400, 400, 500), rng.uniform(0, 2 * np.pi, 500)
    t_days = rng.uniform(-300, 900, (500, 7))
    out.update(rv_t=t_days, rv_period=period, rv_ecc=e2, rv_tp=tp, rv_omega=om, rv_k=rng.uniform(1, 30, 500))
    out["rv"] = o.radial_velocity(t_days, period[:, None], e2[:, None], tp[:, None], om[:, None], out["rv_k"][:, None])
    t_yr = rng.uniform(-1, 3, (500, 9))
    a0 = rng.uniform(0.1, 6, 500)
    x, y = o.photocentre(t_yr, period[:, None], e2[:, None], tp[:, None], a0[:, None], incl[:, None], node[:, None], omega[:, None])
    out.update(pc_t=t_yr, pc_a0=a0, pc_xy=np.stack([x, y]), true_anomaly_in=rng.uniform(-3, 3, 300), true_anomaly_ecc=rng.uniform(0, 0.9, 300))
    out["true_anomaly"] = o.true_anomaly(out["true_anomaly_in"], out["true_anomaly_ecc"])
    # mass
    f = rng.uniform(0, 3, 300)
    m1 = rng.uniform(0.5, 2.0, 300)
    out.update(mass_period=period[:300], mass_k=rng.uniform(1, 80, 300), mass_ecc=e2[:300], mass_f=f, mass_m1=m1, mass_incl=np.arccos(rng.uniform(0.1, 1, 300)))
    out["mass_function"] = o.mass_function(out["mass_period"], out["mass_k"], out["mass_ecc"])
    out["min_companion_mass"] = o.min_companion_mass(f, m1)
    out["companion_mass"] = o.companion_mass(f, m1, out["mass_incl"])
    out["share_inclined"] = o.share_inclined_below(np.linspace(-0.2, 1.8, 30))
    # the scan and the fit
    n = 191
    jd = np.sort(rng.uniform(*o.WINDOWS["DR3"], n))
    scan = o.Scan(jd=jd, psi=rng.uniform(0, np.pi, n), plx_factor=rng.uniform(-0.9, 0.9, n))
    cases = 100
    parameters = dict(parallax=rng.uniform(0.5, 4, cases), pmra=rng.normal(0, 5, cases), pmdec=rng.normal(0, 5, cases), a0_mas=rng.uniform(0, 4, cases), inclination=np.arccos(rng.uniform(0, 1, cases)), node=rng.uniform(0, 2 * np.pi, cases),
                      period_days=rng.uniform(100, 900, cases), ecc=rng.uniform(0, 0.8, cases), t_peri_days=rng.uniform(-400, 400, cases), omega=rng.uniform(0, 2 * np.pi, cases))
    z_err, z_extra, extra, keep = rng.normal(size=(cases, n)), rng.normal(size=(cases, n)), rng.uniform(0, 0.04, cases), rng.uniform(size=(cases, n)) > 0.1
    error = float(o.epoch_error(13.0))
    observed = o.observe(scan, error=error, z_err=z_err, extra=extra, z_extra=z_extra, **parameters)
    out.update({f"fit_{k}": v for k, v in parameters.items()}, fit_jd=jd, fit_psi=scan.psi, fit_plx=scan.plx_factor, fit_z_err=z_err, fit_z_extra=z_extra, fit_extra=extra, fit_keep=keep, fit_error=np.array(error), fit_observed=observed)
    t = scan.t_yr()
    for label, err in (("const", error), ("vary", error * rng.uniform(0.8, 1.5, n))):
        fit = o.fit_five_parameter(t, scan.psi, scan.plx_factor, observed, err, keep=keep)
        out[f"fit_{label}_error"] = np.asarray(err)
        for key in ("parameters", "parameter_errors", "ruwe", "excess_noise", "chi2", "degrees_of_freedom", "n_epochs", "visibility_periods", "ok"):
            out[f"fit_{label}_{key}"] = fit[key]
    # a seeded simulate_fit
    orbit = o.Orbit(period_days=325.006, ecc=0.2103, t_peri_days=-110.37, omega=1.1048)
    sim = o.simulate_fit(scan, orbit, a1_au=1.2, parallax_mas=np.linspace(0.5, 4, 20), G=12.57, n_orient=10, seed=3)
    out["sim_parallax"] = np.linspace(0.5, 4, 20)
    for name, w in sim.windows.items():
        for key, v in w.items():
            out[f"sim_{name}_{key}"] = v
    out["sim_incl"], out["sim_node"] = sim.inclination, sim.node
    return out
