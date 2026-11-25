"""Does the uncertainty of the orbit solution's own elements account for the width of the offset residual of validation/scoreboard.py?

Each star's elements (parallax, pmra, pmdec, A, B, F, G, eccentricity, period, time of periastron) are drawn from the covariance of its nss_two_body_orbit row: the *_error columns scaled by the correlation matrix of corr_vec. Every draw is one case with its own orbit and its own noise
draw; the fitted parallax and proper motion minus the nominal ones of the orbit solution are the predicted offsets. The statistics are those of validation/scoreboard.py with the spread of the draws, which now includes the elements.

corr_vec is the upper triangle of the correlation matrix without the diagonal, column-major (the pair (1,2), then (1,3), (2,3), then (1,4), (2,4), (3,4), ...), over the parameters of the solution type in the order of the DR3 data model of nss_two_body_orbit
(https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_non--single_stars_tables/ssec_dm_nss_two_body_orbit.html): ORBITAL and ASTROSPECTROSB1 below (12 and 15 parameters, 66 and 105 values).

usage:
  simulate ROWS ORBITS OUT [--limit N] [--draws 100] [--device cpu|cuda]   the stars of half B of G below 13 of ROWS (as scoreboard.py) with their rows of ORBITS, in part files of OUT
  score ROWS OUT --scan-widths P PMRA PMDEC [--json PATH]                    the statistics of the draws against the archive
ORBITS is a directory of files *.json, each {"rows": [...]} with source_id, type, bit_index, corr_vec (a string '(x, y, ...)') and the columns ra_error, dec_error, parallax_error, pmra_error, pmdec_error, a_error, b_error, f_error, g_error, c_error, h_error,
period_error, t_peri_error, ecc_error, velocity_error."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scoreboard as sb  # noqa: E402

ORBITAL = ("ra", "dec", "parallax", "pmra", "pmdec", "A", "B", "F", "G", "ecc", "period", "t_peri")
ASTROSPECTROSB1 = ("ra", "dec", "parallax", "pmra", "pmdec", "A", "B", "F", "G", "C", "H", "velocity", "ecc", "period", "t_peri")
USED = ("parallax", "pmra", "pmdec", "A", "B", "F", "G", "ecc", "period", "t_peri")
ERROR_OF = {"ra": "ra_error", "dec": "dec_error", "parallax": "parallax_error", "pmra": "pmra_error", "pmdec": "pmdec_error", "A": "a_error", "B": "b_error", "F": "f_error", "G": "g_error", "C": "c_error", "H": "h_error",
            "velocity": "velocity_error", "ecc": "ecc_error", "period": "period_error", "t_peri": "t_peri_error"}
NOMINAL_OF = {"parallax": "orb_parallax", "pmra": "orb_pmra", "pmdec": "orb_pmdec", "A": "ti_a", "B": "ti_b", "F": "ti_f", "G": "ti_g", "ecc": "ecc", "period": "period", "t_peri": "t_peri"}
MAXIMUM_ECCENTRICITY, MINIMUM_PERIOD = 0.99, 1.0
OFFSETS = ("parallax", "pmra", "pmdec")
PER_STAR = ("d_parallax", "d_pmra", "d_pmdec", "o16_parallax", "o84_parallax", "o16_pmra", "o84_pmra", "o16_pmdec", "o84_pmdec", "clipped", "n_ok")


def parse_vector(text):
    return np.array([float(x) for x in text.strip().strip("()").split(",") if x.strip()])


def correlation(vector, n):
    """The correlation matrix of n parameters from the column-major upper triangle without the diagonal."""
    r = np.eye(n)
    k = 0
    for j in range(1, n):
        for i in range(j):
            r[i, j] = r[j, i] = vector[k]
            k += 1
    return r


def mean_and_covariance(star, row):
    """The nominal values and the covariance of USED for a star (its orbit solution values) and the row of ORBITS; a parameter whose error is empty or not finite is held fixed (variance and covariances zero)."""
    names = ORBITAL if row["type"] == "Orbital" else ASTROSPECTROSB1
    vector = parse_vector(row["corr_vec"])
    n = len(names)
    if len(vector) != n * (n - 1) // 2:
        raise ValueError(f"corr_vec has {len(vector)} values for {n} parameters of {row['type']}")
    r = np.nan_to_num(correlation(vector, n), nan=0.0)  # the correlations of a parameter that was not fitted (bit_index 8179 and 65435) are NaN
    index = [names.index(name) for name in USED]
    sigma = np.array([float(row[ERROR_OF[name]]) if row[ERROR_OF[name]] not in ("", None) else 0.0 for name in USED])
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    covariance = r[np.ix_(index, index)] * sigma[:, None] * sigma[None, :]
    return np.array([float(star[NOMINAL_OF[name]]) for name in USED]), covariance


def sample(mean, covariance, draws, rng):
    """`draws` draws of the elements from a Gaussian by the eigendecomposition of the covariance (negative eigenvalues of rounding set to zero); eccentricity is clipped to 0 to MAXIMUM_ECCENTRICITY and the period to MINIMUM_PERIOD or more.
    Returns the draws and the number of draws that needed a clip."""
    values, vectors = np.linalg.eigh(covariance)
    draws_ = mean + (rng.standard_normal((draws, len(mean))) * np.sqrt(np.maximum(values, 0.0))) @ vectors.T
    fixed = np.diag(covariance) == 0
    draws_[:, fixed] = mean[fixed]  # a parameter with no variance is exactly its nominal value, not the nominal value plus the rounding of the decomposition
    e, p = USED.index("ecc"), USED.index("period")
    clipped = int(((draws_[:, e] < 0) | (draws_[:, e] > MAXIMUM_ECCENTRICITY) | (draws_[:, p] < MINIMUM_PERIOD)).sum())
    draws_[:, e] = np.clip(draws_[:, e], 0.0, MAXIMUM_ECCENTRICITY)
    draws_[:, p] = np.maximum(draws_[:, p], MINIMUM_PERIOD)
    return draws_, clipped


def draw_noise(wb, cases, epochs, g, seed, device):
    """The noise of simulate_fit for `cases` cases: the rejected epochs, the epoch noise, the extra noise and its draws, from a torch generator seeded as simulate_fit seeds its first slice."""
    import torch

    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed) * 1_000_003)
    keep = torch.rand(cases, epochs, generator=generator, dtype=torch.float64, device=device) > wb.gaiafit.REJECTED_SHARE
    z_err = torch.randn(cases, epochs, generator=generator, dtype=torch.float64, device=device)
    z_extra = torch.randn(cases, epochs, generator=generator, dtype=torch.float64, device=device)
    extra = torch.rand(cases, generator=generator, dtype=torch.float64, device=device) * wb.gaiafit.BRIGHT_EXTRA_NOISE if g < wb.gaiafit.BRIGHT_LIMIT else torch.zeros(cases, dtype=torch.float64, device=device)
    return {"keep": keep, "z_err": z_err, "z_extra": z_extra, "extra": extra}


def cases_through_the_fit(wb, scan, g, nu_eff, samples, nominal, noise, device, noise_scale=1.0):
    """Each row of `samples` (USED order) is one case: its orbit by campbell, its noise from `noise` with the epoch error multiplied by `noise_scale` (the fit keeps weighting with the unscaled error, as simulate_fit's formal_error_factor does); the fit of the DR3 window. Returns numpy arrays over the cases: uwe, ruwe, the offsets d_parallax, d_pmra, d_pmdec (fitted minus `nominal`) and ok."""
    import torch
    from wobbler import gaiafit

    cases = len(samples)
    parallax, pmra, pmdec, a, b, f, gg, ecc, period, t_peri = (samples[:, k] for k in range(10))
    elements = wb.campbell(a, b, f, gg, device=device)
    incl, node, omega, a0 = (elements[k] for k in ("inclination", "node", "omega", "a"))
    scan = scan.to(device)
    t, inside = scan.t_yr(), scan.window(*wb.WINDOWS["DR3"])
    tensors = {key: torch.as_tensor(value, device=device) for key, value in noise.items()}
    error = torch.as_tensor(gaiafit.epoch_error(np.full(cases, float(g)), device="cpu").numpy()[:, None], device=device)
    observed = wb.observe(scan, parallax, pmra, pmdec, a0, incl, node, period, ecc, t_peri, omega, error * noise_scale, tensors["z_err"], tensors["extra"], tensors["z_extra"], device=device)
    u0 = wb.u0(np.full(cases, float(g)), np.full(cases, float(nu_eff)), device=device)
    fit = gaiafit.fit_five_parameter(t[inside], scan.psi[inside], scan.plx_factor[inside], observed[:, inside], error, keep=tensors["keep"][:, inside], errors=False, u0=u0, device=device)
    parameters = fit["parameters"].cpu().numpy()
    return {"uwe": fit["uwe"].cpu().numpy(), "ruwe": fit["ruwe"].cpu().numpy(), "d_parallax": parameters[:, 4] - nominal[0], "d_pmra": parameters[:, 1] - nominal[1], "d_pmdec": parameters[:, 3] - nominal[2], "ok": fit["ok"].cpu().numpy()}


def load_orbits(directory):
    rows = {}
    for path in sorted(Path(directory).glob("g*.json")):
        for row in json.loads(path.read_text())["rows"]:
            rows[int(row["source_id"])] = row
    return rows


def simulate(rows, orbits, out, limit=0, draws=100, device="cpu", part_size=2000, scale_by_f=False):
    import wobbler as wb

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    stars = sb.load_rows(rows)
    half = sb.halves([s["g"] for s in stars])
    wanted = [s for s, h in zip(stars, half) if s["g"] < sb.BRIGHT_LIMIT and h == "B"]
    elements = load_orbits(orbits)
    done = set()
    for path in out.glob("part-*.npz"):
        done |= set(np.load(path)["source_id"].tolist())
    todo = [s for s in wanted if s["source_id"] not in done and s["source_id"] in elements]
    todo = todo[:limit] if limit else todo
    for start in range(0, len(todo), part_size):
        results = []
        for star in todo[start:start + part_size]:
            mean, covariance = mean_and_covariance(star, elements[star["source_id"]])
            rng = np.random.default_rng(int(star["source_id"]) % 2**32)
            samples, clipped = sample(mean, covariance, draws, rng)
            scan = wb.Scan.from_position(star["ra"], star["dec"], device=device)
            noise = draw_noise(wb, draws, len(scan.jd), star["g"], int(star["source_id"]) % 2**31, device)
            fit = cases_through_the_fit(wb, scan, star["g"], star["nu_eff"], samples, mean[:3], noise, device, noise_scale=float(wb.uwe_factor(star["g"])) if scale_by_f else 1.0)
            ok = fit["ok"]
            if ok.sum() < draws // 2:
                continue
            row = {"source_id": star["source_id"], "clipped": clipped, "n_ok": int(ok.sum())}
            for p in OFFSETS:
                values = fit["d_" + p][ok]
                row["d_" + p] = float(np.median(values))
                row["o16_" + p], row["o84_" + p] = (float(np.percentile(values, q)) for q in (16, 84))
            results.append(row)
        arrays = {"source_id": np.array([r["source_id"] for r in results], dtype=np.int64)} | {key: np.array([r[key] for r in results]) for key in PER_STAR}
        np.savez(out / f"part-{len(list(out.glob('part-*.npz'))):04d}.npz", **arrays)
        print(f"{start + len(todo[start:start + part_size])} of {len(todo)}", flush=True)
    return len(todo)


def score(rows, out, scan_widths, json_path=None):
    stars = {s["source_id"]: s for s in sb.load_rows(rows)}
    parts = [np.load(p) for p in sorted(Path(out).glob("part-*.npz"))]
    predicted = {key: np.concatenate([p[key] for p in parts]) for key in parts[0].files}
    chosen = [stars[int(i)] for i in predicted["source_id"]]
    archive = {p: np.array([s[p] - s["orb_" + p] for s in chosen]) for p in OFFSETS}
    error = {p: np.array([s[p + "_error"] for s in chosen]) for p in OFFSETS}
    g = np.array([s["g"] for s in chosen])
    rng = np.random.default_rng(sb.SEED_BOOTSTRAP)
    values = {}
    for p, width in zip(OFFSETS, scan_widths):  # the scan's width is in units of the formal error; the archive's stands for the predicted one (they agree to 4 percent in the scoreboard)
        spread = (predicted["o84_" + p] - predicted["o16_" + p]) / 2
        residual = archive[p] - predicted["d_" + p]
        values["z_el_" + p] = residual / error[p]
        values["pull_el_" + p] = residual / spread
        values["pull_el_scan_" + p] = residual / np.sqrt(spread**2 + (width * error[p]) ** 2)
    result = {"stars": int(len(chosen)), "scan_widths": list(scan_widths), "draws_that_needed_a_clip": int((predicted["clipped"] > 0).sum()), "overall": {}, "G": {}}
    for key, x in values.items():
        result["overall"][key] = sb.summary(x, rng.integers(0, len(x), size=(sb.RESAMPLES, len(x))))
    for lo, hi in zip(sb.G_EDGES[:4], sb.G_EDGES[1:5]):
        mask = (g >= lo) & (g < hi)
        index = rng.integers(0, int(mask.sum()), size=(sb.RESAMPLES, int(mask.sum())))
        result["G"][f"{lo:g} to {hi:g}"] = {key: sb.summary(x[mask], index) for key, x in values.items()}
    Path(json_path or Path(out) / "elements.json").write_text(json.dumps(result, indent=1) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("simulate")
    one.add_argument("rows"), one.add_argument("orbits"), one.add_argument("out")
    one.add_argument("--limit", type=int, default=0), one.add_argument("--draws", type=int, default=100), one.add_argument("--device", default="cpu"), one.add_argument("--noise-scale-f", action="store_true", help="multiply the epoch noise by the single-star factor f(G) (the fit keeps the raw weights)")
    two = sub.add_parser("score")
    two.add_argument("rows"), two.add_argument("out"), two.add_argument("--scan-widths", type=float, nargs=3, required=True), two.add_argument("--json")
    arguments = parser.parse_args()
    if arguments.command == "simulate":
        simulate(arguments.rows, arguments.orbits, arguments.out, arguments.limit, arguments.draws, arguments.device, scale_by_f=arguments.noise_scale_f)
    else:
        score(arguments.rows, arguments.out, arguments.scan_widths, arguments.json)


if __name__ == "__main__":
    sys.exit(main())
