"""Figure of the README: Gaia BH1 through wobbler.

Gaia DR3 4373465352415301632 is a Sun-like star on a 185.6 d orbit around a dark companion of 9.6 solar masses (El-Badry et al. 2023, MNRAS 518, 1057, doi:10.1093/mnras/stac3140; Table 1, astrometry + RVs).
The scan epochs are the Gaia scanning law as precomputed by gaiamock (El-Badry et al. 2024, https://github.com/kareemelbadry/gaiamock), read by Scan.from_position: `pip install gaiamock==1.0.0`, then `gaiamock-download` once (about 290 MB; GAIAMOCK_DATA_DIR sets where the data is kept).
The catalogue values come from the Gaia archive (https://gea.esac.esa.int/tap-server/tap, tables gaiadr3.gaia_source and gaiadr3.nss_two_body_orbit).

usage: python docs/make_bh1_figure.py [--catalogue]   (--catalogue queries the Gaia archive, otherwise the values below, taken from it on 2026-11-20, are used)
"""
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from wobbler import u0, uwe_factor, WINDOWS, Orbit, Scan, fit_five_parameter, observe, epoch_error, photocentre, simulate_fit

SOURCE_ID = 4373465352415301632
# Gaia DR3 gaia_source and nss_two_body_orbit of the source (Gaia archive).
CATALOGUE = dict(ra=262.17120727595824, dec=-0.5810919626714084, G=13.771998, parallax=2.069988584035135, parallax_error=0.116677545, ruwe=7.6376343, excess_noise=1.0223111, nu_eff=1.4811925,
                 period=185.7656578920495, ecc=0.48893589298452034, t_peri=-12.024680365644883, A=-0.262289119199277, B=2.9291159041806485, F=1.5248071482113947, G_ti=0.5343685858549198)
# El-Badry et al. 2023, Table 1, astrometry + RVs (Gaia BH1), and K of the RV-only fit.
PAPER = dict(period=185.59, ecc=0.451, t_peri=-1.1, omega=np.radians(12.8), node=np.radians(97.8), inclination=np.radians(126.6), a0_mas=2.67, k_star=66.7, m_star=0.93, m_dark=9.62)
AU_KM = 1.495978707e8


def catalogue_from_archive():
    query = ("SELECT s.ra,s.dec,s.phot_g_mean_mag,s.parallax,s.parallax_error,s.ruwe,s.astrometric_excess_noise,s.nu_eff_used_in_astrometry,o.period,o.eccentricity,o.t_periastron,"
             "o.a_thiele_innes,o.b_thiele_innes,o.f_thiele_innes,o.g_thiele_innes FROM gaiadr3.gaia_source s JOIN gaiadr3.nss_two_body_orbit o USING (source_id) "
             f"WHERE s.source_id={SOURCE_ID}")
    data = urllib.parse.urlencode({"QUERY": query, "REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv"}).encode()
    row = urllib.request.urlopen("https://gea.esac.esa.int/tap-server/tap/sync", data, timeout=60).read().decode().splitlines()[1].split(",")
    keys = ["ra", "dec", "G", "parallax", "parallax_error", "ruwe", "excess_noise", "nu_eff", "period", "ecc", "t_peri", "A", "B", "F", "G_ti"]
    return dict(zip(keys, map(float, row)))


def sky(*arguments):
    """The photocentre offsets (east, north; mas) as NumPy arrays."""
    return tuple(x.cpu().numpy() for x in photocentre(*arguments))


def earth_offsets(jd, ra_deg, dec_deg, parallax):
    """The parallax displacement (east, north, mas) of a star at the given position for a barycentric Earth at the Julian dates jd. The Earth is placed by the low-precision solar formulae of the Astronomical Almanac (Sun's mean longitude and anomaly, accurate to about 0.01 degrees); the displacement is parallax * (X sin a - Y cos a) east and parallax * (X cos a sin d + Y sin a sin d - Z cos d) north for the Earth's position (X, Y, Z) in AU."""
    n = jd - 2451545.0
    mean_longitude = np.radians(280.460 + 0.9856474 * n)
    anomaly = np.radians(357.528 + 0.9856003 * n)
    longitude = mean_longitude + np.radians(1.915) * np.sin(anomaly) + np.radians(0.020) * np.sin(2 * anomaly)
    radius = 1.00014 - 0.01671 * np.cos(anomaly) - 0.00014 * np.cos(2 * anomaly)
    obliquity = np.radians(23.439 - 0.0000004 * n)
    sun = np.stack([radius * np.cos(longitude), radius * np.sin(longitude) * np.cos(obliquity), radius * np.sin(longitude) * np.sin(obliquity)])
    X, Y, Z = -sun  # the Earth as seen from the Sun
    a, d = np.radians(ra_deg), np.radians(dec_deg)
    return parallax * (X * np.sin(a) - Y * np.cos(a)), parallax * (X * np.cos(a) * np.sin(d) + Y * np.sin(a) * np.sin(d) - Z * np.cos(d))


def main():
    cat = catalogue_from_archive() if "--catalogue" in sys.argv else CATALOGUE
    scan = Scan.from_position(cat["ra"], cat["dec"], window="DR3", device="cpu")
    jd, psi, plx_factor = (x.numpy() for x in (scan.jd, scan.psi, scan.plx_factor))
    p = PAPER
    parallax = 2.09  # mas, El-Badry et al. 2023, Table 1
    a1_sini_au = p["k_star"] * p["period"] * 86400 * np.sqrt(1 - p["ecc"] ** 2) / (2 * np.pi) / AU_KM
    orbit = Orbit(period_days=p["period"], ecc=p["ecc"], t_peri_days=p["t_peri"], omega=p["omega"])
    window = {"DR3": WINDOWS["DR3"]}
    t = scan.t_yr().numpy()

    # Panel a: the sky path, proper motion removed: the parallax loop alone and the loop plus the orbit, over the DR3 window.
    check_east, check_north = earth_offsets(jd, cat["ra"], cat["dec"], 1.0)
    along = check_east * np.sin(psi) + check_north * np.cos(psi)
    print("Earth model against the scan's along-scan parallax factor: largest difference", float(np.max(np.abs(along - plx_factor))))
    days = np.linspace(window["DR3"][0], window["DR3"][1], 800)
    loop_east, loop_north = earth_offsets(days, cat["ra"], cat["dec"], parallax)
    orbit_east, orbit_north = sky((days - 2457389.0) / 365.25, p["period"], p["ecc"], p["t_peri"], p["a0_mas"], p["inclination"], p["node"], p["omega"])
    # photocentre takes the time in years from the reference epoch
    obs_loop_east, obs_loop_north = earth_offsets(jd, cat["ra"], cat["dec"], parallax)
    obs_orbit_east, obs_orbit_north = sky(t, p["period"], p["ecc"], p["t_peri"], p["a0_mas"], p["inclination"], p["node"], p["omega"])

    # Panel b: the same scan simulated twice with the same noise, once for a single star and once for the star with its companion, each with Gaia's single-star fit.
    rng = np.random.default_rng(2)
    error = float(epoch_error(cat["G"], device="cpu"))
    z_err, z_extra = rng.normal(size=(1, len(t))), rng.normal(size=(1, len(t)))
    design = np.stack([np.sin(psi), t * np.sin(psi), np.cos(psi), t * np.cos(psi), plx_factor], axis=1)
    residuals, ruwes = {}, {}
    for name, a0 in (("single", 0.0), ("companion", p["a0_mas"])):
        observed = observe(scan, parallax, 0.0, 0.0, a0, p["inclination"], p["node"], p["period"], p["ecc"], p["t_peri"], p["omega"], error, z_err, 0.0, z_extra, device="cpu").numpy()
        fit = {k: v.numpy() if hasattr(v, "numpy") else v for k, v in fit_five_parameter(t, psi, plx_factor, observed, error, device="cpu").items()}
        residuals[name] = observed[0] - design @ fit["parameters"][0]
        ruwes[name] = float(fit["uwe"][0]) * float(uwe_factor(cat["G"])) / float(u0(cat["G"], cat["nu_eff"], device="cpu"))
    print("one simulated scan: RUWE single star", ruwes["single"], "with companion", ruwes["companion"], "residual rms", {k: float(np.sqrt(np.mean(v**2))) for k, v in residuals.items()}, "epoch error", error)

    # Panel c: RUWE against inclination for many node orientations.
    inclinations = np.radians(np.arange(30, 150.1, 2.5))
    ruwe_band = []
    for i, inc in enumerate(inclinations):
        sim = simulate_fit(scan, orbit, a1_au=a1_sini_au, parallax_mas=np.array([parallax]), G=cat["G"], n_orient=400, inclination=inc, windows=window, seed=i, device="cpu", nu_eff=cat["nu_eff"])
        ok = sim.windows["DR3"]["ok"][0].numpy()
        ruwe_band.append(np.percentile(sim.ruwe[0].numpy()[ok], [16, 50, 84]))
    ruwe_band = np.array(ruwe_band)
    degrees = np.degrees(inclinations)
    matches = (ruwe_band[:, 0] <= cat["ruwe"]) & (cat["ruwe"] <= ruwe_band[:, 2])
    ranges = []  # the runs of consecutive inclinations of the grid whose band holds the catalogue's RUWE
    start = None
    for i, flag in enumerate(matches):
        if flag and start is None:
            start = i
        if start is not None and (not flag or i == len(matches) - 1):
            stop = i if flag else i - 1
            ranges.append((degrees[start], degrees[stop]))
            start = None
    print("inclinations whose band holds the catalogue RUWE:", [(float(x), float(y)) for x, y in ranges], "n", int(matches.sum()), "of", len(matches))

    plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12, "legend.fontsize": 11})
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(16, 5.6), constrained_layout=True)
    # panel 1: the path on the sky
    a.plot(loop_east, loop_north, color="0.55", lw=2, label="single star")
    a.plot(loop_east + orbit_east, loop_north + orbit_north, color="C0", lw=1.6, label="star with its companion")
    a.plot(obs_loop_east + obs_orbit_east, obs_loop_north + obs_orbit_north, "o", color="C0", mec="k", mew=0.6, ms=6, label=f"Gaia's {len(scan.jd)} observations")
    a.invert_xaxis()
    a.set_aspect("equal")
    a.set_xlabel("east (mas)")
    a.set_ylabel("north (mas)")
    a.set_title("The star's path on the sky")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False)
    # panel 2: what the single-star fit leaves
    years = 2000.0 + (jd - 2451544.5) / 365.25
    b.axhline(0, color="k", lw=0.6)
    b.plot(years, residuals["single"], "o", color="0.45", ms=5, label="single star")
    b.plot(years, residuals["companion"], "o", color="C3", ms=5, label="star with its companion")
    b.set_xlabel("year")
    b.set_ylabel("measurement minus fit, along the scan (mas)")
    b.set_title("How far Gaia's measurements\nmiss the single-star fit")
    b.set_ylim(-2.4, 1.8)
    b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False)
    b.text(0.03, 0.04, f"RUWE is about 1 for a single star.\nThis draw gives {ruwes['companion']:.1f}; Gaia DR3 reports {cat['ruwe']:.2f}.", transform=b.transAxes, ha="left", va="bottom", fontsize=11, bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.95))
    # panel 3: RUWE for each tilt of the orbit
    step = degrees[1] - degrees[0]
    for k, (low, high) in enumerate(ranges):
        c.axvspan(low - step / 2, high + step / 2, color="C2", alpha=0.25, lw=0, label="tilts that agree with Gaia" if k == 0 else None)
    c.fill_between(degrees, ruwe_band[:, 0], ruwe_band[:, 2], color="C0", alpha=0.3, lw=0, label="middle 68 percent of orbits")
    c.plot(degrees, ruwe_band[:, 1], color="C0", lw=2, label="median")
    c.axhline(cat["ruwe"], color="k", lw=1.5, label=f"Gaia DR3: {cat['ruwe']:.2f}")
    c.axvline(np.degrees(p["inclination"]), color="C3", ls="--", lw=1.5, label="published tilt, 126.6")
    c.set_xlabel("inclination (degrees; 90 = seen edge-on)")
    c.set_ylabel("RUWE")
    c.set_title("RUWE for each tilt of the orbit")
    c.set_ylim(0, 16)
    c.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False, ncol=2)
    out = Path(__file__).parent / "fig-bh1.png"
    fig.savefig(out, dpi=100)
    print(out)


if __name__ == "__main__":
    main()
