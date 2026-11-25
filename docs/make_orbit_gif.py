"""Animation of the README: the orbit of Gaia BH1 or Gaia BH3 on the sky, written as docs/orbit-bh1.gif or docs/orbit-bh3.gif.

usage: GAIAMOCK_DATA_DIR=... python docs/make_orbit_gif.py bh1|bh3 [--frames N]   (BH3 needs validation/bh3_catalogue.json and `pip install gaiamock==1.0.0`, as docs/make_bh3_figure.py does)

The scene is the sky with the proper motion kept and the parallax removed, since the parallax is the Earth's motion and not the binary's. The barycentre moves along its straight proper-motion track. The star (the photocentre, because the companion is dark) is the barycentre plus the
orbit's offset from photocentre(), and the dark companion lies opposite it: the barycentre minus (m_star / m_dark) times the star's offset. The view follows the barycentre, so the cross stays at the centre and the orbit is not lost in the length of the track, and each body leaves
a fading trail in sky coordinates. The animation carries no observation marks; the static figures carry Gaia's observations. BH1 runs over the DR3 window; BH3 runs over one orbital period from the start of that window, with a note while the clock is inside it.
"""
import datetime
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "validation"))
sys.path.insert(0, str(HERE))
import wobbler as wb  # noqa: E402
from make_bh1_figure import PAPER  # noqa: E402

REFERENCE_JD = 2457389.0  # J2016.0, the reference epoch of DR3 and of wobbler's times
# Gaia DR3 gaia_source of BH1 (source 4373465352415301632): pmra and pmdec in mas/yr at the reference epoch 2016.0, read 2026-11-25 from the mirror of the archive at ARI Heidelberg (https://gaia.ari.uni-heidelberg.de/tap).
BH1_PROPER_MOTION = (-6.4368962467077075, -25.480478587267733)
# Gaia Collaboration, Panuzzo et al. 2024, A&A 686, L2 (doi:10.1051/0004-6361/202449763, arXiv:2404.10486), Section 3: the black hole's mass 32.70 +- 0.82 solar masses (Eq. 5, and the abstract) and the visible star's 0.76 +- 0.05 (from the isochrones of Section 2.4).
BH3_MASSES = (0.76, 32.70)
TRAIL_DAYS = 130.0  # the length of each body's trail, about the span the view shows at the star's speed along the track
SECONDS = 9.0
FPS = 12


def setup(name):
    """The scene of a star: the dates of the animation, the window of DR3, the barycentre's offset and the star's offset from it as functions of the date, the masses, the size of the view and of the scale bar (mas)."""
    if name == "bh1":
        p = PAPER

        def orbit_of(jd):
            east, north = wb.photocentre((jd - REFERENCE_JD) / 365.25, p["period"], p["ecc"], p["t_peri"], p["a0_mas"], p["inclination"], p["node"], p["omega"])
            return east.cpu().numpy(), north.cpu().numpy()

        window = wb.WINDOWS["DR3"]
        return dict(title="Gaia BH1", jd=np.linspace(window[0], window[1], 10), window=window, orbit_of=orbit_of, proper_motion=BH1_PROPER_MOTION, masses=(p["m_star"], p["m_dark"]), view=5.0, bar=2.0,
                    span=window[1] - window[0], inset=None)
    import bh3

    raw = json.loads((HERE.parent / "validation" / "bh3_catalogue.json").read_text())
    c = {k: float(v) for k, v in raw.items() if k not in ("read_on", "source_id")}
    orbit = bh3.ORBITS["combined"]
    scan = wb.Scan.from_position(c["ra"], c["dec"], window="DR3", device="cpu")
    window = (float(scan.jd.min()), float(scan.jd.max()))
    t_peri = orbit["t_peri_jd"] - bh3.REFERENCE_JD

    def orbit_of(jd):
        east, north = wb.photocentre((jd - bh3.REFERENCE_JD) / 365.25, orbit["period"], orbit["ecc"], t_peri, orbit["a0"], np.radians(orbit["inclination"]), np.radians(orbit["node"]), np.radians(orbit["omega"]))
        return east.cpu().numpy(), north.cpu().numpy()

    return dict(title="Gaia BH3", jd=None, window=window, orbit_of=orbit_of, proper_motion=(orbit["pmra"][0], orbit["pmdec"][0]), masses=BH3_MASSES, view=36.0, bar=20.0, span=orbit["period"], inset=1.0)


def date_of(jd):
    return (datetime.datetime(1970, 1, 1) + datetime.timedelta(days=float(jd) - 2440587.5)).strftime("%Y-%m-%d")


def main():
    name = sys.argv[1]
    frames = int(sys.argv[sys.argv.index("--frames") + 1]) if "--frames" in sys.argv else int(SECONDS * FPS)
    scene = setup(name)
    start = scene["window"][0]
    ratio = scene["masses"][0] / scene["masses"][1]
    pm = scene["proper_motion"]

    def bodies(jd):
        """The barycentre, the star and the companion (east, north in mas, from the barycentre at the reference epoch) at the dates jd."""
        jd = np.atleast_1d(jd)
        years = (jd - REFERENCE_JD) / 365.25
        bary = np.array([pm[0] * years, pm[1] * years])
        offset = np.array(scene["orbit_of"](jd))
        return bary, bary + offset, bary - ratio * offset

    clock = np.linspace(start, start + scene["span"], frames, endpoint=False)
    view = scene["view"]

    plt.rcParams.update({"font.size": 11})
    fig, axis = plt.subplots(figsize=(5.0, 5.0), dpi=60, constrained_layout=True)
    axis.set_aspect("equal")
    axis.invert_xaxis()
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(scene["title"])
    star_trail, companion_trail = (LineCollection([], linewidths=2.2), LineCollection([], linewidths=2.2))
    axis.add_collection(companion_trail)
    axis.add_collection(star_trail)
    star_dot, = axis.plot([], [], "o", color="C0", mec="k", mew=0.8, ms=11, zorder=7)
    companion_dot, = axis.plot([], [], "o", color="k", ms=8, zorder=7)
    cross, = axis.plot([0], [0], "x", color="0.4", mew=2.4, ms=12, zorder=8)
    axis.legend([cross, star_dot, companion_dot], ["barycentre", "star", "dark companion"], loc="upper right", frameon=False, handletextpad=0.3, borderpad=0.2, fontsize=10)
    date_text = axis.text(0.03, 0.04, "", transform=axis.transAxes, fontsize=12, va="bottom")
    window_text = axis.text(0.03, 0.10, "", transform=axis.transAxes, fontsize=10, color="C3", va="bottom")
    axis.text(0.03, 0.965, "the view follows the barycentre", transform=axis.transAxes, fontsize=9, va="top", color="0.35")
    bar = scene["bar"]
    axis.plot([-0.0], [0], alpha=0)  # keeps the data limits valid before the first frame sets them
    scale_line, = axis.plot([], [], color="k", lw=2)
    scale_text = axis.text(0, 0, f"{bar:g} mas", fontsize=9, ha="center", va="bottom")
    zoom = None
    if scene["inset"]:
        zoom = axis.inset_axes([0.60, 0.07, 0.36, 0.36])
        zoom.set_aspect("equal")
        zoom.invert_xaxis()
        zoom.set_xlim(scene["inset"], -scene["inset"])
        zoom.set_ylim(-scene["inset"], scene["inset"])
        zoom.set_xticks([])
        zoom.set_yticks([])
        zoom.set_title("zoom on the companion", fontsize=8)
        zoom_trail = LineCollection([], linewidths=1.8)
        zoom.add_collection(zoom_trail)
        zoom.plot([0], [0], "x", color="0.4", mew=2, ms=9)
        zoom_dot, = zoom.plot([], [], "o", color="k", ms=6)

    def fading(points, colour):
        """Line segments through points (2 x n) with an opacity that grows from 0 at the oldest to 1 at the newest."""
        segments = np.stack([points[:, :-1].T, points[:, 1:].T], axis=1)
        alpha = np.linspace(0.0, 1.0, len(segments)) ** 1.5
        rgba = np.tile(matplotlib.colors.to_rgba(colour), (len(segments), 1))
        rgba[:, 3] = alpha
        return segments, rgba

    def update(index):
        now = clock[index]
        past = np.linspace(now - TRAIL_DAYS, now, 60)
        bary_past, star_past, companion_past = bodies(past)
        bary, star, companion = (x[:, -1] for x in (bary_past, star_past, companion_past))
        axis.set_xlim(bary[0] + view, bary[0] - view)
        axis.set_ylim(bary[1] - view, bary[1] + view)
        for collection, points, colour in ((star_trail, star_past, "C0"), (companion_trail, companion_past, "0.1")):
            segments, rgba = fading(points, colour)
            collection.set_segments(segments)
            collection.set_color(rgba)
        star_dot.set_data([star[0]], [star[1]])
        companion_dot.set_data([companion[0]], [companion[1]])
        cross.set_data([bary[0]], [bary[1]])
        date_text.set_text(date_of(now))
        inside = scene["window"][0] <= now <= scene["window"][1]
        window_text.set_text("in Gaia DR3's window" if inside and name == "bh3" else "")
        corner = (bary[0] + 0.16 * view, bary[1] - 0.9 * view)  # the scale bar is below the cross, between the date and the inset
        scale_line.set_data([corner[0] - bar / 2, corner[0] + bar / 2], [corner[1], corner[1]])
        scale_text.set_position((corner[0], corner[1] + 0.03 * view))
        if zoom is not None:
            relative = companion_past - bary_past
            segments, rgba = fading(relative, "0.1")
            zoom_trail.set_segments(segments)
            zoom_trail.set_color(rgba)
            zoom_dot.set_data([relative[0, -1]], [relative[1, -1]])
        return ()

    out = HERE / f"orbit-{name}.gif"
    note = animation.FuncAnimation(fig, update, frames=frames, blit=False)
    note.save(out, writer=animation.PillowWriter(fps=FPS))
    stills = [int(x) for x in np.linspace(0, frames - 1, 4)]
    for number, index in enumerate(stills):
        update(index)
        fig.savefig(HERE / f"orbit-{name}-frame{number}.png", dpi=60)
    print(out, f"{out.stat().st_size / 1e6:.2f} MB", frames, "frames,", f"{frames / FPS:.1f} s per loop")


if __name__ == "__main__":
    main()
