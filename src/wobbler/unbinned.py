"""The sky-dependent epoch error of gaiamock's unbinned mode (https://github.com/kareemelbadry/gaiamock, gaiamock/gaiamock.py: get_realistic_epoch_astrometry_errors; README section "A modified version to predict RUWE more reliably for small orbits and single stars").

In that mode the observations are single CCD measurements (Scan.from_position(..., binned=False)), the noise of each is the per-CCD error of its magnitude plus a term from the median RUWE of the sky position, and the fit does not correct the reduced chi-square for binning (fit_five_parameter(..., binned=False), simulate_fit(..., binned=False)).

The map of the median RUWE is the file healpix_16_med_ruwe.npz of gaiamock's author, which this package does not ship: it carries no licence of its own. Download it from the link in the README of gaiamock (Box) and install it with `gaiamock-download healpix_16_med_ruwe.npz`, which puts it in $GAIAMOCK_DATA_DIR or ~/.gaiamock, where read_sky_map looks; or pass its path.
"""
import os
from pathlib import Path

import numpy as np

MAP_FILE = "healpix_16_med_ruwe.npz"
NSIDE = 16
PIXELS = 12 * NSIDE**2
RUWE_TO_MAS = 0.125  # gaiamock: the sky-averaged per-CCD error at G = 12, which turns a median RUWE above 1 into an error in mas
JITTER = 0.01  # gaiamock adds 0.01 mas times one standard normal draw to the error of a whole star


def ang2pix_ring(nside, ra_deg, dec_deg):
    """The HEALPix pixel in the RING scheme of positions (degrees), as healpy.ang2pix(nside, radians(90 - dec), radians(ra), nest=False) gives it: Gorski et al. 2005 (ApJ 622, 759), as in the HEALPix C code (ang2pix_ring_z_phi)."""
    z = np.sin(np.radians(np.asarray(dec_deg, dtype=float)))
    ra = np.asarray(ra_deg, dtype=float)
    z, ra = np.broadcast_arrays(z, ra)
    za = np.abs(z)
    tt = np.mod(np.radians(ra), 2 * np.pi) * 2 / np.pi  # in [0, 4)
    ncap = 2 * nside * (nside - 1)
    npix = 12 * nside**2
    # equatorial belt
    temp1, temp2 = nside * (0.5 + tt), nside * z * 0.75
    jp, jm = (temp1 - temp2).astype(np.int64), (temp1 + temp2).astype(np.int64)
    ring = nside + 1 + jp - jm
    shift = 1 - (ring & 1)
    ip = np.mod((jp + jm - nside + shift + 1) // 2, 4 * nside)
    belt = ncap + (ring - 1) * 4 * nside + ip
    # polar caps
    fraction = tt - np.floor(tt)
    radius = nside * np.sqrt(3 * (1 - za))
    jp, jm = (fraction * radius).astype(np.int64), ((1 - fraction) * radius).astype(np.int64)
    ring = jp + jm + 1
    ip = np.mod((tt * ring).astype(np.int64), 4 * ring)
    cap = np.where(z > 0, 2 * ring * (ring - 1) + ip, npix - 2 * ring * (ring + 1) + ip)
    return np.where(za <= 2 / 3, belt, cap)


def sky_map_path(path=None):
    """The path of the map: `path` when given, else the first existing file in $GAIAMOCK_DATA_DIR and ~/.gaiamock (the directories where gaiamock-download puts it)."""
    if path is not None:
        return Path(path)
    candidates = [Path(os.environ["GAIAMOCK_DATA_DIR"]).expanduser()] if os.environ.get("GAIAMOCK_DATA_DIR") else []
    candidates.append(Path.home() / ".gaiamock")
    for directory in candidates:
        if (directory / MAP_FILE).is_file():
            return directory / MAP_FILE
    raise FileNotFoundError(f"{MAP_FILE} was not found in {[str(d) for d in candidates]}. It is not shipped with wobbler: download it from the link in the README of gaiamock (https://github.com/kareemelbadry/gaiamock, section 'A modified version to predict RUWE more reliably') and run `gaiamock-download {MAP_FILE}`, or pass its path as sky_map.")


def read_sky_map(path=None):
    """The median RUWE of each of the 3,072 HEALPix pixels of level 16 (RING scheme), from the file at `path` or the one that sky_map_path finds."""
    with np.load(sky_map_path(path)) as archive:
        values = np.asarray(archive["med_ruwe"], dtype=float)
    if values.shape != (PIXELS,):
        raise ValueError(f"the map must hold {PIXELS} values (HEALPix level {NSIDE}); it holds the shape {values.shape}")
    return values


def sky_extra_noise(ra_deg, dec_deg, sky_map):
    """The term in mas that the median RUWE of the position adds to the per-CCD error: (median RUWE - 1) * 0.125."""
    return (np.asarray(sky_map)[ang2pix_ring(NSIDE, ra_deg, dec_deg)] - 1.0) * RUWE_TO_MAS


def realistic_error(ccd_error, ra_deg, dec_deg, jitter_z, sky_map):
    """The error in mas with which gaiamock draws the noise of a star's CCD measurements: the per-CCD error of its magnitude, plus the map term of its position, plus 0.01 times `jitter_z`, one standard normal draw per star. The fit weights with the per-CCD error alone."""
    return ccd_error + sky_extra_noise(ra_deg, dec_deg, sky_map) + JITTER * np.asarray(jitter_z)
