"""The scan of one star: the barycentric times, scan angles and along-scan parallax factors of Gaia's observations of it, as tensors, read from a scanning-law table (the columns of a GOST export: ObservationTimeAtBarycentre[BarycentricJulianDateInTCB], scanAngle[rad], parallaxFactorAlongScan)."""
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ._tensor import DTYPE, as_tensor, pick_device

COLUMNS = ("ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]", "scanAngle[rad]", "parallaxFactorAlongScan")
DR3_REFERENCE_JD = 2457389.0  # the reference epoch of DR3 astrometry (J2016.0): gaiamock.rescale_times_astrometry for 'dr3'
DAYS_PER_YEAR = 365.25


@dataclass(frozen=True)
class Scan:
    """The epochs of a scan as float64 tensors (one device): jd, psi (scan angle, radians) and plx_factor (along-scan parallax factor). The arrays may be NumPy arrays, lists or tensors, of any length including 0 and 1; `device` places them (default: the device of the tensors given, else the GPU when there is one, else the CPU)."""

    jd: torch.Tensor
    psi: torch.Tensor
    plx_factor: torch.Tensor
    device: object = None
    position: tuple = None
    binned: bool = True

    def __post_init__(self):
        device = pick_device(self.jd, self.psi, self.plx_factor, device=self.device)
        for name in ("jd", "psi", "plx_factor"):
            object.__setattr__(self, name, torch.atleast_1d(as_tensor(getattr(self, name), device)))
        object.__setattr__(self, "device", device)

    def __len__(self):
        return self.jd.numel()

    def to(self, device):
        """The same scan on another device."""
        return Scan(self.jd, self.psi, self.plx_factor, device=device, position=self.position, binned=self.binned)

    @classmethod
    def from_gost(cls, path, device=None, target=None, window=None):
        """The scan of the table at `path`, sorted by time. `target` selects the rows of that name in the Target column, which a GOST export of a batch of positions has (the first column of the CSV that was uploaded); `window` ("DR3" or "DR2") keeps the rows with low < jd < high of wobbler.WINDOWS[window]. With no target every row is read."""
        from .gaiafit import WINDOWS

        if window is not None and window not in WINDOWS:
            raise ValueError(f"window must be one of {sorted(WINDOWS)}, not {window!r}")
        with open(Path(path), newline="") as handle:
            reader = csv.DictReader(handle, skipinitialspace=True)
            missing = [name for name in COLUMNS if name not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(f"{path} lacks the columns {missing}; it has {reader.fieldnames}")
            records = list(reader)
        if target is not None:
            names = sorted({r["Target"] for r in records if "Target" in r})
            if target not in names:
                raise ValueError(f"{path} has no rows of the target {target!r}; its targets are {names}")
            records = [r for r in records if r["Target"] == target]
        table = np.array([(float(r[COLUMNS[0]]), float(r[COLUMNS[1]]), float(r[COLUMNS[2]])) for r in records], dtype=float).reshape(-1, 3)
        if window is not None:
            low, high = WINDOWS[window]
            table = table[(table[:, 0] > low) & (table[:, 0] < high)]
        table = table[np.argsort(table[:, 0], kind="stable")]
        return cls(jd=table[:, 0], psi=table[:, 1], plx_factor=table[:, 2], device=device)

    @classmethod
    def from_gost_batches(cls, directory, window=None, device=None):
        """{target: Scan} of every target of every batch_*.csv export in `directory` (the files that validation/gost_fetch.py writes), each sorted by time and cut to `window` as from_gost does. The directory is read once, which a scoreboard of many stars needs and from_gost on one file per star would not give."""
        from .gaiafit import WINDOWS

        if window is not None and window not in WINDOWS:
            raise ValueError(f"window must be one of {sorted(WINDOWS)}, not {window!r}")
        files = sorted(Path(directory).glob("batch_*.csv"))
        if not files:
            raise ValueError(f"no batch_*.csv in {directory}")
        rows = {}
        for path in files:
            with open(path, newline="") as handle:
                reader = csv.DictReader(handle, skipinitialspace=True)
                missing = [name for name in ("Target", *COLUMNS) if name not in (reader.fieldnames or [])]
                if missing:
                    raise ValueError(f"{path} lacks the columns {missing}; it has {reader.fieldnames}")
                for record in reader:
                    rows.setdefault(record["Target"], []).append((float(record[COLUMNS[0]]), float(record[COLUMNS[1]]), float(record[COLUMNS[2]])))
        scans = {}
        for target, values in rows.items():
            table = np.array(values, dtype=float)
            if window is not None:
                low, high = WINDOWS[window]
                table = table[(table[:, 0] > low) & (table[:, 0] < high)]
            table = table[np.argsort(table[:, 0], kind="stable")]
            scans[target] = cls(jd=table[:, 0], psi=table[:, 1], plx_factor=table[:, 2], device=device)
        return scans

    @classmethod
    def from_position(cls, ra_deg, dec_deg, window="DR3", device=None, binned=True):
        """The scan of the star at (ra_deg, dec_deg) from the precomputed scanning law of gaiamock (https://github.com/kareemelbadry/gaiamock; gaiamock.get_gost_one_position, the nearest of its 49,152 sky positions), sorted by time, with one epoch per field-of-view transit, or with `binned=False` one per CCD (gaiamock's unbinned mode, from the 3,072 sky positions of its level-16 scans; needs its file individual_ccds.zip, installed with `gaiamock-download individual_ccds.zip`). `window` is "DR3" (gaiamock's DR3 window) or "DR2" (the same epochs cut at the end of DR2's data). Needs gaiamock and its scanning-law data: `pip install gaiamock==1.0.0`, then `gaiamock-download` (GAIAMOCK_DATA_DIR sets where the data is kept)."""
        from .gaiafit import WINDOWS

        if window not in WINDOWS:
            raise ValueError(f"window must be one of {sorted(WINDOWS)}, not {window!r}")
        try:
            from gaiamock.gaiamock import fetch_table_element, get_gost_one_position
        except ImportError as error:
            raise ImportError("Scan.from_position needs gaiamock; install it with `pip install gaiamock==1.0.0`") from error
        table = get_gost_one_position(ra_deg, dec_deg, data_release="dr3", binned=binned)
        jd, psi, plx_factor = (np.asarray(fetch_table_element(name, table), dtype=float) for name in COLUMNS)
        low, high = WINDOWS[window]
        keep = (jd > low) & (jd < high)
        order = np.argsort(jd[keep], kind="stable")
        return cls(jd=jd[keep][order], psi=psi[keep][order], plx_factor=plx_factor[keep][order], device=device, position=(float(ra_deg), float(dec_deg)), binned=binned)

    def window(self, low_jd, high_jd):
        """Boolean mask of the epochs with low_jd < jd < high_jd."""
        return (self.jd > low_jd) & (self.jd < high_jd)

    def t_yr(self, reference_jd=DR3_REFERENCE_JD):
        """The times in years from the reference epoch."""
        return (self.jd - reference_jd) / DAYS_PER_YEAR
