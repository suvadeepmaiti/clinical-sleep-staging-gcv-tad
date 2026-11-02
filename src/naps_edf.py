"""Minimal EDF/EDF+ reader for NAPS (plain EDF, record durations 1-4 s, mixed sampling rates)."""
import hashlib
import numpy as np


def read_edf(path):
    raw = np.fromfile(path, dtype=np.uint8)
    h = raw[:256].tobytes().decode("latin-1")
    ns = int(h[252:256])
    body = raw[256:256 * (ns + 1)].tobytes().decode("latin-1")
    field = lambda off, w: [body[off * ns + i * w: off * ns + (i + 1) * w].strip() for i in range(ns)]
    labels, dims = field(0, 16), field(96, 8)
    pmin, pmax, dmin, dmax = (np.array(field(o, 8), dtype=float) for o in (104, 112, 120, 128))
    nsamp = np.array(field(216, 8), dtype=int)
    n_records, record_seconds = int(h[236:244]), float(h[244:252])
    data = raw[256 * (ns + 1):].view("<i2")
    usable = (data.size // nsamp.sum()) * nsamp.sum()  # tolerate a truncated final record
    records = data[:usable].reshape(-1, nsamp.sum())
    return {"labels": labels, "dims": dims, "pmin": pmin, "pmax": pmax, "dmin": dmin, "dmax": dmax,
            "nsamp": nsamp, "n_records": records.shape[0], "header_records": n_records,
            "record_seconds": record_seconds, "start_time": h[176:184], "start_date": h[168:176],
            "offsets": np.r_[0, np.cumsum(nsamp)], "records": records,
            "sha256": hashlib.sha256(raw).hexdigest()}


def signal(edf, i):
    """Physical samples (float64) and sampling rate of signal i."""
    lo, hi = edf["offsets"][i], edf["offsets"][i + 1]
    digital = edf["records"][:, lo:hi].reshape(-1).astype(np.float64)
    gain = (edf["pmax"][i] - edf["pmin"][i]) / (edf["dmax"][i] - edf["dmin"][i])
    return digital * gain + (edf["pmin"][i] - gain * edf["dmin"][i]), edf["nsamp"][i] / edf["record_seconds"]
