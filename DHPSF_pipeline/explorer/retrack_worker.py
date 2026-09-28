"""Re-tracking worker: runs ``pipeline.track`` in a separate process.

Running in a child process keeps the web server responsive (no GIL contention while
frames are being rendered) and isolates the explorer from import errors while the
pipeline module is being edited.  The pipeline module is imported lazily and
re-imported when its file changes.
"""
from __future__ import annotations

import importlib
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np

PIPELINE_DIR = Path(__file__).resolve().parent.parent
TRACK_PARAMS = ("link_distance_px", "gap_distance_px", "max_frame_gap", "angle_cost_deg",
                "kalman_process_noise", "kalman_measurement_noise", "kalman_initial_velocity",
                "min_track_length")
METHODS = ("lap", "kalman", "nearest")

_P = None
_P_mtime = None


def _pipeline():
    global _P, _P_mtime
    path = PIPELINE_DIR / "pipeline.py"
    mt = path.stat().st_mtime
    if _P is None or mt != _P_mtime:
        if str(PIPELINE_DIR) not in sys.path:
            sys.path.insert(0, str(PIPELINE_DIR))
        if "pipeline" in sys.modules:
            _P = importlib.reload(sys.modules["pipeline"])
        else:
            _P = importlib.import_module("pipeline")
        _P_mtime = mt
    return _P


def defaults() -> dict:
    P = _pipeline()
    cfg = asdict(P.Config())
    out = {k: cfg[k] for k in TRACK_PARAMS if k in cfg}
    out["method"] = cfg.get("tracker", "lap")
    out["methods"] = list(METHODS)
    return out


def build_rows(P, cols: dict) -> np.ndarray:
    n = len(next(iter(cols.values())))
    rows = np.full((n, P.NCOL), np.nan)
    for name, arr in cols.items():
        if name in P.C:
            rows[:, P.C[name]] = np.asarray(arr, dtype=float)
    return rows


def run(cols: dict, params: dict) -> dict:
    """cols: {column name: 1-D array} from the loaded CSV; params: tracker settings.

    Returns {"track_number": int array (1..K), "short": bool array, "summary": {...}}.
    Tracks shorter than min_track_length are flagged in ``short`` (not removed).
    """
    t0 = time.perf_counter()
    P = _pipeline()
    method = params.get("method", "lap")
    if method not in METHODS:
        raise ValueError(f"unknown tracking method {method!r}")
    valid = {f.name for f in fields(P.Config)}
    kw = {"tracker": method}
    for k in TRACK_PARAMS:
        if k in params and params[k] is not None and k in valid:
            kw[k] = int(params[k]) if k in ("max_frame_gap", "min_track_length") else float(params[k])
    cfg = P.Config(**kw)
    rows = build_rows(P, cols)
    for need in ("xMean", "yMean", "frame_number"):
        if not np.isfinite(rows[:, P.C[need]]).all():
            raise ValueError(f"column {need} has missing values")
    if not np.isfinite(rows[:, P.C["angleDegrees"]]).all():
        rows[:, P.C["angleDegrees"]] = np.nan_to_num(rows[:, P.C["angleDegrees"]])
    out = P.track(rows, cfg)
    ids = out[:, 9].astype(np.int64)
    # relabel 1..K in order of first appearance (stable, compact)
    _, first = np.unique(ids, return_index=True)
    order = np.argsort(first)
    uniq = np.unique(ids)[order]
    remap = {int(u): i + 1 for i, u in enumerate(uniq)}
    ids = np.array([remap[int(i)] for i in ids], dtype=np.int64)
    counts = np.bincount(ids)
    lengths = counts[1:]
    min_len = int(kw.get("min_track_length", getattr(cfg, "min_track_length", 1)))
    short = counts[ids] < min_len
    frames = rows[:, P.C["frame_number"]]
    nframes = int(np.nanmax(frames) - np.nanmin(frames) + 1) if len(frames) else 0
    kept = lengths[lengths >= min_len]
    summary = {
        "method": method,
        "params": {k: getattr(cfg, k) for k in TRACK_PARAMS if hasattr(cfg, k)},
        "n_tracks": int(kept.size),
        "n_tracks_total": int(lengths.size),
        "n_full_length": int((kept >= nframes).sum()),
        "median_length": float(np.median(kept)) if kept.size else 0.0,
        "n_short_tracks": int((lengths < min_len).sum()),
        "n_short_localizations": int(short.sum()),
        "n_localizations": int(len(ids)),
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "pipeline_version": getattr(P, "VERSION", None),
    }
    return {"track_number": ids, "short": short, "summary": summary}
