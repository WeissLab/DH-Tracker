"""DH-PSF localization / tracking explorer -- local web server.

Stdlib ``http.server`` + numpy/tifffile/PIL/scipy only.  Serves a single-page
app (``static/``) plus a small JSON/JPEG API:

    GET  /api/movies
    GET  /api/movie/<name>/info
    GET  /api/movie/<name>/frame/<i>?vmin=&vmax=&gamma=&bin=1|2|4&q=90
    GET  /api/movie/<name>/localizations
    POST /api/export            {movie, label, rows:[int], roi?:...}
    GET  /api/exports/<file>    (download a written export)
    GET  /api/runs              result sets (runs/* and results*), newest first
    POST /api/runs/select       {id}  switch the active results folder
    GET  /api/browse?path=      list a folder (sub-folders + .tif files); path=drives
    GET  /api/inspect?path=     analyze.py --inspect (cached per path + mtime)
    GET  /api/check_calibration?path=   analyze.py --check-calibration (usable z-stack? cached)
    GET  /api/check_movie?path=         analyze.py --check-movie (bright / dim, cached)
    GET  /api/recent_calibrations       analyze.py --recent-calibrations
    POST /api/analyze           start analyze.py (one job at a time)
    GET  /api/jobs/<id>         job status + log tail  (/api/jobs/current = latest job)
    POST /api/jobs/<id>/cancel

Run:
    python DHPSF_pipeline/explorer/app.py --results DHPSF_pipeline/results
then open http://127.0.0.1:8765
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as _dt
import io
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
import traceback
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
REPO = HERE.parent.parent  # the project folder (launchers, guide video)
PIPELINE_DIR = HERE.parent  # <project>/DHPSF_pipeline
sys.path.insert(0, str(PIPELINE_DIR))
import user_settings  # noqa: E402  (the user's data and results folders)
RUNS_DIR = PIPELINE_DIR / "runs"            # main(): the chosen results folder, or --runs-dir
ANALYZE_SCRIPT = PIPELINE_DIR / "analyze.py"  # overridable with --analyze-script (testing)

DEFAULT_PIXEL_SIZE_UM = 0.325
PIXEL_SIZE_UM = DEFAULT_PIXEL_SIZE_UM
REQUIRED_COLUMNS = ["x1", "x2", "xMean", "y1", "y2", "yMean", "angleDegrees",
                    "zMicrons", "frame_number", "track_number"]
OPTIONAL_COLUMNS = ["residualRMS", "minLobeSNR", "jointEmitterCount", "lobeSeparationPixels",
                    "zStatus", "recovered", "trackLength", "flag"]
MIME = {
    ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
    ".json": "application/json", ".png": "image/png", ".jpg": "image/jpeg",
    ".svg": "image/svg+xml", ".ico": "image/x-icon", ".csv": "text/csv; charset=utf-8",
    ".mat": "application/octet-stream", ".txt": "text/plain; charset=utf-8",
    ".mp4": "video/mp4", ".webm": "video/webm", ".gif": "image/gif", ".vtk": "application/octet-stream",
}


# ----------------------------------------------------------------------------
# Localization loading / export (pure functions; unit-tested)
# ----------------------------------------------------------------------------
def find_localization_file(results: Path, movie: str) -> Path | None:
    """Prefer v2 ``{movie}_localizations.csv``; fall back to v1 ``{movie}_payload.csv``."""
    for name in (f"{movie}_localizations.csv", f"{movie}_payload.csv"):
        p = Path(results) / name
        if p.is_file():
            return p
    return None


def find_drift_file(results: Path, movie: str) -> Path | None:
    p = Path(results) / f"{movie}_drift.csv"
    return p if p.is_file() else None


def find_rejected_file(results: Path, movie: str) -> Path | None:
    p = Path(results) / f"{movie}_rejected.csv"
    return p if p.is_file() else None


# -- motion analysis (track_analysis.py: aTrack-style classes, ExaTrack-style moving states) --
MOTION_CODES = {"brownian": 0, "confined": 1, "directed": 2}


def motion_files(results: Path, movie: str):
    """(track file, states file) of the motion analysis, when both exist."""
    t, s = Path(results) / f"{movie}_track_motion.csv", Path(results) / f"{movie}_motion_states.csv"
    return (t, s) if t.is_file() and s.is_file() else None


def motion_state(results: Path, movie: str) -> str:
    """'none', 'stale' (older than the localizations) or 'current'."""
    files = motion_files(results, movie)
    if not files:
        return "none"
    loc = find_localization_file(results, movie)
    if loc is not None and min(f.stat().st_mtime for f in files) < loc.stat().st_mtime:
        return "stale"
    return "current"


def attach_motion(loc: dict, results: Path, movie: str) -> None:
    """Add per-localization motion columns to a loaded localization table (in place).

    pMoving (probability of the moving state), and per track: motionClass (0 Brownian,
    1 confined, 2 directed), framesMoving, speedWhileMoving, dTrack (diffusion coefficient).
    """
    tfile, sfile = motion_files(results, movie)
    with open(tfile, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    per_track = {}
    for r in rows:
        try:
            per_track[float(r["track_number"])] = (float(MOTION_CODES.get(r["motion_class"], np.nan)),
                                                   float(r["frames_moving"]), float(r["speed_while_moving"]), float(r["D"]))
        except (KeyError, ValueError):
            continue
    st = np.genfromtxt(sfile, delimiter=",", names=True)
    st = np.atleast_1d(st)
    key = [(int(f), int(t)) for f, t in zip(st["frame_number"], st["track_number"])]
    p = dict(zip(key, st["p_moving"].astype(float)))
    # refined positions (Kalman smoother): shifts to add to the coordinates, and their SDs
    ref_names = {"refDx": "refine_dx_px", "refDy": "refine_dy_px", "refDz": "refine_dz_um",
                 "refSdX": "refined_sd_x_px", "refSdY": "refined_sd_y_px", "refSdZ": "refined_sd_z_um"}
    has_ref = all(c in (st.dtype.names or ()) for c in ref_names.values())
    has_stage = "stage" in (st.dtype.names or ())          # ordered-stages analysis (track_analysis --stages)
    row_of = {k: j for j, k in enumerate(key)}
    d = loc["data"]
    n = loc["n"]
    tid, fr = d["track_number"], d["frame_number"]
    cols = {"pMoving": np.full(n, np.nan), "motionClass": np.full(n, np.nan), "framesMoving": np.full(n, np.nan),
            "speedWhileMoving": np.full(n, np.nan), "dTrack": np.full(n, np.nan)}
    # per-localization columns copied from the states file: {explorer column: file column}
    copy = {**(ref_names if has_ref else {}), **({"stage": "stage"} if has_stage else {})}
    cols.update({c: np.full(n, np.nan) for c in copy})
    for i in range(n):
        if not (np.isfinite(tid[i]) and np.isfinite(fr[i])):
            continue
        k = (int(fr[i]), int(tid[i]))
        cols["pMoving"][i] = p.get(k, np.nan)
        if copy and k in row_of:
            for c, s in copy.items():
                cols[c][i] = st[s][row_of[k]]
        tr = per_track.get(float(tid[i]))
        if tr:
            cols["motionClass"][i], cols["framesMoving"][i], cols["speedWhileMoving"][i], cols["dTrack"][i] = tr
    for c, v in cols.items():
        d[c] = v
        if c not in loc["columns"]:
            loc["columns"].append(c)


# z convention. Results now report z as height (up +, the indenter pushes into -z; pipeline.to_lab_z).
# Folders written earlier report it along the calibration stack (the opposite sign) and carry no
# z_convention; they are flipped on loading, so every view and export shows the same convention.
# convert_z_up.py converts such a folder's files for good.
LEGACY_Z_COLUMNS = ("zMicrons", "zStabilized", "refDz")


def z_is_stack(results) -> bool:
    """True for a result folder in the old stack-index z convention (no z_convention "up" recorded)."""
    for name in ("z_convention.json", "run_info.json"):
        p = Path(results) / name
        if p.is_file():
            try:
                if json.loads(p.read_text()).get("z_convention") == "up":
                    return False
            except (OSError, ValueError):
                pass
    return True


def motion_info(state, movie: str) -> dict:
    """What the explorer shows about the motion analysis of one movie: status, summary, running job."""
    results = state.results
    s = motion_state(results, movie)
    summary = None
    p = Path(results) / "track_motion_summary.json"
    if s != "none" and p.is_file():
        try:
            summary = json.loads(p.read_text()).get(movie)
        except (OSError, ValueError):
            summary = None
    return {"state": s, "summary": summary, "job": state.motion_job(movie)}


DRIFT_Z_SIGMA = 3.0   # frames; matches pipeline Config.drift_z_smooth_frames


def smooth_series(v: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing over frames (edges held), re-referenced to its median (as pipeline.estimate_drift)."""
    v = np.asarray(v, float)
    if len(v) < 2 or sigma <= 0:
        return v.copy()
    r = int(np.ceil(3 * sigma))
    w = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    padded = np.concatenate((np.full(r, v[0]), v, np.full(r, v[-1])))
    s = np.convolve(padded, w / w.sum(), mode="valid")
    return s - np.median(s)


def load_drift(path: Path) -> dict:
    """Read ``{movie}_drift.csv`` (frame_number,dx,dy,dz; dx/dy px, dz µm; relative to the movie's median,
    or to frame 1 in folders written before rereference_drift.py).

    Returns ``{"frame_number": int array, "dx", "dy", "dz": float arrays}`` sorted by frame.
    Missing dz -> 0.  NaN shifts are treated as 0 (no correction).
    """
    loc = load_localizations(path, required=("frame_number", "dx", "dy"))
    d = loc["data"]
    fr = d["frame_number"]
    ok = np.isfinite(fr)
    order = np.argsort(fr[ok], kind="stable")
    out = {"frame_number": fr[ok][order].astype(np.int64)}
    for c in ("dx", "dy", "dz"):
        a = d[c][ok][order] if c in d else np.zeros(int(ok.sum()))
        out[c] = np.where(np.isfinite(a), a, 0.0)
    # Focus drifts smoothly: dz is the time-smoothed z drift. Pipeline versions that wrote only
    # the per-frame dz (no dzRaw column) are smoothed here the same way.
    if "dzRaw" in d:
        out["dzRaw"] = np.where(np.isfinite(d["dzRaw"][ok][order]), d["dzRaw"][ok][order], 0.0)
    elif "dz" in d:
        out["dzRaw"], out["dz"] = out["dz"], smooth_series(out["dz"], DRIFT_Z_SIGMA)
    out["path"] = str(path)
    out["n"] = int(len(out["frame_number"]))
    return out


def drift_lookup(drift: dict, frames: np.ndarray) -> tuple:
    """Per-row (dx, dy, dz) for 1-based frame numbers (0 where the drift table has no entry)."""
    lut = {int(f): i for i, f in enumerate(drift["frame_number"])}
    idx = np.array([lut.get(int(f), -1) if np.isfinite(f) else -1 for f in frames], dtype=np.int64)
    res = []
    for c in ("dx", "dy", "dz"):
        a = np.zeros(len(idx))
        m = idx >= 0
        a[m] = drift[c][idx[m]]
        res.append(a)
    return tuple(res)


STAB_COLUMNS = ("xStabilized", "yStabilized", "zStabilized")


def add_stabilized_columns(loc: dict, drift: dict | None) -> list:
    """Add xStabilized/yStabilized/zStabilized computed from drift when the CSV lacks them.

    Returns the list of columns that were added (empty if already present or no drift).
    """
    if drift is None or all(c in loc["data"] for c in STAB_COLUMNS):
        return []
    dx, dy, dz = drift_lookup(drift, loc["data"]["frame_number"])
    src = {"xStabilized": ("xMean", dx), "yStabilized": ("yMean", dy), "zStabilized": ("zMicrons", dz)}
    added = []
    for c in STAB_COLUMNS:
        if c not in loc["data"]:
            base, d = src[c]
            loc["data"][c] = loc["data"][base] - d
            loc["columns"].append(c)
            added.append(c)
    return added


def coord_settings(mode) -> dict:
    """Normalize a coordinate request: {corr, xy, z} flags, or a legacy mode string."""
    if isinstance(mode, dict):
        return {"corr": bool(mode.get("corr")), "xy": bool(mode.get("xy")), "z": bool(mode.get("z"))}
    return {"raw": {"corr": False, "xy": False, "z": False},
            "corrected": {"corr": True, "xy": False, "z": False},
            "stabilized": {"corr": True, "xy": True, "z": True}}.get(str(mode), {"corr": False, "xy": False, "z": False})


def coord_label(c: dict) -> str:
    parts = ["lateral-corrected" if c["corr"] else "raw"]
    stab = [s for s, on in (("XY", c["xy"]), ("Z", c["z"])) if on]
    if stab:
        parts.append(" + ".join(stab) + " stabilized")
    return " · ".join(parts)


def analysis_columns(loc: dict, mode, drift: dict | None, pixel_size_um: float = 1.0) -> dict:
    """xAnalysis/yAnalysis/zAnalysis (all in µm) for the explorer's coordinate settings.

    corr: xCorrected/yCorrected (fallback xMean/yMean) instead of xMean/yMean;
    xy: minus the frame's XY drift (dx, dy); z: zMicrons minus the frame's z drift (dz).
    x and y (image pixels, one-based) are then multiplied by pixel_size_um.
    Returns the columns plus the effective settings label (after fallbacks).
    """
    d = loc["data"]
    c = coord_settings(mode)
    if drift is None:
        c["xy"] = c["z"] = False
    if not ("xCorrected" in d and "yCorrected" in d):
        c["corr"] = False
    x = d["xCorrected"] if c["corr"] else d["xMean"]
    y = d["yCorrected"] if c["corr"] else d["yMean"]
    x = np.where(np.isfinite(x), x, d["xMean"])
    y = np.where(np.isfinite(y), y, d["yMean"])
    z = d["zMicrons"].copy()
    if c["xy"] or c["z"]:
        dx, dy, dz = drift_lookup(drift, d["frame_number"])
        if c["xy"]:
            x, y = x - dx, y - dy
        if c["z"]:
            z = z - dz
    return {"xAnalysis": x * pixel_size_um, "yAnalysis": y * pixel_size_um, "zAnalysis": z, "mode": coord_label(c)}


def load_localizations(path: Path, required=REQUIRED_COLUMNS) -> dict:
    """Read a localization CSV into ``{"columns": [...], "data": {col: float64 array}}``.

    Every column is parsed as float (non-numeric -> NaN).  Required columns are
    validated; optional v2 columns are passed through when present.
    """
    path = Path(path)
    with open(path, "r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        try:
            header = [h.strip() for h in next(reader)]
        except StopIteration:
            raise ValueError(f"{path.name}: empty file")
        rows = [r for r in reader if r and any(c.strip() for c in r)]
    missing = [c for c in required if c not in header]
    if missing:
        raise ValueError(f"{path.name}: missing required columns {missing}")
    ncol = len(header)
    arr = np.full((len(rows), ncol), np.nan, dtype=np.float64)
    for i, r in enumerate(rows):
        for j in range(min(ncol, len(r))):
            s = r[j].strip()
            if not s:
                continue
            try:
                arr[i, j] = float(s)
            except ValueError:
                low = s.lower()
                if low in ("true", "yes"):
                    arr[i, j] = 1.0
                elif low in ("false", "no"):
                    arr[i, j] = 0.0
                # else: stays NaN
    data = {h: arr[:, j] for j, h in enumerate(header) if h}
    columns = [h for h in header if h]
    return {"columns": columns, "data": data, "n": len(rows), "path": str(path)}


def localizations_to_json(loc: dict, decimals: int = 4) -> bytes:
    """Compact column-array JSON; NaN -> null."""
    out_cols = {}
    for c in loc["columns"]:
        a = loc["data"][c]
        fin = np.isfinite(a)
        if fin.all() and np.all(a == np.round(a)):
            out_cols[c] = [int(v) for v in a]
        else:
            r = np.round(a, decimals)
            out_cols[c] = [float(v) if f else None for v, f in zip(r.tolist(), fin.tolist())]
    payload = {"columns": loc["columns"], "n": loc["n"], "source": Path(loc["path"]).name,
               "data": out_cols}
    return json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")


_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_label(s: str, default: str = "export") -> str:
    s = _SAFE.sub("_", str(s or "")).strip("._")
    return (s or default)[:60]


def write_export(loc: dict, rows, out_dir: Path, movie: str, label: str,
                 extra: dict | None = None, timestamp: str | None = None,
                 drift: dict | None = None, overrides: dict | None = None,
                 pixel_size_um: float | None = None) -> dict:
    """Write the given row subset to ``<out_dir>/<ts>_<movie>_<label>.csv`` and ``.mat``.

    If ``drift`` is given and the table lacks xStabilized/yStabilized/zStabilized,
    those columns are computed (xMean-dx, yMean-dy, zMicrons-dz) and included.
    ``overrides`` ({column: full-length array}) replaces or appends columns, e.g. the
    track_number from an interactive re-tracking.
    """
    import scipy.io

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = np.asarray(sorted(set(int(r) for r in rows)), dtype=np.int64)
    if rows.size and (rows.min() < 0 or rows.max() >= loc["n"]):
        raise ValueError("row index out of range")
    ts = timestamp or _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{ts}_{safe_label(movie, 'movie')}_{safe_label(label)}"
    csv_path = out_dir / f"{stem}.csv"
    mat_path = out_dir / f"{stem}.mat"
    tmp = {"columns": list(loc["columns"]), "data": dict(loc["data"])}  # never mutate the cache
    add_stabilized_columns(tmp, drift)
    for c, arr in (overrides or {}).items():
        arr = np.asarray(arr, dtype=float)
        if arr.shape != (loc["n"],):
            raise ValueError(f"override {c!r} has wrong length")
        if c not in tmp["data"]:
            tmp["columns"].append(c)
        tmp["data"][c] = arr
    cols = tmp["columns"]
    sub = {c: tmp["data"][c][rows] for c in cols}
    # sort by track, then frame for convenience
    if rows.size:
        order = np.lexsort((sub["frame_number"], sub["track_number"]))
        sub = {c: v[order] for c, v in sub.items()}
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for i in range(rows.size):
            w.writerow([_fmt(sub[c][i]) for c in cols])
    mat = {c: sub[c].reshape(-1, 1) for c in cols}
    mat["columns"] = np.array(cols, dtype=object).reshape(1, -1)
    mat["source_file"] = Path(loc["path"]).name
    mat["movie"] = movie
    mat["pixel_size_um"] = pixel_size_um or PIXEL_SIZE_UM
    if extra:
        mat["export_info_json"] = json.dumps(extra)
    scipy.io.savemat(str(mat_path), mat, do_compression=True)
    return {"csv": str(csv_path), "mat": str(mat_path), "stem": stem, "rows": int(rows.size),
            "tracks": int(np.unique(sub["track_number"]).size) if rows.size else 0}


FIELD_VARS = ("ux", "uy", "uz", "areal_strain_pct", "shear_strain_pct", "tilt_deg")


def write_field_export(body: dict, out_dir: Path, timestamp: str | None = None) -> dict:
    """Deformation field on the warped median plane, all frames -> CSV (long), MAT and a VTK series.

    body: movie, label, info (dict), xs (nx), ys (ny), z0 (ny x nx reference heights, µm),
    frames (list), and per variable in FIELD_VARS a list over frames of ny x nx arrays (µm, %, °).
    """
    import scipy.io
    movie = safe_label(body.get("movie", "movie"), "movie")
    xs, ys = np.asarray(body["xs"], float), np.asarray(body["ys"], float)
    z0 = np.asarray(body["z0"], float)
    frames = np.asarray(body["frames"], int)
    nx, ny, nf = xs.size, ys.size, frames.size
    if z0.shape != (ny, nx) or nf == 0:
        raise ValueError("field export: inconsistent grid")
    F = {}
    for v in FIELD_VARS:
        a = np.asarray(body[v], float)
        if a.shape != (nf, ny, nx):
            raise ValueError(f"field export: {v} has shape {a.shape}, expected {(nf, ny, nx)}")
        F[v] = a
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = timestamp or _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{ts}_{movie}_{safe_label(body.get('label', 'deformation'))}"
    X, Y = np.meshgrid(xs, ys)
    # CSV, one row per frame and grid node
    csv_path = out_dir / f"{stem}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["frame_number", "x_um", "y_um", "z0_um", *[f"{v}_um" if v.startswith("u") else v for v in FIELD_VARS]])
        for k, f in enumerate(frames):
            cols = [F[v][k].ravel() for v in FIELD_VARS]
            for i, (x, y, z) in enumerate(zip(X.ravel(), Y.ravel(), z0.ravel())):
                w.writerow([int(f), f"{x:.3f}", f"{y:.3f}", f"{z:.3f}", *[f"{c[i]:.5g}" for c in cols]])
    # MAT: arrays frames x ny x nx
    mat_path = out_dir / f"{stem}.mat"
    scipy.io.savemat(str(mat_path), {"x_um": xs.reshape(1, -1), "y_um": ys.reshape(-1, 1), "z0_um": z0,
                                     "frame_number": frames.reshape(-1, 1), **F,
                                     "info_json": json.dumps(body.get("info", {}))}, do_compression=True)
    # VTK: one legacy structured grid per frame (the plane moved by the field); ParaView opens the
    # numbered files as a time series
    vtk_dir = out_dir / f"{stem}_vtk"
    vtk_dir.mkdir(exist_ok=True)
    for k, f in enumerate(frames):
        px, py, pz = X + F["ux"][k], Y + F["uy"][k], z0 + F["uz"][k]
        lines = ["# vtk DataFile Version 3.0", f"{movie} deformation frame {int(f)}", "ASCII", "DATASET STRUCTURED_GRID",
                 f"DIMENSIONS {nx} {ny} 1", f"POINTS {nx * ny} float"]
        lines += [f"{a:.4f} {b:.4f} {c:.4f}" for a, b, c in zip(px.ravel(), py.ravel(), pz.ravel())]
        lines.append(f"POINT_DATA {nx * ny}")
        lines.append("VECTORS displacement_um float")
        lines += [f"{a:.5g} {b:.5g} {c:.5g}" for a, b, c in zip(F["ux"][k].ravel(), F["uy"][k].ravel(), F["uz"][k].ravel())]
        for v in FIELD_VARS[3:]:
            lines += [f"SCALARS {v} float 1", "LOOKUP_TABLE default"] + [f"{a:.5g}" for a in F[v][k].ravel()]
        (vtk_dir / f"plane_{int(f):04d}.vtk").write_text("\n".join(lines) + "\n", encoding="ascii")
    (out_dir / f"{stem}_info.json").write_text(json.dumps(body.get("info", {}), indent=1), encoding="utf-8")
    return {"csv": str(csv_path), "mat": str(mat_path), "vtk_dir": str(vtk_dir), "frames": int(nf),
            "csv_url": "/api/exports/" + csv_path.name, "mat_url": "/api/exports/" + mat_path.name}


def write_animation(body: dict, out_dir: Path, timestamp: str | None = None) -> dict:
    """Save an animation made in the browser: kind 'mp4'/'webm' (base64 video) or 'gif' (PNG frames -> GIF)."""
    import base64
    import io
    movie = safe_label(body.get("movie", "movie"), "movie")
    kind = body.get("kind")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = timestamp or _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{ts}_{movie}_{safe_label(body.get('label', 'deformation'))}"
    b64 = lambda s: base64.b64decode(s.split(",", 1)[1] if s.startswith("data:") else s)
    if kind in ("mp4", "webm"):
        p = out_dir / f"{stem}.{kind}"
        p.write_bytes(b64(body["data"]))
    elif kind == "gif":
        from PIL import Image
        imgs = [Image.open(io.BytesIO(b64(s))).convert("RGB") for s in body["frames"]]
        if not imgs:
            raise ValueError("no frames")
        pal = [im.convert("P", palette=Image.Palette.ADAPTIVE, colors=256) for im in imgs]
        p = out_dir / f"{stem}.gif"
        fps = float(body.get("fps") or 10)
        pal[0].save(p, save_all=True, append_images=pal[1:], duration=int(round(1000 / fps)), loop=0, optimize=True)
    else:
        raise ValueError("kind must be mp4, webm or gif")
    return {"path": str(p), "url": "/api/exports/" + p.name, "bytes": p.stat().st_size}


def _fmt(v: float) -> str:
    if not math.isfinite(v):
        return "NaN"
    if v == int(v) and abs(v) < 1e15:
        return str(int(v))
    return repr(float(v))


# ----------------------------------------------------------------------------
# Movie access + frame rendering
# ----------------------------------------------------------------------------
class LRU:
    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        self.d: "collections.OrderedDict" = collections.OrderedDict()
        self.lock = threading.Lock()

    def get(self, k):
        with self.lock:
            if k in self.d:
                self.d.move_to_end(k)
                return self.d[k]
        return None

    def put(self, k, v):
        with self.lock:
            self.d[k] = v
            self.d.move_to_end(k)
            while len(self.d) > self.maxsize:
                self.d.popitem(last=False)


class Movie:
    def __init__(self, name: str, path: Path):
        self.name, self.path = name, Path(path)
        self._mm = None
        self._stats = None
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self.path.is_file()

    @property
    def mm(self):
        with self._lock:
            if self._mm is None:
                import tifffile
                try:
                    self._mm = tifffile.memmap(str(self.path), mode="r")
                except Exception:  # non-contiguous TIFF: fall back to page reads
                    self._mm = _PagedTiff(self.path)
            return self._mm

    def frame(self, i: int) -> np.ndarray:
        return np.asarray(self.mm[i])

    def info(self, pixel_size_um: float | None = None) -> dict:
        mm = self.mm
        shape = tuple(int(s) for s in mm.shape)
        if len(shape) == 2:
            shape = (1,) + shape
        if self._stats is None:
            mid = self.frame(shape[0] // 2).astype(np.float32)
            p = np.percentile(mid, [0.1, 1, 50, 99, 99.9, 99.99])
            self._stats = {"p0_1": float(p[0]), "p1": float(p[1]), "p50": float(p[2]),
                           "p99": float(p[3]), "p99_9": float(p[4]), "p99_99": float(p[5]),
                           "max": float(mid.max()), "min": float(mid.min())}
        s = self._stats
        return {"name": self.name, "path": str(self.path), "frames": shape[0],
                "height": shape[1], "width": shape[2], "dtype": str(mm.dtype),
                "pixel_size_um": pixel_size_um or PIXEL_SIZE_UM, "stats": s,
                "default_vmin": s["p1"], "default_vmax": s["p99_9"], "default_gamma": 1.0}


class _PagedTiff:
    def __init__(self, path):
        import tifffile
        self.tf = tifffile.TiffFile(str(path))
        p0 = self.tf.pages[0]
        self.shape = (len(self.tf.pages),) + tuple(p0.shape)
        self.dtype = p0.dtype
        self.lock = threading.Lock()

    def __getitem__(self, i):
        with self.lock:
            return self.tf.pages[int(i)].asarray()


def render_frame(frame: np.ndarray, vmin: float, vmax: float, gamma: float,
                 binning: int = 1, quality: int = 90, fmt: str = "jpeg",
                 shift: tuple | None = None, cval: float = 0.0) -> bytes:
    """Contrast-map a frame to 8-bit JPEG/PNG.

    ``shift=(dx, dy)`` (pixels) is the estimated drift of this frame; the image is
    resampled so that content at x appears at x-dx (i.e. stabilized coordinates).
    """
    from PIL import Image

    f = frame.astype(np.float32)
    if shift is not None and (abs(shift[0]) > 1e-3 or abs(shift[1]) > 1e-3):
        from scipy import ndimage
        # output[y, x] = input[y + dy, x + dx]
        f = ndimage.shift(f, (-float(shift[1]), -float(shift[0])), order=1, mode="constant",
                          cval=float(cval), prefilter=False)
    if binning > 1:
        h, w = f.shape
        h2, w2 = h // binning, w // binning
        f = f[:h2 * binning, :w2 * binning].reshape(h2, binning, w2, binning).mean(axis=(1, 3))
    span = max(float(vmax) - float(vmin), 1e-6)
    x = np.clip((f - float(vmin)) / span, 0.0, 1.0)
    if abs(gamma - 1.0) > 1e-6:
        x = np.power(x, float(gamma))
    img = (x * 255.0 + 0.5).astype(np.uint8)
    buf = io.BytesIO()
    if fmt == "png":
        Image.fromarray(img).save(buf, "PNG", compress_level=1)
    else:
        Image.fromarray(img).save(buf, "JPEG", quality=int(quality))
    return buf.getvalue()


# ----------------------------------------------------------------------------
# Application state
# ----------------------------------------------------------------------------
class State:
    def __init__(self, results: Path, movies: dict, run_id: str | None = None, pixel_size_um: float | None = None):
        self._loc_lock = threading.Lock()
        self._pool = None
        self._pool_lock = threading.Lock()
        self.motion_jobs = {}             # (results, movie) -> on-demand motion analysis job
        self.switch(results, movies, run_id, pixel_size_um)

    def switch(self, results: Path, movies: dict, run_id: str | None = None, pixel_size_um: float | None = None):
        """Point the explorer at another results folder: new movies, empty caches, no re-tracking."""
        with self._loc_lock:
            self.pixel_size_um = pixel_size_um or DEFAULT_PIXEL_SIZE_UM
            self.results = Path(results)
            self.run_id = run_id or str(self.results)
            self.movies = {k: Movie(k, v) for k, v in movies.items()}
            self.frame_cache = LRU(400)
            self._loc_cache = {}
            self.retracks = {}            # movie -> {track_number, short, summary, loc_path, loc_mtime}

    def _run_worker(self, fn_name: str, *args, timeout: float = 600):
        """Run retrack_worker.<fn_name>(*args) in a persistent child process."""
        from concurrent.futures import ProcessPoolExecutor
        from concurrent.futures.process import BrokenProcessPool
        import retrack_worker
        for attempt in range(2):
            with self._pool_lock:
                if self._pool is None:
                    self._pool = ProcessPoolExecutor(max_workers=1)
                pool = self._pool
            try:
                return pool.submit(getattr(retrack_worker, fn_name), *args).result(timeout=timeout)
            except BrokenProcessPool:
                with self._pool_lock:
                    self._pool = None
                if attempt:
                    raise

    def tracking_defaults(self) -> dict:
        return self._run_worker("defaults", timeout=120)

    def retrack(self, movie: str, params: dict) -> dict:
        loc = self.loc(movie)
        if loc is None:
            raise KeyError(f"no localizations for {movie}")
        cols = {c: loc["data"][c] for c in ("x1", "x2", "xMean", "y1", "y2", "yMean", "angleDegrees",
                                             "zMicrons", "frame_number") if c in loc["data"]}
        res = self._run_worker("run", cols, params)
        self.retracks[movie] = {**res, "loc_path": loc["path"], "loc_mtime": loc["mtime"]}
        return res

    def current_retrack(self, movie: str):
        r = self.retracks.get(movie)
        loc = self.loc(movie)
        if r is None or loc is None or r["loc_path"] != loc["path"] or r["loc_mtime"] != loc["mtime"]:
            return None
        return r

    def _cached(self, key, path: Path | None, loader):
        if path is None:
            return None
        mt = path.stat().st_mtime
        with self._loc_lock:
            c = self._loc_cache.get(key)
            if c and c["mtime"] == mt and c["path"] == str(path):
                return c
            obj = loader(path)
            obj["mtime"] = mt
            obj["path"] = str(path)
            self._loc_cache[key] = obj
            return obj

    @staticmethod
    def _load_loc(path, motion=None, flip_z=False):
        loc = load_localizations(path)
        if motion is not None:
            try:
                attach_motion(loc, *motion)
            except Exception as e:   # a broken motion file must not hide the localizations
                print(f"motion results not attached: {e}", file=sys.stderr)
        if flip_z:                    # result folder from before z was reported as height (see z_is_stack)
            for c in LEGACY_Z_COLUMNS:
                if c in loc["data"]:
                    loc["data"][c] = -loc["data"][c]
            loc["z_converted"] = True
        loc["json"] = localizations_to_json(loc)
        return loc

    def loc(self, movie: str) -> dict | None:
        # the motion results (when current) are part of the table: key the cache on their state
        files = motion_files(self.results, movie) if motion_state(self.results, movie) == "current" else None
        sig = tuple(f.stat().st_mtime for f in files) if files else None
        flip = z_is_stack(self.results)
        return self._cached(("loc", movie, sig, flip), find_localization_file(self.results, movie),
                            lambda p: self._load_loc(p, (self.results, movie) if files else None, flip))

    # on-demand motion analysis (older runs): track_analysis.py in a separate process
    def motion_job(self, movie: str) -> dict:
        return dict(self.motion_jobs.get((str(self.results), movie), {"state": "none"}))

    def start_motion(self, movie: str, frame_interval_ms=None, stages=None) -> dict:
        key = (str(self.results), movie)
        job = self.motion_jobs.get(key)
        if job and job.get("state") == "running":
            return dict(job)
        cmd = [sys.executable, str(PIPELINE_DIR / "track_analysis.py"), str(self.results), "--movie", movie]
        if frame_interval_ms:
            cmd += ["--frame-interval", repr(float(frame_interval_ms))]
        if stages:
            cmd += ["--stages", stages]
        job = {"state": "running", "started": time.time(), "message": "analysing motion (about 10–40 seconds)"}
        self.motion_jobs[key] = job

        def run():
            try:
                cp = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, cwd=str(PIPELINE_DIR),
                                    creationflags=_NO_WINDOW)
                if cp.returncode == 0:
                    job.update(state="done", message=(cp.stdout.strip().splitlines() or ["done"])[-1])
                else:
                    job.update(state="error", message=" | ".join((cp.stderr or cp.stdout).strip().splitlines()[-3:]))
            except Exception as e:
                job.update(state="error", message=str(e))
        threading.Thread(target=run, daemon=True).start()
        return dict(job)

    def rejected(self, movie: str) -> dict | None:
        flip = z_is_stack(self.results)
        return self._cached(("rej", movie, flip), find_rejected_file(self.results, movie),
                            lambda p: self._load_loc(p, None, flip))

    def drift(self, movie: str) -> dict | None:
        flip = z_is_stack(self.results)

        def _load(p):
            d = load_drift(p)
            if flip:
                for c in ("dz", "dzRaw"):
                    if c in d:
                        d[c] = -d[c]
            d["json"] = json.dumps({"frame_number": d["frame_number"].tolist(),
                                    "dx": np.round(d["dx"], 4).tolist(), "dy": np.round(d["dy"], 4).tolist(),
                                    "dz": np.round(d["dz"], 4).tolist(),
                                    **({"dzRaw": np.round(d["dzRaw"], 4).tolist()} if "dzRaw" in d else {}),
                                    "source": Path(p).name},
                                   separators=(",", ":")).encode()
            d["map"] = {int(f): (float(x), float(y)) for f, x, y in zip(d["frame_number"], d["dx"], d["dy"])}
            return d
        return self._cached(("drift", movie, flip), find_drift_file(self.results, movie), _load)


STATE: State | None = None


# ----------------------------------------------------------------------------
# Result sets ("runs"), file browsing, TIFF inspection, analysis jobs
# ----------------------------------------------------------------------------
class HttpError(Exception):
    def __init__(self, code: int, msg: str):
        super().__init__(msg)
        self.code, self.msg = code, msg


TIFF_EXT = (".tif", ".tiff")
OBJECTIVE_PIXEL_UM = {"20x": 0.325, "10x": 0.63}
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def read_json_retry(path: Path, tries: int = 4):
    """Read a JSON file that another process may be replacing; None if missing/unreadable.

    Note (Windows): while any process has the file open -- even for os.stat -- a concurrent
    ``os.replace`` onto it fails with PermissionError, so the writer (analyze.py) must retry
    its atomic replace.  We keep reads short and rare (one per 2 s poll)."""
    path = Path(path)
    for i in range(tries):
        try:
            with open(path, "rb") as fh:
                data = fh.read()
            return json.loads(data.decode("utf-8-sig"))
        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, PermissionError, OSError):
            time.sleep(0.05 * (i + 1))
    return None


def write_json_atomic(path: Path, obj: dict):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    for i in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:  # Windows: reader holds the file for a moment
            time.sleep(0.05 * (i + 1))
    os.replace(tmp, path)


def _loc_movie_names(d: Path) -> list:
    names = []
    for suffix in ("_localizations.csv", "_payload.csv"):
        for p in sorted(d.glob("*" + suffix)):
            n = p.name[: -len(suffix)]
            if n not in names:
                names.append(n)
    return names


def has_results(d: Path) -> bool:
    d = Path(d)
    return (d / "run_info.json").is_file() or bool(_loc_movie_names(d))


def run_id_for(path: Path, base: Path | None = None, runs_dir: Path | None = None) -> str:
    """'runs/<name>' for folders in the runs dir, '<name>' for base/results*, else the absolute path."""
    base = Path(base or PIPELINE_DIR).resolve()
    runs_dir = Path(runs_dir or RUNS_DIR).resolve()
    p = Path(path).resolve()
    if p.parent == runs_dir:
        return "runs/" + p.name
    if p.parent == base and p.name.startswith("results"):
        return p.name
    return str(p)


def resolve_run_id(run_id: str, base: Path | None = None, runs_dir: Path | None = None) -> Path:
    """Inverse of run_id_for; rejects anything outside the runs dir / base/results*."""
    base = Path(base or PIPELINE_DIR)
    runs_dir = Path(runs_dir or RUNS_DIR)
    rid = str(run_id or "").replace("\\", "/")
    if rid.startswith("runs/"):
        name = rid[5:]
        p = runs_dir / name
    else:
        name = rid
        p = base / name
        if not name.startswith("results"):
            raise HttpError(404, f"unknown run {run_id!r}")
    if not name or "/" in name or name in (".", "..") or not p.is_dir():
        raise HttpError(404, f"unknown run {run_id!r}")
    return p


def run_entry(run_id: str, d: Path) -> dict:
    d = Path(d)
    info = read_json_retry(d / "run_info.json") if (d / "run_info.json").is_file() else None
    status = read_json_retry(d / "status.json") if (d / "status.json").is_file() else None
    loc_names = _loc_movie_names(d)
    movies = list((info or {}).get("movies", {}).keys()) or loc_names
    if (d / "run_info.json").is_file():
        mtime = (d / "run_info.json").stat().st_mtime
    elif (d / "status.json").is_file():
        mtime = (d / "status.json").stat().st_mtime
    elif loc_names:
        mtime = max(p.stat().st_mtime for p in d.glob("*.csv"))
    else:
        mtime = d.stat().st_mtime
    return {"id": run_id, "label": d.name, "path": str(d), "mtime": mtime,
            "date": _dt.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M"),
            "movies": movies, "pixel_size_um": (info or {}).get("pixel_size_um"),
            "state": (status or {}).get("state"), "complete": bool(loc_names)}


def list_runs(base: Path | None = None, runs_dir: Path | None = None) -> list:
    """Every folder in runs_dir plus base/results* folders holding results, newest first."""
    base = Path(base or PIPELINE_DIR)
    runs_dir = Path(runs_dir or RUNS_DIR)
    out = []
    if runs_dir.is_dir():
        for d in runs_dir.iterdir():
            if d.is_dir():
                out.append(run_entry("runs/" + d.name, d))
    for d in base.glob("results*"):
        if d.is_dir() and has_results(d):
            out.append(run_entry(d.name, d))
    out.sort(key=lambda r: r["mtime"], reverse=True)
    return out


def newest_complete_run(runs_dir: Path | None = None) -> Path | None:
    runs_dir = Path(runs_dir or RUNS_DIR)
    if not runs_dir.is_dir():
        return None
    best = None
    for d in runs_dir.iterdir():
        if not (d.is_dir() and (d / "run_info.json").is_file() and _loc_movie_names(d)):
            continue
        st = read_json_retry(d / "status.json") if (d / "status.json").is_file() else None
        if st and st.get("state") not in (None, "done"):
            continue
        mt = (d / "run_info.json").stat().st_mtime
        if best is None or mt > best[0]:
            best = (mt, d)
    return best[1] if best else None


def run_movies(results: Path, overrides: dict | None = None, add_new: bool = False):
    """(movies, pixel_size_um) for a results folder: from run_info.json; without it (older folders),
    the movies it holds results for, whose TIFF paths are then unknown (give them with --movie)."""
    try:
        run_m, pixel = movies_from_run_info(Path(results))
    except (ValueError, OSError):
        run_m, pixel = None, None
    movies = dict(run_m) if run_m else {n: Path(results) / f"{n}.tif" for n in _loc_movie_names(Path(results))}
    for k, v in (overrides or {}).items():
        if add_new or k in movies:
            movies[k] = Path(v)
    return movies, (float(pixel) if pixel else DEFAULT_PIXEL_SIZE_UM)


# -- browsing ----------------------------------------------------------------
def list_drives() -> list:
    if os.name == "nt":
        import ctypes
        mask = ctypes.windll.kernel32.GetLogicalDrives()
        return [f"{chr(65 + i)}:\\" for i in range(26) if (mask >> i) & 1]
    return ["/"]


_SKIP_DIRS = {"system volume information", "$recycle.bin", "recovery", "config.msi", "__pycache__"}


def settings_info() -> dict:
    """The user's folders (user_settings.py) and the results folder in use."""
    legacy = user_settings.LEGACY_RESULTS
    return {"data_dir": str(user_settings.data_dir() or ""), "results_dir": str(RUNS_DIR),
            "settings_file": str(user_settings.settings_path()),
            # results made before a results folder was chosen stay in the program folder
            "legacy_results": str(legacy) if legacy.resolve() != RUNS_DIR.resolve() and legacy.is_dir()
            and any(legacy.iterdir()) else ""}


GITHUB_REPO = "WeissLab/DH-Tracker"
_VERSION_CACHE: dict = {}


def version_info(now: float | None = None, fetch=None) -> dict:
    """Installed version (version.txt, written by install.ps1) and the newest one on GitHub (asked at most
    every 6 hours, 5 s timeout; None offline). Copies not made by the installer have no version.txt and
    are not checked."""
    vf = REPO / "version.txt"
    installed = vf.read_text(encoding="ascii", errors="ignore").strip() if vf.is_file() else None
    out = {"installed": installed, "latest": None, "update_available": False, "repo": GITHUB_REPO}
    if not installed:
        return out
    now = time.time() if now is None else now
    if _VERSION_CACHE.get("t", -1e9) + 6 * 3600 < now:
        def ask():
            import urllib.request
            req = urllib.request.Request(f"https://api.github.com/repos/{GITHUB_REPO}/commits/main",
                                         headers={"Accept": "application/vnd.github.sha", "User-Agent": "DH-Tracker-2026"})
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.read(64).decode("ascii", "ignore").strip()
        try:
            latest = (fetch or ask)()
            latest = latest if re.fullmatch(r"[0-9a-f]{40}", latest or "") else None
        except Exception:
            latest = None
        _VERSION_CACHE.update(t=now if latest else now - 5.5 * 3600, latest=latest)   # offline: ask again in 30 min
    out["latest"] = _VERSION_CACHE.get("latest")
    out["update_available"] = bool(out["latest"]) and out["latest"] != installed
    return out


_DIALOG_LOCK = threading.Lock()


def choose_folder(which: str) -> dict:
    """Open a folder dialog on this computer (the server runs here) and save the choice."""
    global RUNS_DIR
    if which not in ("data", "results"):
        raise HttpError(400, "which must be 'data' or 'results'")
    if which == "results" and JOBS.running() is not None:
        raise HttpError(409, "an analysis is running; change the results folder when it has finished")
    if not _DIALOG_LOCK.acquire(blocking=False):
        raise HttpError(409, "a folder dialog is already open")
    try:
        start = user_settings.data_dir() if which == "data" else RUNS_DIR
        p = user_settings.ask_folder("DH-Tracker-2026: " + ("folder with your movies (TIFF files)" if which == "data"
                                                            else "where to save results"), start)
    finally:
        _DIALOG_LOCK.release()
    if p is None:
        return dict(settings_info(), changed=False)
    if which == "data":
        user_settings.save(data_dir=p)
    else:
        p.mkdir(parents=True, exist_ok=True)
        user_settings.save(results_dir=p)
        RUNS_DIR = p.resolve()
        JOBS.runs_dir = RUNS_DIR
    return dict(settings_info(), changed=True)


def browse_dir(path: str | None, default: Path | None = None) -> dict:
    """List sub-folders and .tif/.tiff files of a folder (read-only).  ``path='drives'`` lists
    the drive roots.  Hidden/system folders are left out."""
    if path == "drives":
        return {"path": "drives", "parent": None, "dirs": list_drives(), "files": []}
    p = Path(path) if path else Path(default or REPO)
    if not p.is_absolute():
        raise HttpError(400, f"not an absolute path: {path}")
    p = Path(os.path.abspath(p))
    if not p.is_dir():
        raise HttpError(404, f"folder not found: {p}")
    dirs, files = [], []
    try:
        with os.scandir(p) as it:
            for e in it:
                name = e.name
                if name.startswith(("$", ".")) or name.lower() in _SKIP_DIRS:
                    continue
                try:
                    if e.is_dir():
                        attrs = getattr(e.stat(), "st_file_attributes", 0)
                        if attrs & 0x6:  # FILE_ATTRIBUTE_HIDDEN | FILE_ATTRIBUTE_SYSTEM
                            continue
                        dirs.append(name)
                    elif e.is_file() and name.lower().endswith(TIFF_EXT):
                        st = e.stat()
                        files.append({"name": name, "size": st.st_size, "mtime": st.st_mtime})
                except OSError:
                    continue
    except PermissionError:
        raise HttpError(403, f"permission denied: {p}")
    dirs.sort(key=str.lower)
    files.sort(key=lambda f: f["name"].lower())
    parent = str(p.parent) if p.parent != p else ("drives" if os.name == "nt" else None)
    return {"path": str(p), "parent": parent, "dirs": dirs, "files": files}


# -- inspection --------------------------------------------------------------
def guess_objective(path) -> str | None:
    """'10x' / '20x' from the file name, else from the nearest folder name."""
    p = Path(path)
    for part in [p.name] + [q.name for q in p.parents]:
        m = re.search(r"(?i)(?<![0-9a-z])(10|20)x(?![a-z0-9])", part)
        if m:
            return m.group(1) + "x"
    return None


def fallback_inspect(path) -> dict:
    """Basic TIFF facts with tifffile (used when analyze.py --inspect is unavailable)."""
    import tifffile
    p = Path(path)
    roi = None
    with tifffile.TiffFile(str(p)) as tf:
        s = tf.series[0]
        shape = tuple(int(v) for v in s.shape)
        dtype = str(s.dtype)
        try:
            mm = tf.micromanager_metadata or {}
            r = (mm.get("Summary") or {}).get("ROI")
            if isinstance(r, (list, tuple)) and len(r) == 4:
                roi = [int(v) for v in r]
        except Exception:
            roi = None
    frames = int(np.prod(shape[:-2])) if len(shape) > 2 else 1
    obj = guess_objective(p)
    return {"path": str(p), "frames": frames, "height": shape[-2], "width": shape[-1], "dtype": dtype,
            "objective_guess": obj, "pixel_size_guess_um": OBJECTIVE_PIXEL_UM.get(obj),
            "micromanager_roi": roi}


def default_run_name(movie_path, now: _dt.datetime | None = None) -> str:
    """First movie's stem (without .ome / _MMStack_PosN) + _YYYYMMDD-HHMM."""
    stem = Path(movie_path).name
    for suf in (".tiff", ".tif", ".ome"):
        if stem.lower().endswith(suf):
            stem = stem[: -len(suf)]
    stem = re.sub(r"(?i)_MMStack_Pos\d+$", "", stem)
    now = now or _dt.datetime.now()
    return f"{safe_label(stem, 'run')}_{now:%Y%m%d-%H%M}"


def _finite(obj):
    """NaN / inf -> None (our JSON responses are strict)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_finite(v) for v in obj]
    return obj


def _parse_json_stdout(text: str):
    try:
        return _finite(json.loads(text))
    except json.JSONDecodeError:
        for line in reversed(text.strip().splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return _finite(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return None


def tiff_path(path: str) -> Path:
    """Validate a user-supplied path to an existing .tif/.tiff file."""
    if not path:
        raise HttpError(400, "path required")
    p = Path(path)
    if not p.is_absolute():
        raise HttpError(400, f"not an absolute path: {path}")
    if p.suffix.lower() not in TIFF_EXT:
        raise HttpError(400, f"not a .tif file: {path}")
    if not p.is_file():
        raise HttpError(404, f"file not found: {path}")
    return p


class FilePreviews:
    """Read-only frames of arbitrary TIFF files (calibration planes, example movie frames) for the
    New analysis window.  Contrast defaults to the 1st / 99.9th percentile of the middle frame, so
    all planes of a stack share one scale."""

    def __init__(self):
        self.clear()

    def clear(self):
        """Forget open files (their memory maps close once garbage-collected)."""
        self.files = LRU(6)
        self.frames = LRU(300)

    def movie(self, path: str) -> Movie:
        p = tiff_path(path)
        st = p.stat()
        key = (str(p.resolve()), st.st_mtime, st.st_size)
        m = self.files.get(key)
        if m is None:
            m = Movie(p.stem, p)
            self.files.put(key, m)
        return m

    def info(self, path: str) -> dict:
        info = dict(self.movie(path).info())
        info.pop("pixel_size_um", None)
        return info

    def frame(self, path: str, q: dict) -> bytes:
        m = self.movie(path)
        info = m.info()
        n = int(q.get("frame", 1))                      # 1-based, like analyze.py --frame
        if not 1 <= n <= info["frames"]:
            raise HttpError(404, f"frame {n} out of range 1..{info['frames']}")
        vmin = float(q.get("vmin", info["default_vmin"]))
        vmax = float(q.get("vmax", info["default_vmax"]))
        gamma = float(q.get("gamma", 1.0))
        binning = int(q.get("bin", 1))
        binning = binning if binning in (1, 2, 4, 8) else 1
        key = (m.path, n, round(vmin, 3), round(vmax, 3), round(gamma, 3), binning)
        body = self.frames.get(key)
        if body is None:
            body = render_frame(m.frame(n - 1), vmin, vmax, gamma, binning, 88)
            self.frames.put(key, body)
        return body


class Inspector:
    """Runs ``analyze.py --inspect PATH`` (cached per path + mtime + size); falls back to
    :func:`fallback_inspect` when the script is missing or fails.  Also runs
    ``analyze.py --preview PATH --frame N`` (bead detections on one frame, cached)."""

    def __init__(self, script: Path | None = None, python: str = sys.executable, timeout: float = 60,
                 preview_timeout: float = 180):
        self.script, self.python, self.timeout = script, python, timeout
        self.preview_timeout = preview_timeout
        self.cache: dict = {}
        self.previews: dict = {}
        self.lock = threading.Lock()

    def _engine(self, args: list, timeout: float, what: str):
        """(parsed JSON dict or None, warning or None)."""
        script = Path(self.script or ANALYZE_SCRIPT)
        if not script.is_file():
            return None, f"{script.name} not found"
        try:
            cp = subprocess.run([self.python, str(script)] + args, capture_output=True, text=True,
                                timeout=timeout, cwd=str(script.parent), creationflags=_NO_WINDOW)
        except subprocess.TimeoutExpired:
            return None, f"analyze.py {what} timed out after {timeout:.0f} s"
        if cp.returncode != 0:
            tail = (cp.stderr or cp.stdout or "").strip().splitlines()[-3:]
            return None, f"analyze.py {what} failed (exit {cp.returncode}): " + " | ".join(tail)
        res = _parse_json_stdout(cp.stdout)
        if not isinstance(res, dict):
            return None, f"analyze.py {what} printed no JSON"
        return res, None

    def preview(self, path: str, frame: int) -> dict:
        p = tiff_path(path)
        frame = int(frame)
        if frame < 1:
            raise HttpError(400, "frame is 1-based")
        st = p.stat()
        key = (str(p.resolve()), st.st_mtime, st.st_size, frame)
        with self.lock:
            if key in self.previews:
                return self.previews[key]
        res, warn = self._engine(["--preview", str(p), "--frame", str(frame)], self.preview_timeout, "--preview")
        if res is None:
            raise HttpError(502, warn)
        with self.lock:
            self.previews[key] = res
            while len(self.previews) > 64:
                self.previews.pop(next(iter(self.previews)))
        return res

    def check_calibration(self, path: str) -> dict:
        """``analyze.py --check-calibration``: is this a usable calibration z-stack? (cached)"""
        p = tiff_path(path)
        st = p.stat()
        key = ("check", str(p.resolve()), st.st_mtime, st.st_size)
        with self.lock:
            if key in self.previews:
                return self.previews[key]
        res, warn = self._engine(["--check-calibration", str(p)], self.preview_timeout, "--check-calibration")
        if res is None:
            raise HttpError(502, warn)
        with self.lock:
            self.previews[key] = res
        return res

    def check_movie(self, path: str) -> dict:
        """``analyze.py --check-movie``: quick brightness check (bright / dim) of a movie (cached)."""
        p = tiff_path(path)
        st = p.stat()
        key = ("movie", str(p.resolve()), st.st_mtime, st.st_size)
        with self.lock:
            if key in self.previews:
                return self.previews[key]
        res, warn = self._engine(["--check-movie", str(p)], self.preview_timeout, "--check-movie")
        if res is None:
            raise HttpError(502, warn)
        with self.lock:
            self.previews[key] = res
        return res

    def recent_calibrations(self) -> dict:
        """``analyze.py --recent-calibrations``: calibration stacks used before, with their cache state."""
        res, warn = self._engine(["--recent-calibrations"], self.timeout, "--recent-calibrations")
        return res if res is not None else {"calibrations": [], "warning": warn}

    def __call__(self, path: str) -> dict:
        if not path:
            raise HttpError(400, "path required")
        p = Path(path)
        if not p.is_file():
            raise HttpError(404, f"file not found: {path}")
        st = p.stat()
        key = (str(p.resolve()), st.st_mtime, st.st_size)
        with self.lock:
            if key in self.cache:
                return dict(self.cache[key])
        res, warn = self._engine(["--inspect", str(p)], self.timeout, "--inspect")
        if res is None:
            try:
                res = fallback_inspect(p)
            except Exception as e:
                raise HttpError(422, f"cannot read {p.name} as TIFF: {e}")
            res["source"] = "fallback"
            res["warning"] = warn
        else:
            res["source"] = "analyze.py"
            with self.lock:
                self.cache[key] = dict(res)
        return res


# -- analysis jobs -----------------------------------------------------------
class JobBusy(Exception):
    pass


def _unique_dir(p: Path) -> Path:
    if not p.exists():
        return p
    for i in range(2, 1000):
        q = p.with_name(f"{p.name}_{i}")
        if not q.exists():
            return q
    raise HttpError(409, f"cannot find a free folder name for {p}")


def log_tail(path: Path, n: int = 15, max_bytes: int = 65536) -> list:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            start = max(0, size - max_bytes)
            fh.seek(start)
            text = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    if start == 0 and text.startswith("# "):   # the command line header written by JobManager
        text = text.split("\n", 1)[1] if "\n" in text else ""
    lines = [ln.rstrip() for ln in re.split(r"[\r\n]+", text)]
    return [ln for ln in lines if ln.strip()][-n:]


# -- README as a web page (the ☰ menu's Help section and F1) -----------------------------------------
def _md_inline(s: str) -> str:
    import html as _h
    codes = []                                        # code spans out of the way first (bold may wrap them)

    def keep(m):
        codes.append(f"<code>{_h.escape(m.group(1))}</code>")
        return f"\x00{len(codes)-1}\x00"
    p = _h.escape(re.sub(r"`([^`]+)`", keep, s), quote=False)
    p = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", r'<a href="\2" target="_blank">\1</a>', p)
    p = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", p)
    p = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", p)
    return re.sub(r"\x00(\d+)\x00", lambda m: codes[int(m.group(1))], p)


def markdown_to_html(md: str) -> str:
    """Small Markdown renderer for the READMEs: headings, paragraphs, nested lists, tables, code blocks,
    bold / italic / inline code / links (the subset they use; no extra package needed)."""
    import html as _h
    lines, out, i = md.splitlines(), [], 0
    list_stack = []                                   # (indent, tag)

    def close_lists(to_indent=-1):
        while list_stack and list_stack[-1][0] > to_indent:
            out.append(f"</li></{list_stack.pop()[1]}>")
    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if s.startswith("```"):
            close_lists()
            code, i = [], i + 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i]); i += 1
            ind = min((len(c) - len(c.lstrip()) for c in code if c.strip()), default=0)
            out.append("<pre><code>" + _h.escape("\n".join(c[ind:] for c in code)) + "</code></pre>")
            i += 1; continue
        if not s:
            i += 1; continue
        m = re.match(r"(#{1,6})\s+(.*)", s)
        if m and not line.startswith(" "):
            close_lists()
            n = len(m.group(1))
            slug = re.sub(r"[^a-z0-9]+", "-", m.group(2).lower()).strip("-")
            out.append(f'<h{n} id="{slug}">{_md_inline(m.group(2))}</h{n}>')
            i += 1; continue
        if s.startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|?\s*:?-{2,}", lines[i + 1]):
            close_lists()
            cells = lambda r: [c.strip() for c in r.strip().strip("|").split("|")]
            head, i = cells(s), i + 2
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(cells(lines[i])); i += 1
            thead = "" if not any(head) else \
                "<thead><tr>" + "".join(f"<th>{_md_inline(c)}</th>" for c in head) + "</tr></thead>"
            out.append("<table>" + thead + "<tbody>"
                       + "".join("<tr>" + "".join(f"<td>{_md_inline(c)}</td>" for c in r) + "</tr>" for r in rows)
                       + "</tbody></table>")
            continue
        m = re.match(r"(\s*)([-*]|\d+\.)\s+(.*)", line)
        if m:
            ind, tag = len(m.group(1)), ("ol" if m.group(2)[0].isdigit() else "ul")
            if list_stack and ind == list_stack[-1][0]:
                out.append("</li><li>")
            elif not list_stack or ind > list_stack[-1][0]:
                out.append(f"<{tag}><li>"); list_stack.append((ind, tag))
            else:
                close_lists(ind)
                out.append("</li><li>" if list_stack else f"<{tag}><li>")
                if not list_stack:
                    list_stack.append((ind, tag))
            out.append(_md_inline(m.group(3)))
            i += 1; continue
        if list_stack and line.startswith(" "):      # continuation of a list item
            out.append(" " + _md_inline(s)); i += 1; continue
        close_lists()
        para = [s]; i += 1
        while i < len(lines) and lines[i].strip() and not re.match(r"\s*([-*]|\d+\.|#|\||```)", lines[i]):
            para.append(lines[i].strip()); i += 1
        out.append("<p>" + _md_inline(" ".join(para)) + "</p>")
    close_lists()
    return "\n".join(out)


README_PAGES = {"/readme": ("DH-PSF bead tracking: README", PIPELINE_DIR / "README.md"),
                "/readme/explorer": ("DH-Tracker-2026 explorer: README", HERE / "README.md")}


def readme_page(path: str) -> bytes:
    import html as _h
    title, src = README_PAGES[path]
    body = markdown_to_html(src.read_text(encoding="utf-8")) if src.is_file() else f"<p>{_h.escape(str(src))} not found.</p>"
    nav = ('<nav><a href="https://www.weisslab.ca" target="_blank">www.WeissLab.ca</a> · '
           '<a href="/readme">Program README</a> · <a href="/readme/explorer">Explorer README</a>'
           + (' · <a href="/guide.mp4" target="_blank">Video guide</a>' if GUIDE_VIDEO.is_file() else "") + "</nav>")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{_h.escape(title)}</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>
:root{{color-scheme:light dark;--bg:#fbfbfc;--fg:#1d232b;--dim:#5e6875;--line:#dde2e8;--code:#f0f2f5;--acc:#1d6788}}
@media (prefers-color-scheme:dark){{:root{{--bg:#12151a;--fg:#dde2e8;--dim:#98a2ae;--line:#2a313b;--code:#1c2129;--acc:#74b6d8}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 system-ui,Segoe UI,sans-serif}}
main{{max-width:900px;margin:0 auto;padding:18px 22px 60px}} nav{{font-size:13.5px;color:var(--dim);padding:10px 0;border-bottom:1px solid var(--line)}}
a{{color:var(--acc)}} h1{{font-size:26px;margin:22px 0 8px}} h2{{font-size:20px;margin:30px 0 8px;padding-top:8px;border-top:1px solid var(--line)}}
h3{{font-size:16.5px;margin:20px 0 6px}} code{{background:var(--code);padding:1px 5px;border-radius:4px;font-size:.9em}}
pre{{background:var(--code);padding:10px 12px;border-radius:6px;overflow-x:auto}} pre code{{padding:0;background:none}}
table{{border-collapse:collapse;margin:10px 0;font-size:14px}} th,td{{border:1px solid var(--line);padding:5px 9px;vertical-align:top;text-align:left}}
li{{margin:2px 0}}</style></head><body><main>{nav}{body}</main></body></html>""".encode("utf-8")


GUIDE_VIDEO = REPO / "DH-Tracker-2026 guide.mp4"


def process_start_time(pid: int) -> float | None:
    """Start time (epoch s) of a running process, or None if it is not running. Recorded at launch
    and compared later, so a process number reused by another program is never mistaken for a job."""
    if not pid:
        return None
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except OSError:
            return None
        return 0.0
    import ctypes
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(0x1000, False, int(pid))          # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return None
    try:
        code = ctypes.c_ulong()
        if not k32.GetExitCodeProcess(h, ctypes.byref(code)) or code.value != 259:   # STILL_ACTIVE
            return None
        t = [ctypes.c_ulonglong() for _ in range(4)]      # creation, exit, kernel, user (FILETIME)
        if not k32.GetProcessTimes(h, *[ctypes.byref(v) for v in t]):
            return None
        return t[0].value / 1e7 - 11644473600.0
    finally:
        k32.CloseHandle(h)


class Job:
    def __init__(self, job_id, cmd, output: Path, run_id: str, name: str):
        self.id, self.cmd, self.output, self.run_id, self.name = job_id, cmd, Path(output), run_id, name
        self.status_path = self.output / "status.json"
        self.log_path = self.output / "analyze.log"
        self.job_path = self.output / "job.json"      # process id, so a reopened explorer can find the job
        self.proc: subprocess.Popen | None = None
        self.pid: int | None = None                   # set for jobs started by an earlier explorer session
        self.pid_started: float | None = None
        self.log_fh = None
        self.started = time.time()
        self.ended: float | None = None
        self.cancelled = False
        self.finalized = False
        self.lock = threading.Lock()

    @property
    def alive(self) -> bool:
        if self.proc is not None:
            return self.proc.poll() is None
        if self.pid:                                  # adopted: the same process (start time) still running
            t = process_start_time(self.pid)
            return t is not None and (self.pid_started is None or abs(t - self.pid_started) < 2.0)
        return False


class JobManager:
    """Launches the analysis engine as a child process, one job at a time.

    ``command`` is the argv prefix (default ``[sys.executable, analyze.py]``); the manager
    appends ``--calibration ... --movie ... --output ... --status-file ...``.  Tests pass a
    tiny fake script instead of the real engine.
    """

    def __init__(self, runs_dir: Path | None = None, command: list | None = None):
        self.runs_dir = Path(runs_dir or RUNS_DIR)
        self.command = list(command) if command else None
        self.jobs: dict = {}
        self.lock = threading.RLock()   # re-entrant: start() holds it while running() -> adopt() takes it too

    def _command(self) -> list:
        return self.command or [sys.executable, str(ANALYZE_SCRIPT)]

    def running(self) -> Job | None:
        self.adopt()
        for j in list(self.jobs.values()):
            if j.alive:
                return j
        return None

    @staticmethod
    def validate(body: dict) -> dict:
        def num(key, required=True, default=None):
            v = body.get(key, default)
            if v in (None, ""):
                if required:
                    raise ValueError(f"{key} is required")
                return None
            try:
                v = float(v)
            except (TypeError, ValueError):
                raise ValueError(f"{key} must be a number")
            if not (math.isfinite(v) and v > 0):
                raise ValueError(f"{key} must be > 0")
            return v

        cal = str(body.get("calibration") or "").strip()
        if not cal:
            raise ValueError("calibration file is required")
        if not Path(cal).is_file():
            raise ValueError(f"calibration file not found: {cal}")
        movies = [str(m).strip() for m in (body.get("movies") or []) if str(m).strip()]
        if not movies:
            raise ValueError("at least one movie is required")
        for m in movies:
            if not Path(m).is_file():
                raise ValueError(f"movie not found: {m}")
        name = str(body.get("name") or "").strip()
        planes = body.get("calibration_planes")
        if planes in (None, "", []):
            planes = None
        else:
            if isinstance(planes, str):
                planes = planes.split(":")
            try:
                a, b = (int(v) for v in planes)
            except (TypeError, ValueError):
                raise ValueError("calibration_planes must be [first, last]")
            if not 1 <= a <= b:
                raise ValueError("calibration_planes must satisfy 1 <= first <= last")
            planes = (a, b)
        return {"calibration": cal, "movies": movies,
                "pixel_size_um": num("pixel_size_um"),
                # None: analyze.py derives it (stage speed in the name × frame interval), or stops if it cannot
                "z_step_um": num("z_step_um", required=False),
                "z_range_um": num("z_range_um", required=False),
                "calibration_planes": planes,
                "min_snr": num("min_snr", required=False),
                "name": safe_label(name, "run")[:80] if name else None,
                "matlab": bool(body.get("matlab", True)),
                "true_depth": bool(body.get("true_depth", False)),
                "frame_interval_ms": num("frame_interval_ms", required=False)}

    def start(self, body: dict) -> dict:
        p = self.validate(body)
        with self.lock:
            cur = self.running()
            if cur is not None:
                raise JobBusy(f"an analysis is already running ({cur.name})")
            name = p["name"] or default_run_name(p["movies"][0])
            out = _unique_dir(self.runs_dir / name)
            out.mkdir(parents=True)
            name = out.name
            cmd = self._command() + ["--calibration", p["calibration"]]
            for m in p["movies"]:
                cmd += ["--movie", m]
            cmd += ["--output", str(out), "--status-file", str(out / "status.json"),
                    "--pixel-size", repr(p["pixel_size_um"]), "--name", name]
            if p["z_step_um"] is not None:
                cmd += ["--z-step", repr(p["z_step_um"])]
            if p["z_range_um"] is not None:
                cmd += ["--z-range", repr(p["z_range_um"])]
            if p["calibration_planes"] is not None:
                cmd += ["--calibration-planes", "%d:%d" % p["calibration_planes"]]
            if p["min_snr"] is not None:
                cmd += ["--min-snr", repr(p["min_snr"])]
            if not p["matlab"]:
                cmd.append("--no-matlab")
            if p["true_depth"]:
                cmd.append("--true-depth")
            if p["frame_interval_ms"] is not None:
                cmd += ["--frame-interval", repr(p["frame_interval_ms"])]
            job = Job(uuid.uuid4().hex[:12], cmd, out, "runs/" + name, name)
            now = _dt.datetime.now().isoformat(timespec="seconds")
            write_json_atomic(job.status_path, {"state": "running", "stage": "starting", "progress": 0.0,
                                                "message": "launching analysis", "started": now,
                                                "updated": now, "output": str(out), "error": None})
            job.log_fh = open(job.log_path, "w", encoding="utf-8")
            job.log_fh.write("# " + subprocess.list2cmdline(cmd) + "\n")
            job.log_fh.flush()
            env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
            try:
                job.proc = subprocess.Popen(cmd, stdout=job.log_fh, stderr=subprocess.STDOUT,
                                            stdin=subprocess.DEVNULL, cwd=str(PIPELINE_DIR), env=env,
                                            creationflags=_NO_WINDOW)
            except OSError as e:
                job.log_fh.close()
                write_json_atomic(job.status_path, {"state": "error", "stage": "launch", "progress": 0.0,
                                                    "message": "could not start", "started": now,
                                                    "updated": now, "output": str(out), "error": str(e)})
                raise
            self.jobs[job.id] = job
            write_json_atomic(job.job_path, {"job_id": job.id, "pid": job.proc.pid,
                                             "pid_started": process_start_time(job.proc.pid),
                                             "name": name, "run_id": job.run_id, "output": str(out), "cmd": cmd})
            threading.Thread(target=self._watch, args=(job,), daemon=True).start()
        return {"job_id": job.id, "output": str(out), "run_id": job.run_id, "name": name}

    # -- jobs started by an earlier explorer session (the analysis keeps running when the explorer closes)
    _adopt_checked = 0.0

    def adopt(self, force: bool = False):
        """Find analyses still marked running in runs/: re-attach to those whose process is alive (progress,
        Cancel), and mark the others interrupted (their process ended without finishing, e.g. a restart)."""
        if not force and time.time() - self._adopt_checked < 5:
            return
        self._adopt_checked = time.time()
        if not self.runs_dir.is_dir():
            return
        known = {str(j.output) for j in list(self.jobs.values())}
        for d in self.runs_dir.iterdir():
            jp, sp = d / "job.json", d / "status.json"
            if str(d) in known or not (jp.is_file() and sp.is_file()):
                continue
            st = read_json_retry(sp) or {}
            if st.get("state") not in (None, "running"):
                continue
            info = read_json_retry(jp) or {}
            job = Job(info.get("job_id") or uuid.uuid4().hex[:12], info.get("cmd") or [], d,
                      info.get("run_id") or "runs/" + d.name, info.get("name") or d.name)
            job.pid, job.pid_started = info.get("pid"), info.get("pid_started")
            try:
                job.started = _dt.datetime.fromisoformat(st.get("started")).timestamp()
            except (TypeError, ValueError):
                pass
            with self.lock:
                self.jobs[job.id] = job
            if job.alive:
                threading.Thread(target=self._watch_adopted, args=(job,), daemon=True).start()
            else:
                self._finalize(job)

    def _watch_adopted(self, job: Job):
        while job.alive:
            time.sleep(2)
        self._finalize(job)

    def _watch(self, job: Job):
        try:
            job.proc.wait()
        finally:
            self._finalize(job)

    def _finalize(self, job: Job):
        """Once the process has exited: close the log and make status.json final."""
        with job.lock:
            if job.finalized or job.alive:
                return
            job.finalized = True
            job.ended = job.ended or time.time()
            try:
                if job.log_fh:
                    job.log_fh.close()
            except OSError:
                pass
            st = read_json_retry(job.status_path) or {}
            state = st.get("state")
            now = _dt.datetime.now().isoformat(timespec="seconds")
            rc = job.proc.returncode if job.proc is not None else None
            if job.cancelled:
                st.update(state="cancelled", message="cancelled by user", updated=now,
                          error=st.get("error"))
            elif state in ("done", "error"):
                return
            elif rc == 0 and (job.output / "run_info.json").is_file():
                st.update(state="done", progress=1.0, message=st.get("message") or "finished", updated=now)
            elif job.proc is None:   # adopted job whose process is gone without writing a final state
                st.update(state="error", updated=now, message="interrupted: the analysis stopped before finishing "
                          "(computer restarted or process ended)", error="\n".join(log_tail(job.log_path, 15)) or None)
            else:
                tail = log_tail(job.log_path, 15)
                st.update(state="error", updated=now,
                          message=f"analysis process exited (code {rc}) before finishing",
                          error="\n".join(tail) or f"exit code {rc}")
            st.setdefault("output", str(job.output))
            write_json_atomic(job.status_path, st)

    def get(self, job_id: str) -> Job:
        job = self.jobs.get(job_id)
        if job is None:
            raise HttpError(404, f"unknown job {job_id!r}")
        return job

    def status(self, job_id: str) -> dict:
        job = self.get(job_id)
        if not job.alive:
            self._finalize(job)
        st = read_json_retry(job.status_path) or {}
        alive = job.alive
        if alive and st.get("state") in (None, "done", "error", "cancelled"):
            st["state"] = "running"   # engine wrote a final state but is still shutting down
        out = {"state": "running", "stage": "", "progress": 0.0, "message": "", "error": None}
        out.update(st)
        out.update({"job_id": job.id, "name": job.name, "run_id": job.run_id, "output": str(job.output),
                    "pid": job.proc.pid if job.proc else job.pid, "alive": alive,
                    "elapsed_s": round((job.ended or time.time()) - job.started, 1),
                    "log_tail": log_tail(job.log_path, 15)})
        try:
            out["progress"] = float(out.get("progress") or 0.0)
        except (TypeError, ValueError):
            out["progress"] = 0.0
        return out

    def current(self) -> dict | None:
        self.adopt()
        if not self.jobs:
            return None
        job = max(self.jobs.values(), key=lambda j: j.started)
        return self.status(job.id)

    def cancel(self, job_id: str) -> dict:
        job = self.get(job_id)
        job.cancelled = True
        if job.proc is None:                           # adopted from an earlier explorer session
            if job.alive:
                kill_pid_tree(job.pid)
            for _ in range(30):
                if not job.alive:
                    break
                time.sleep(0.5)
            self._finalize(job)
            return self.status(job_id)
        if job.alive:
            kill_tree(job.proc)
        try:
            job.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            job.proc.kill()
            job.proc.wait(timeout=5)
        self._finalize(job)
        return self.status(job_id)


def kill_tree(proc: subprocess.Popen):
    if os.name == "nt":
        kill_pid_tree(proc.pid)
    else:
        proc.kill()


def kill_pid_tree(pid: int):
    """Stop a process and everything it started (the pipeline and its worker processes)."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True,
                       creationflags=_NO_WINDOW)
    else:
        try:
            os.kill(pid, 9)
        except OSError:
            pass


JOBS = JobManager()
INSPECT = Inspector()
PREVIEWS = FilePreviews()


LOOPBACK = ("127.0.0.1", "::1", "localhost")
LOOPBACK_NAMES = ("127.0.0.1", "localhost", "[::1]")


class Handler(BaseHTTPRequestHandler):
    server_version = "DHPSFExplorer/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quieter log: skip frame requests
        msg = fmt % args
        if "/frame/" in msg and '" 200' in msg:
            return
        sys.stderr.write("[explorer] %s\n" % msg)

    # -- request guard --
    # The explorer reads folders and TIFF files anywhere on this computer and starts analyses, so it
    # answers only its own pages. Requests must name it by a loopback address, which blocks DNS
    # rebinding (another site's domain pointed at 127.0.0.1). API calls must not come from another
    # site's page: a form or <img> on a web page can send requests to 127.0.0.1:8765, and a UNC path
    # (\\server\share) in one would make Windows try to sign in to that server with the user's account.
    def _refused(self, api: bool) -> str | None:
        host = (self.headers.get("Host") or "").strip().lower()
        if self.server.server_address[0] in LOOPBACK and host.rsplit(":", 1)[0] not in LOOPBACK_NAMES:
            return "unknown host"
        if api:
            site = self.headers.get("Sec-Fetch-Site")
            if site and site not in ("same-origin", "none"):
                return "request from another site"
            origin = self.headers.get("Origin")
            if origin and origin.lower() != f"http://{host}":
                return "request from another site"
        return None

    # -- helpers --
    def _send(self, code, body: bytes, ctype: str, extra: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")          # not embeddable in another site's page
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, allow_nan=False).encode(), "application/json",
                   {"Cache-Control": "no-store"})

    def _err(self, code, msg):
        self._json({"error": msg}, code)

    def _movie(self, name):
        m = self.st.movies.get(name)
        if m is None:
            raise KeyError(f"unknown movie {name!r}")
        return m

    # -- routes --
    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        try:
            u = urlparse(self.path)
            path = unquote(u.path)
            q = {k: v[-1] for k, v in parse_qs(u.query).items()}
            why = self._refused(path.startswith("/api/"))
            if why:
                return self._err(403, f"refused: {why}")
            if path.startswith("/api/"):
                self.st = state_for(q.get("run"))
                return self._api_get(path, q)
            if path in README_PAGES:
                return self._send(200, readme_page(path), "text/html; charset=utf-8", {"Cache-Control": "no-cache"})
            if path == "/guide.mp4" and GUIDE_VIDEO.is_file():
                return self._send(200, GUIDE_VIDEO.read_bytes(), "video/mp4", {"Cache-Control": "no-cache"})
            return self._static(path)
        except HttpError as e:
            self._err(e.code, e.msg)
        except KeyError as e:
            self._err(404, str(e))
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as e:
            traceback.print_exc()
            self._err(500, f"{type(e).__name__}: {e}")

    def _static(self, path):
        if path in ("", "/"):
            path = "/index.html"
        target = (STATIC / path.lstrip("/")).resolve()
        if STATIC.resolve() not in target.parents or not target.is_file():
            return self._err(404, "not found")
        body = target.read_bytes()
        ctype = MIME.get(target.suffix.lower(), "application/octet-stream")
        cache = "max-age=86400" if "vendor" in target.parts else "no-cache"
        self._send(200, body, ctype, {"Cache-Control": cache})

    def _api_get(self, path, q):
        parts = [p for p in path.split("/") if p][1:]  # drop 'api'
        if parts == ["movies"]:
            out = []
            for name, m in self.st.movies.items():
                lp = find_localization_file(self.st.results, name)
                dp = find_drift_file(self.st.results, name)
                rp = find_rejected_file(self.st.results, name)
                out.append({"name": name, "path": str(m.path), "available": m.available,
                            "localizations": lp.name if lp else None,
                            "drift": dp.name if dp else None, "rejected": rp.name if rp else None})
            return self._json({"movies": out, "results": str(self.st.results), "run": self.st.run_id,
                               "pixel_size_um": self.st.pixel_size_um})
        if parts == ["runs"]:
            runs = list_runs()
            if not any(r["id"] == STATE.run_id for r in runs) and STATE.results.is_dir():
                runs.insert(0, run_entry(STATE.run_id, STATE.results))   # custom --results folder
            return self._json({"runs": runs, "active": STATE.run_id})
        if parts == ["browse"]:
            return self._json(browse_dir(q.get("path") or None, user_settings.data_dir()))
        if parts == ["settings"]:
            return self._json(settings_info())
        if parts == ["version"]:
            return self._json(version_info())
        if parts == ["inspect"]:
            res = INSPECT(q.get("path", ""))
            res["default_run_name"] = default_run_name(q.get("path", ""))
            return self._json(res)
        if parts == ["file_info"]:
            return self._json(PREVIEWS.info(q.get("path", "")))
        if parts == ["preview_frame"]:
            body = PREVIEWS.frame(q.get("path", ""), q)
            return self._send(200, body, "image/jpeg", {"Cache-Control": "private, max-age=600"})
        if parts == ["check_calibration"]:
            return self._json(INSPECT.check_calibration(q.get("path", "")))
        if parts == ["check_movie"]:
            return self._json(INSPECT.check_movie(q.get("path", "")))
        if parts == ["recent_calibrations"]:
            return self._json(INSPECT.recent_calibrations())
        if parts == ["preview"]:
            try:
                frame = int(q.get("frame", ""))
            except ValueError:
                raise HttpError(400, "frame (1-based) required")
            return self._json(INSPECT.preview(q.get("path", ""), frame))
        if parts == ["jobs", "current"]:
            return self._json({"job": JOBS.current()})
        if len(parts) == 2 and parts[0] == "jobs":
            return self._json(JOBS.status(parts[1]))
        if parts == ["tracking", "defaults"]:
            return self._json(self.st.tracking_defaults())
        if len(parts) >= 3 and parts[0] == "movie":
            m = self._movie(parts[1])
            if parts[2] == "info":
                if not m.available:
                    return self._err(404, f"movie file not found: {m.path}")
                info = m.info(self.st.pixel_size_um)
                lp = find_localization_file(self.st.results, m.name)
                dp = find_drift_file(self.st.results, m.name)
                rp = find_rejected_file(self.st.results, m.name)
                info["localizations"] = lp.name if lp else None
                info["drift"] = dp.name if dp else None
                info["rejected"] = rp.name if rp else None
                info["motion"] = motion_info(self.st, m.name)
                return self._json(info)
            if parts[2] == "motion":
                return self._json(motion_info(self.st, m.name))
            if parts[2] == "frame" and len(parts) == 4:
                return self._frame(m, int(parts[3]), q)
            if parts[2] in ("localizations", "rejected"):
                loc = self.st.loc(m.name) if parts[2] == "localizations" else self.st.rejected(m.name)
                if loc is None:
                    return self._err(404, f"no {parts[2]} CSV for {m.name} in {self.st.results}")
                return self._send(200, loc["json"], "application/json", {"Cache-Control": "no-store"})
            if parts[2] == "drift":
                d = self.st.drift(m.name)
                if d is None:
                    return self._err(404, f"no drift CSV for {m.name} in {self.st.results}")
                return self._send(200, d["json"], "application/json", {"Cache-Control": "no-store"})
        if len(parts) == 2 and parts[0] == "exports":
            name = parts[1]
            if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", name):
                return self._err(404, "not found")
            p = (self.st.results / "exports" / name)
            if p.parent.resolve() != (self.st.results / "exports").resolve() or not p.is_file():
                return self._err(404, "not found")
            return self._send(200, p.read_bytes(), MIME.get(p.suffix, "application/octet-stream"),
                              {"Content-Disposition": f'attachment; filename="{name}"'})
        return self._err(404, "unknown endpoint")

    def _frame(self, m: Movie, i: int, q: dict):
        info = m.info()
        if not (0 <= i < info["frames"]):
            return self._err(404, "frame out of range")
        vmin = float(q.get("vmin", info["default_vmin"]))
        vmax = float(q.get("vmax", info["default_vmax"]))
        gamma = float(q.get("gamma", 1.0))
        binning = int(q.get("bin", 1))
        if binning not in (1, 2, 4):
            binning = 1
        quality = min(max(int(q.get("q", 90)), 30), 100)
        fmt = "png" if q.get("fmt") == "png" else "jpeg"
        shift = None
        if q.get("stab") in ("1", "true"):
            d = self.st.drift(m.name)
            if d is not None:
                shift = d["map"].get(i + 1, (0.0, 0.0))
        key = (m.name, i, round(vmin, 3), round(vmax, 3), round(gamma, 4), binning, quality, fmt,
               None if shift is None else (round(shift[0], 4), round(shift[1], 4)))
        body = self.st.frame_cache.get(key)
        if body is None:
            body = render_frame(m.frame(i), vmin, vmax, gamma, binning, quality, fmt,
                                shift=shift, cval=info["stats"]["p50"])
            self.st.frame_cache.put(key, body)
        self._send(200, body, "image/png" if fmt == "png" else "image/jpeg",
                   {"Cache-Control": "private, max-age=3600"})

    def do_POST(self):
        try:
            u = urlparse(self.path)
            why = self._refused(True)
            if why:
                self.close_connection = True           # the body is not read
                return self._err(403, f"refused: {why}")
            n = int(self.headers.get("Content-Length") or 0)
            if n > 200_000_000:     # animation frames can be large
                return self._err(413, "payload too large")
            body = json.loads(self.rfile.read(n) or b"{}")
            self.st = state_for({k: v[-1] for k, v in parse_qs(u.query).items()}.get("run"))
            if u.path == "/api/export":
                return self._export(body)
            if u.path == "/api/export_roi":
                return self._export_roi(body)
            if u.path == "/api/export_field":
                return self._json(write_field_export(body, self.st.results / "exports"))
            if u.path == "/api/export_animation":
                return self._json(write_animation(body, self.st.results / "exports"))
            if u.path == "/api/runs/select":
                return self._select_run(body)
            if u.path == "/api/open_exports":
                # this run's exports folder in the computer's file browser (the server runs on this computer)
                out = self.st.results / "exports"
                out.mkdir(parents=True, exist_ok=True)
                if os.name == "nt":
                    os.startfile(str(out))
                else:
                    subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(out)])
                return self._json({"folder": str(out)})
            if u.path == "/api/settings/choose":
                return self._json(choose_folder(str(body.get("which", ""))))
            if u.path == "/api/analyze":
                try:
                    return self._json(JOBS.start(body))
                except JobBusy as e:
                    return self._err(409, str(e))
            parts = [p for p in unquote(u.path).split("/") if p]
            if len(parts) == 4 and parts[:2] == ["api", "movie"] and parts[3] == "retrack":
                return self._retrack(parts[2], body)
            if len(parts) == 4 and parts[:2] == ["api", "movie"] and parts[3] == "motion":
                m = self._movie(parts[2])
                if find_localization_file(self.st.results, m.name) is None:
                    return self._err(404, f"no localizations for {m.name}")
                fi = body.get("frame_interval_ms")
                fi = float(fi) if fi not in (None, "") else None
                if fi is not None and not (math.isfinite(fi) and fi > 0):
                    return self._err(400, "frame_interval_ms must be > 0")
                stages = body.get("stages")
                if stages not in (None, "", "indentation"):
                    return self._err(400, "stages must be empty or 'indentation'")
                self.st.start_motion(m.name, fi, stages or None)
                return self._json(motion_info(self.st, m.name))
            if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "cancel":
                return self._json(JOBS.cancel(parts[2]))
            return self._err(404, "unknown endpoint")
        except HttpError as e:
            self._err(e.code, e.msg)
        except (ValueError, KeyError) as e:
            self._err(400, f"{type(e).__name__}: {e}")
        except Exception as e:
            traceback.print_exc()
            self._err(500, f"{type(e).__name__}: {e}")

    def _export(self, body):
        movie = body["movie"]
        self._movie(movie)
        loc = self.st.loc(movie)
        if loc is None:
            return self._err(404, "no localizations")
        rows = body.get("rows")
        if rows is None and "tracks" in body:
            tr = np.asarray(body["tracks"], dtype=float)
            rows = np.nonzero(np.isin(loc["data"]["track_number"], tr))[0].tolist()
        if not rows:
            return self._err(400, "nothing to export (0 rows)")
        extra = {k: body[k] for k in ("filters", "selection", "rois", "label", "note") if k in body}
        extra["source"] = Path(loc["path"]).name
        extra["z_convention"] = "up: z is height, positive toward the indenter (which comes from above)" + \
            (" (converted from the folder's stack-index z on loading)" if loc.get("z_converted") else "")
        drift = self.st.drift(movie)
        if drift is not None:
            extra["drift_source"] = Path(drift["path"]).name
        overrides = None
        if body.get("tracking") == "retrack":
            rt = self.st.current_retrack(movie)
            if rt is None:
                return self._err(409, "re-tracking result is stale or missing; run it again")
            overrides = {"pipelineTrackNumber": loc["data"]["track_number"],
                         "track_number": rt["track_number"].astype(float),
                         "shortTrack": rt["short"].astype(float)}
            extra["tracking"] = rt["summary"]
        else:
            extra["tracking"] = "pipeline"
        mode = body.get("coordinates", "raw")
        px = self.st.pixel_size_um or PIXEL_SIZE_UM
        ana = analysis_columns(loc, mode, drift, px)
        extra["coordinates"] = ana.pop("mode")
        extra["analysis_units"] = f"xAnalysis, yAnalysis, zAnalysis in µm (x, y: image pixels × {px} µm/px)"
        overrides = {**(overrides or {}), **ana}
        res = write_export(loc, rows, self.st.results / "exports", movie, body.get("label", "export"), extra,
                           drift=drift, overrides=overrides, pixel_size_um=self.st.pixel_size_um)
        res["csv_url"] = "/api/exports/" + Path(res["csv"]).name
        res["mat_url"] = "/api/exports/" + Path(res["mat"]).name
        return self._json(res)

    def _retrack(self, movie, body):
        self._movie(movie)
        if body.get("method") == "reset":
            self.st.retracks.pop(movie, None)
            return self._json({"reset": True})
        res = self.st.retrack(movie, body)
        return self._json({"track_number": res["track_number"].tolist(),
                           "short": res["short"].astype(int).tolist(),
                           "summary": res["summary"]})

    def _select_run(self, body):
        rid = str(body.get("id") or "")
        if rid == STATE.run_id:
            path = STATE.results
        else:
            path = resolve_run_id(rid)
        if not _loc_movie_names(path):
            raise HttpError(409, f"{path.name} has no localization results yet")
        select_results(path)
        return self._json({"run": STATE.run_id, "results": str(STATE.results), "pixel_size_um": PIXEL_SIZE_UM,
                           "movies": list(STATE.movies)})

    def _export_roi(self, body):
        movie = safe_label(body.get("movie", "movie"))
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = self.st.results / "exports"
        out.mkdir(parents=True, exist_ok=True)
        p = out / f"{ts}_{movie}_rois.json"
        p.write_text(json.dumps(body, indent=1), encoding="utf-8")
        return self._json({"json": str(p), "url": "/api/exports/" + p.name})


def resolve_results(arg: str, runs_dir: Path | None = None) -> Path:
    """'auto' -> the newest complete run in runs/ (run_info.json + localizations, not failed),
    else the (empty) runs folder itself, e.g. on a new installation before the first analysis.
    Relative paths are tried against the CWD first, then against the repository root."""
    if arg == "auto":
        newest = newest_complete_run(runs_dir)
        if newest is not None:
            return newest
        runs = Path(runs_dir or RUNS_DIR)
        runs.mkdir(parents=True, exist_ok=True)
        return runs
    p = Path(arg)
    if p.is_absolute():
        return p
    for base in (Path.cwd(), REPO):
        if (base / p).is_dir():
            return (base / p).resolve()
    return (Path.cwd() / p).resolve()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="DH-PSF localization explorer")
    ap.add_argument("--results", default="auto",
                    help="result folder to show first; 'auto' (default) = the newest finished run in "
                         "DHPSF_pipeline/runs")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1",
                    help="address to listen on; anything but 127.0.0.1 lets other computers on the network "
                         "browse this computer's folders and start analyses (no password)")
    ap.add_argument("--movie", action="append", default=[], metavar="NAME=PATH",
                    help="add/override a movie (repeatable)")
    ap.add_argument("--runs-dir", default=None,
                    help="folder for new analysis runs (default: the results folder chosen at setup / in ☰ › Folders)")
    ap.add_argument("--analyze-script", default=None,
                    help="analysis engine script (default DHPSF_pipeline/analyze.py; for testing)")
    ap.add_argument("--open-browser", action="store_true",
                    help="open the explorer in the default web browser once the server is up")
    return ap.parse_args(argv)


def movies_from_run_info(results: Path):
    """Movie paths and pixel size written by pipeline.py (results/run_info.json), if present."""
    info_path = results / "run_info.json"
    if not info_path.is_file():
        return None, None
    info = json.loads(info_path.read_text())
    movies = {name: Path(m["path"]) for name, m in info.get("movies", {}).items()}
    return movies, info.get("pixel_size_um")


MOVIE_OVERRIDES: dict = {}   # --movie NAME=PATH (re-applied to runs that contain NAME)


def select_results(path: Path, initial: bool = False):
    """Make ``path`` the active results folder (movies + pixel size from its run_info.json)."""
    global STATE, PIXEL_SIZE_UM
    path = Path(path)
    movies, pixel = run_movies(path, MOVIE_OVERRIDES, add_new=initial)
    PIXEL_SIZE_UM = pixel
    rid = run_id_for(path)
    if STATE is None:
        STATE = State(path, movies, rid, pixel)
    else:
        STATE.switch(path, movies, rid, pixel)
    return STATE


# Each page names its run in every request (?run=...), so several tabs or windows can show different
# runs at once; STATE (set by /api/runs/select) is only the default, e.g. for a newly opened page.
OTHER_STATES: dict = {}          # run id -> State, most recently used last
_OTHER_LOCK = threading.Lock()
MAX_OTHER_STATES = 4


def state_for(run_id: str | None) -> State:
    if not run_id or run_id == STATE.run_id:
        return STATE
    with _OTHER_LOCK:
        st = OTHER_STATES.pop(run_id, None)
        if st is None:
            path = resolve_run_id(run_id)
            if not _loc_movie_names(path):
                raise HttpError(409, f"{path.name} has no localization results yet")
            movies, pixel = run_movies(path, MOVIE_OVERRIDES)
            st = State(path, movies, run_id_for(path), pixel)
        OTHER_STATES[run_id] = st
        while len(OTHER_STATES) > MAX_OTHER_STATES:
            old = OTHER_STATES.pop(next(iter(OTHER_STATES)))
            if old._pool is not None:
                old._pool.shutdown(wait=False, cancel_futures=True)
        return st


def main(argv=None):
    global STATE, RUNS_DIR, ANALYZE_SCRIPT
    a = parse_args(argv)
    RUNS_DIR = Path(a.runs_dir or user_settings.results_dir()).resolve()
    if a.analyze_script:
        ANALYZE_SCRIPT = Path(a.analyze_script).resolve()
    JOBS.runs_dir = RUNS_DIR
    results = resolve_results(a.results, RUNS_DIR)
    for spec in a.movie:
        k, _, v = spec.partition("=")
        MOVIE_OVERRIDES[k.strip()] = Path(v.strip())
    select_results(results, initial=True)
    if a.host not in ("127.0.0.1", "localhost", "::1"):
        print("warning: serving on a non-loopback host", file=sys.stderr)
    if a.host not in LOOPBACK:
        print(f"WARNING: listening on {a.host}: anyone who can reach this computer on port {a.port} can browse "
              "its folders, read TIFF files and start analyses; there is no password.", flush=True)
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    srv.daemon_threads = True
    print(f"DH-Tracker-2026 explorer: http://{a.host}:{a.port}  (results={results})", flush=True)
    if a.open_browser:
        import webbrowser
        threading.Timer(0.5, webbrowser.open, args=(f"http://{a.host}:{a.port}",)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        if STATE._pool is not None:
            STATE._pool.shutdown(cancel_futures=True)


if __name__ == "__main__":
    main()
