"""Classical double-helix PSF localization, empirical calibration, and LAP tracking (v2).

Per frame: DoG peaks with noise-adaptive strong/weak thresholds; every peak inside a
crowded region is fitted jointly as a free circular Gaussian; fitted lobes are then
paired by a global maximum-weight matching scored against the calibrated
separation-versus-angle curve. Across frames: two-stage LAP tracking, then
track-guided recovery (a missed bead is refitted where its own track and the local
sample motion predict it), re-tracking, and rejection of transient tracks.

XY uses one-based image pixels; positive Y points down. Angles are axial (180 degree
periodic). Every frame is fitted independently; no temporal averaging is applied to
any coordinate. See README.md for assumptions and limitations.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict, replace
import json
import os
from pathlib import Path
import subprocess
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent/'.mplconfig'))
import numpy as np
import tifffile
import networkx as nx
from scipy.ndimage import gaussian_filter, gaussian_filter1d, maximum_filter, zoom
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.interpolate import PchipInterpolator
from scipy.io import savemat

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent


def find_matlab() -> str | None:
    """The matlab executable: on PATH, else the newest release in the usual install folders; None if absent."""
    import shutil
    exe = shutil.which('matlab')
    if exe:
        return exe
    roots = [Path(os.environ.get(v, '')) / 'MATLAB' for v in ('ProgramFiles', 'ProgramW6432') if os.environ.get(v)]
    roots += [Path(f'{d}:/Program Files/MATLAB') for d in 'CDEF'] + [Path('/Applications'), Path('/usr/local/MATLAB')]
    found = [p for r in roots if r.is_dir() for p in r.glob('R20*/bin/matlab*') if p.name in ('matlab', 'matlab.exe')]
    found += [p for p in Path('/Applications').glob('MATLAB_R20*.app/bin/matlab')] if Path('/Applications').is_dir() else []
    return str(max(found, key=lambda p: p.parts[-3])) if found else None
# Defaults (20x data set); override with --calibration / --movie.
CALIBRATION = DATA/'20X/3. Inter-Lobe Calibration/20ms_0.1%beads_80um_10ums_2_MMStack_Pos0.ome_corrected.tif'
MOVIES = DATA/'20X/4. Indentation'
DEFAULT_MOVIES = {'cells': MOVIES/'20x_10ms_20um-indent_100ums_cells1_1_crop.tif',
                  'collagen': MOVIES/'20x_10ms_20um-indent_100ums_collagen1_2_crop.tif'}
_READERS = {}


class _PagedTiff:
    """Frame access for TIFFs that cannot be memory-mapped (e.g. some OME/Micro-Manager stacks)."""

    def __init__(self, path):
        self.file = tifffile.TiffFile(path)
        self.pages = self.file.series[0].pages
        shape = self.file.series[0].shape
        self.shape = (int(np.prod(shape[:-2])), *shape[-2:])

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        return self.pages[index].asarray()


def stack_reader(path):
    """Cached per-process frame reader: memory map when possible, else page reads."""
    key = str(path)
    if key not in _READERS:
        try:
            _READERS[key] = tifffile.memmap(key, mode='r')
        except ValueError:
            _READERS[key] = _PagedTiff(key)
    return _READERS[key]


def read_frame(path, index):
    return np.asarray(stack_reader(path)[index], dtype=np.float32)


def frame_count(path):
    return len(stack_reader(path))
COLUMNS = ['x1', 'x2', 'xMean', 'y1', 'y2', 'yMean', 'angleDegrees',
           'zMicrons', 'frame_number', 'track_number']
QUALITY = ['residualRMS', 'minLobeSNR', 'jointEmitterCount', 'lobeSeparationPixels',
           'amplitude1', 'amplitude2', 'sigma1', 'sigma2', 'pairCost', 'recovered',
           'zStatus', 'separationResidual', 'sharedLobe', 'templateScore', 'fieldSeparationOffset',
           'backgroundLevel', 'roiSignal', 'lobeSignal1', 'lobeSignal2',
           'xPrecisionPx', 'yPrecisionPx', 'anglePrecisionDeg', 'zPrecisionUm']
NCOL = len(COLUMNS)+len(QUALITY)
C = {name: i for i, name in enumerate(COLUMNS+QUALITY)}
VERSION = 2
FIT_REVISION = 3    # per-frame fit output changed (2: sandwich least-squares covariance; 3: local split refits, local-noise threshold); refits stale caches


@dataclass
class Config:
    # Detection
    dog_small: float = 1.2
    dog_large: float = 12.
    peak_window: int = 5
    seed_sigma_frames: float = 1.  # seeds from a Gaussian running average over frames/planes (0 = raw frame)
    threshold_sigma: float = 6.   # strong peaks: DoG > this many robust DoG noise SDs
    weak_fraction: float = .55    # weak (partner-only) peaks, as a fraction of the strong threshold
    threshold: float = 0.         # optional absolute DoG floor in counts
    local_noise_tile_px: int = 32  # threshold rises where a tile this size is noisier than the frame (0 = off)
    local_noise_min_ratio: float = 2.  # ... by at least this factor (robust SD of second differences)
    # Pairing
    min_separation: float = 9.
    max_separation: float = 27.
    seed_max_ratio: float = 8.
    max_ratio: float = 4.
    sep_sd_px: float = 1.
    sep_gate_px: float = 3.5
    sep_sd_mode: str = 'fixed'    # 'precision': separation prior SD from the lobes' precision (see pair_cost)
    sep_sd_floor_px: float = .15  # ... plus this calibration / field-map floor
    calibration_pruning: bool = True  # movies: only peaks at a calibrated lobe separation form pair edges
    prune_slack_px: float = 3.    # ... within sep_gate_px + this
    ratio_log_sd: float = .5
    sigma_sd: float = .6
    pair_gate_cost: float = 25.
    # Fitting and acceptance
    center_bound: float = 3.
    min_sigma: float = .8
    max_sigma: float = 5.
    fit_padding: int = 9
    max_group_gaussians: int = 24
    max_nfev: int = 200
    lsmr_min_params: int = 40     # fits with more parameters (> 9 Gaussians) use a sparse Jacobian
    noise_model: str = 'lsq'      # 'lsq': least squares; 'poisson': sCMOS maximum likelihood (needs camera_file)
    camera_file: str = ''         # camera.npz from camera_calibration.py (offset, variance, gain maps; ADU)
    camera_offset_x: float = 0.   # position of this image's top-left pixel on the camera sensor
    camera_offset_y: float = 0.
    fit_solver: str = 'trf'       # 'trf': scipy least_squares (default); 'lm': own Levenberg-Marquardt on the
                                  # normal equations (same results, but measured 4-6x slower on CPU; kept as a
                                  # basis for a batched GPU fitter)
    min_snr: float = 4.           # minimum lobe amplitude / camera pixel noise
    min_pair_snr: float = 0.      # > 0: accept pairs on their matched-filter SNR (pair_snr) instead, with
    lobe_floor_snr: float = 1.    #     each lobe only >= this x noise (used for dim data)
    roi_radius_px: float = 5.     # roiSignal: pixels within this distance of either lobe centre
    ring_shadow_ratio: float = .2
    ring_shadow_px: float = 10.
    shared_excess: float = .4     # shared lobe must exceed its partner by this x leftover amplitude
    shared_min_snr: float = 15.   # a shared-lobe pair's own lobe must reach this x camera noise
    split_sigma_ratio: float = 1.35  # a lobe this much wider than its window's other lobes is tried as two lobes
    split_min_gain: float = 1.     # ... kept if it adds a bead or lowers the total pairing cost by this much
    split_max_tries: int = 6       # ... at most this many split refits per fit window
    split_reach_sigmas: float = 3.  # ... refitting the lobes within this x max_sigma of the split lobe (others fixed)
    pixel_size_um: float = .325
    # Tracking
    tracker: str = 'lap'          # 'lap' (two-stage LAP), 'kalman' (constant-velocity Kalman + LAP), 'nearest'
    kalman_process_noise: float = .5      # acceleration SD, px/frame^2
    kalman_measurement_noise: float = .3  # localization SD, px
    kalman_initial_velocity: float = 2.   # prior velocity SD for a new track, px/frame
    link_distance_px: float = 8.
    gap_distance_px: float = 12.
    max_frame_gap: int = 6        # endpoint frame difference for gap closing
    angle_cost_deg: float = 15.   # 15 degrees of rotation costs as much as 1 px of motion
    link_z_um: float = 8.         # z-aware linking: never link across a z change above this ...
    link_z_sigma: float = 4.      # ... or this many combined z error bars, whichever is larger ...
    link_z_rate_um: float = 1.    # ... plus this much per skipped frame when closing gaps
    ghost_lobe_px: float = 8.     # ghost track: both lobes within this of other tracks' lobes ...
    ghost_fraction: float = .6    # ... in at least this fraction of its frames, and ...
    ghost_dim_ratio: float = .5   # ... at most this bright relative to those tracks, or ...
    ghost_z_um: float = 15.       # ... this far from their z
    # Track-guided recovery and transient rejection
    recover: bool = True
    recover_rounds: int = 6
    recover_reach_frames: int = 8
    recover_min_snr: float = 3.
    recover_center_bound: float = 4.
    recover_amplitude_fraction: float = .35  # recovered lobes must reach this x the track's median lobe amplitude
    recover_max_turn_deg: float = 15.        # recovered angle must stay this close to the predicted angle
    recover_min_track: int = 3
    axial_scale: float = 1.       # exported z = calibration z x this (true depth; 1.33 for an air objective and a
                                  # watery sample, paraxial n_sample/n_immersion); processing uses calibration units
    drift_z_smooth_frames: float = 3.  # z stabilization: Gaussian smoothing (sigma, frames) of the whole-field
                                       # z drift -- focus drifts smoothly; frame-to-frame dz is measurement noise
    jump_window: int = 5           # jump outliers: compare with the track's +-jump_window frames ...
    jump_px: float = 3.            # ... flag if further than this (+2x the neighbours' spread) ...
    jump_angle_deg: float = 20.    # ... or rotated more than this ...
    jump_z_um: float = 2.5         # ... or more than max(this, jump_z_sigma x the track's own frame-to-frame
    jump_z_sigma: float = 4.       #     z noise) away in z (3 x the local scatter for tracks under 7 frames)
    state_z_um: float = 5.         # two-state fits (alternating / sandwiched): states at least this far apart
    min_track_length: int = 3
    reject_track_snr: float = 15.      # low-quality track: median lobe SNR below this (and below
    reject_track_snr_fraction: float = .25  # this fraction of the movie's median track SNR) ...
    reject_track_template: float = .6  # ... and median template score below this
    satellite_distance_px: float = 10.     # satellite track: midpoint this close to a brighter track ...
    satellite_brightness_ratio: float = 3.  # ... that is at least this much brighter ...
    satellite_max_turn_deg: float = 20.    # ... at nearly the same angle ...
    satellite_fraction: float = .8         # ... in at least this fraction of its frames
    # Calibration
    z_step_um: float = 1.
    z_zero: str = 'auto'   # 'auto': horizontal or vertical lobes, whichever the stack's middle plane is closer to;
                           # 'middle': middle plane of the stack; or a lobe angle in degrees (e.g. '0' = horizontal)
    z_range_um: float | None = None    # report z only within +- this of z = 0 (pairing still uses the full calibration)
    sensor_size_px: int = 2048    # full camera width; cropped images are placed on it for the lateral model
    field_correction: bool = True  # learn a per-movie field map of lobe separation (off: calibration only)
    calibration_min_planes: int = 30
    calibration_min_beads_per_plane: int = 5
    calibration_max_residual_deg: float = 8.


# ----------------------------------------------------------------------------- detection

def dog(image, cfg):
    return gaussian_filter(image, cfg.dog_small)-gaussian_filter(image, cfg.dog_large)


def pixel_noise(image):
    """Camera noise SD from vertical neighbour differences (robust to sparse beads)."""
    d = np.diff(image[::2, ::2].astype(np.float32), axis=0).ravel()
    sd = np.std(d)
    for _ in range(3):  # clipped RMS: integer counts make a MAD too coarse
        sd = np.sqrt(np.mean(d[np.abs(d) < 3.5*sd]**2))/.997
    return max(float(sd/np.sqrt(2)), 1e-3)


def local_noise_ratio(image, tile, min_ratio=2.):
    """Pixel noise per tile relative to the whole frame's, interpolated to every pixel.

    Robust SD (MAD) of vertical second differences, which cancel smooth lobes and backgrounds.
    Tiles under min_ratio x the frame's noise get 1: around bright 20x beads and cells the ratio
    reaches ~2 (first differences, clipped RMS: up to ~7, which hid real overlapping beads),
    while a genuinely noisy region (noisy planted patches) is 3-40x.
    """
    img = np.asarray(image, dtype=np.float32)
    d = img[2:]-2*img[1:-1]+img[:-2]
    ny, nx = d.shape[0]//tile, d.shape[1]//tile
    if ny < 2 or nx < 2:
        return np.ones(img.shape, np.float32)
    t = d[:ny*tile, :nx*tile].reshape(ny, tile, nx, tile).transpose(0, 2, 1, 3).reshape(ny, nx, -1)
    mad = lambda a, axis=None: 1.4826*np.median(np.abs(a-np.median(a, axis=axis, keepdims=True)), axis=axis)
    ratio = mad(t, axis=2)/max(float(mad(d[::2, ::2])), 1e-3)
    ratio = np.where(ratio >= min_ratio, ratio, 1.)
    full = zoom(ratio, (img.shape[0]/ny, img.shape[1]/nx), order=1)
    out = np.ones(img.shape, np.float32)
    h, w = min(full.shape[0], img.shape[0]), min(full.shape[1], img.shape[1])
    out[:h, :w] = full[:h, :w]
    return out


def find_peaks(image, cfg):
    """DoG maxima above the weak threshold, their DoG values, and the strong threshold.

    With local_noise_tile_px > 0 the DoG is divided by local_noise_ratio, so the threshold rises
    where the image is noisier than the frame as a whole (and never drops). Without it, a noisy
    region gives hundreds of noise peaks that chain into huge fit windows: on frames with noisy
    planted beads, 427 s per frame and many false detections.
    """
    hp = dog(image, cfg)
    sub = hp[::4, ::4]
    noise = 1.4826*np.median(np.abs(sub-np.median(sub)))
    strong = max(cfg.threshold, cfg.threshold_sigma*noise)
    if cfg.local_noise_tile_px > 0:
        hp = hp/local_noise_ratio(image, cfg.local_noise_tile_px, cfg.local_noise_min_ratio)
    mask = (hp == maximum_filter(hp, cfg.peak_window)) & (hp > cfg.weak_fraction*strong)
    yx = np.column_stack(np.nonzero(mask))
    return yx[:, ::-1].astype(float), hp[mask], strong


def _plausible_pairs(xy_a, xy_b, cfg, model):
    """Could peaks a and b (zero-based xy) be the two lobes of one bead?

    Without a calibration: at least min_separation-2 apart. With one (movies): their
    separation must fit the calibrated separation at that angle (plus the movie's field
    map) within sep_gate_px + prune_slack_px -- the lobes of a bead are 16-23 px apart at
    10x, not anywhere in 7-27 px, so fewer spurious edges merge windows (~1.8x faster).
    """
    d = xy_b-xy_a
    sep = np.hypot(d[:, 0], d[:, 1])
    if model is None or not cfg.calibration_pruning:
        return sep >= cfg.min_separation-2
    angle = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 180
    mid = (xy_a+xy_b)/2+1
    r = model.separation_residuals(angle, sep, mid[:, 0], mid[:, 1])
    return np.isfinite(r) & (np.abs(r) <= cfg.sep_gate_px+cfg.prune_slack_px)


def seed_mask(xy, values, strong, cfg, model=None):
    """Peaks worth a Gaussian: strong peaks plus weak peaks that can partner a strong one."""
    use = values >= strong
    if len(xy) > 1 and use.any():
        weak = np.flatnonzero(~use)
        if len(weak):
            strong_idx = np.flatnonzero(use)
            tree = cKDTree(xy[strong_idx])
            for i, near in zip(weak, tree.query_ball_point(xy[weak], cfg.max_separation)):
                near = strong_idx[np.asarray(near, int)]
                near = near[values[near] <= cfg.seed_max_ratio*values[i]]
                if len(near):
                    use[i] = bool(_plausible_pairs(np.repeat(xy[i][None], len(near), 0), xy[near], cfg, model).any())
    return use


def _union_find(n, edges):
    parent = np.arange(n)
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a, b in edges:
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[rb] = ra
    groups = {}
    for i in range(n):
        groups.setdefault(root(i), []).append(i)
    return list(groups.values())


def candidate_groups(xy, values, strong, shape, cfg, model=None):
    """Fit windows: lobes connected by plausible pair edges, merged where windows overlap.

    Returns a list of (lo, hi) inclusive pixel windows, clipped to the image.
    """
    if len(xy) < 2:
        return []
    ab = cKDTree(xy).query_pairs(cfg.max_separation, output_type='ndarray')
    if not len(ab):
        return []
    a, b = ab.T
    hi_v, lo_v = np.maximum(values[a], values[b]), np.minimum(values[a], values[b])
    keep = (hi_v <= cfg.seed_max_ratio*lo_v) & (hi_v >= strong)
    keep[keep] = _plausible_pairs(xy[a[keep]], xy[b[keep]], cfg, model)
    ab = ab[keep]
    if not len(ab):
        return []
    members = np.unique(ab)
    index = {m: i for i, m in enumerate(members)}
    components = [[members[i] for i in g] for g in
                  _union_find(len(members), [(index[i], index[j]) for i, j in ab])]
    boxes = np.array([np.r_[xy[c].min(axis=0), xy[c].max(axis=0)] for c in components])
    boxes[:, :2] -= cfg.fit_padding
    boxes[:, 2:] += cfg.fit_padding
    overlap = []
    for i in range(len(boxes)):
        js = np.flatnonzero((boxes[i, 0] <= boxes[i+1:, 2]) & (boxes[i, 2] >= boxes[i+1:, 0]) &
                            (boxes[i, 1] <= boxes[i+1:, 3]) & (boxes[i, 3] >= boxes[i+1:, 1]))+i+1
        overlap.extend((i, j) for j in js)
    windows = []
    for merged in _union_find(len(boxes), overlap):
        box = np.r_[boxes[merged, :2].min(axis=0), boxes[merged, 2:].max(axis=0)]
        windows.extend(_split_window(xy, box, box.copy(), cfg))
    out = []
    for box, core in windows:
        lo = np.maximum(np.floor(box[:2]), 0).astype(int)
        hi = np.minimum(np.ceil(box[2:]), [shape[1]-1, shape[0]-1]).astype(int)
        out.append((lo, hi, core))
    return out


def _split_window(xy, box, core, cfg):
    """Recursively halve crowded windows (long axis, median cut) into overlapping sub-windows.

    Each sub-window keeps only pairs whose midpoint falls in its core; the overlap
    (max separation + padding) guarantees such pairs, and their neighbours, lie fully
    inside it. Returns [(box, core)].
    """
    inside = _inside(xy, box)
    if inside.sum() <= cfg.max_group_gaussians:
        return [(box, core)]
    axis = 0 if box[2]-box[0] >= box[3]-box[1] else 1
    cut = float(np.median(xy[inside, axis]))
    margin = cfg.max_separation+cfg.fit_padding
    halves = []
    for side in (0, 1):
        b, c = box.copy(), core.copy()
        if side == 0:
            b[2+axis] = min(box[2+axis], cut+margin); c[2+axis] = min(core[2+axis], cut)
        else:
            b[axis] = max(box[axis], cut-margin); c[axis] = max(core[axis], cut)
        halves.append((b, c))
    if any(_inside(xy, b).sum() >= inside.sum() for b, _ in halves):
        return [(box, core)]   # no progress possible (dense cluster smaller than the overlap)
    return [w for b, c in halves for w in _split_window(xy, b, c, cfg)]


def _inside(xy, box):
    return (xy[:, 0] >= box[0]) & (xy[:, 0] <= box[2]) & (xy[:, 1] >= box[1]) & (xy[:, 1] <= box[3])


def model_jac(p, x, y):
    """Sum of circular Gaussians plus an affine background, with exact Jacobian."""
    result = p[-3] + p[-2]*x + p[-1]*y
    jac = np.empty((len(x), len(p)))
    for k in range((len(p)-3)//4):
        amp, cx, cy, sigma = p[4*k:4*k+4]
        dx, dy = x-cx, y-cy
        r2 = dx*dx + dy*dy
        e = np.exp(-r2/(2*sigma*sigma))
        g = amp*e
        result = result + g
        jac[:, 4*k:4*k+4] = np.column_stack((e, g*dx/sigma**2, g*dy/sigma**2, g*r2/sigma**3))
    jac[:, -3:] = np.column_stack((np.ones(len(x)), x, y))
    return result, jac


class _SparseModel:
    """Sum of Gaussians + affine background with a sparse Jacobian.

    Each Gaussian is evaluated only on pixels within `reach` of its seed (centre bound +
    4 x max sigma), so a window of many lobes costs ~ sum of lobe footprints, not
    pixels x parameters. Truncation error < exp(-8) of the amplitude.
    """

    def __init__(self, x, y, seeds, bounds, cfg):
        from scipy import sparse
        self.sparse = sparse
        self.x, self.y, self.n = x, y, len(x)
        self.idx = []
        for (cx, cy), cb in zip(seeds, bounds):
            reach = cb+4*cfg.max_sigma
            self.idx.append(np.flatnonzero((np.abs(x-cx) <= reach) & (np.abs(y-cy) <= reach)))
        k = len(seeds)
        rows = [np.repeat(i, 4) for i in self.idx]
        self.rows = np.concatenate(rows+[np.arange(self.n)]*3)
        self.cols = np.concatenate([np.tile(4*j+np.arange(4), len(i)) for j, i in enumerate(self.idx)] +
                                   [np.full(self.n, 4*k+c) for c in range(3)])
        self.shape = (self.n, 4*k+3)
        self.bg_data = np.concatenate((np.ones(self.n), x, y))

    def __call__(self, p):
        result = p[-3]+p[-2]*self.x+p[-1]*self.y
        blocks = []
        for j, i in enumerate(self.idx):
            amp, cx, cy, sigma = p[4*j:4*j+4]
            dx, dy = self.x[i]-cx, self.y[i]-cy
            r2 = dx*dx+dy*dy
            e = np.exp(-r2/(2*sigma*sigma))
            g = amp*e
            result[i] += g
            blocks.append(np.column_stack((e, g*dx/sigma**2, g*dy/sigma**2, g*r2/sigma**3)).ravel())
        data = np.concatenate(blocks+[self.bg_data])
        jac = self.sparse.csr_matrix((data, (self.rows, self.cols)), shape=self.shape)
        return result, jac


class _Fit:
    """Minimal least_squares-like result."""

    def __init__(self, x, fun, status):
        self.x, self.fun, self.status = x, fun, status


def _levenberg_marquardt(compute, data, p0, lower, upper, max_iter):
    """Bounded Levenberg-Marquardt on the normal equations (J^T J + lambda diag) dp = -J^T r.

    `compute(p)` returns (model, Jacobian) with a dense or sparse Jacobian. Only the
    p x p system is solved per step (p <= ~200), so the cost is dominated by building
    J and J^T J -- far cheaper than an SVD of J or iterative LSMR solves for windows
    with many Gaussians. Bounds are enforced by clipping (projected steps).
    """
    p = np.clip(np.asarray(p0, float), lower, upper)
    model, J = compute(p)
    r = model-data
    cost = float(r@r)
    lam = 1e-3
    status = 0
    for _ in range(max_iter):
        JtJ = J.T@J
        JtJ = JtJ.toarray() if hasattr(JtJ, 'toarray') else np.asarray(JtJ)
        g = np.asarray(J.T@r).ravel()
        diag = np.maximum(np.diag(JtJ), 1e-12)
        improved = False
        while lam < 1e10:
            A = JtJ+lam*np.diag(diag)
            try:
                step = -np.linalg.solve(A, g)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            pn = np.clip(p+step, lower, upper)
            mn, Jn = compute(pn)
            rn = mn-data
            cn = float(rn@rn)
            if cn < cost:
                rel = (cost-cn)/max(cost, 1e-30)
                small = np.max(np.abs(pn-p)/(np.abs(p)+1e-3)) < 1e-6
                p, model, J, r, cost = pn, mn, Jn, rn, cn
                lam = max(lam/3, 1e-9)
                improved = True
                if rel < 1e-7 or small:
                    status = 1
                break
            lam *= 4
        if not improved:
            status = 2      # no further decrease possible: at a (local) minimum
            break
        if status == 1:
            break
    return _Fit(p, r, max(status, 1))


_CAMERAS = {}


def camera_patch(cfg, lo, hi):
    """(offset, variance, gain) for image pixels lo..hi (inclusive, zero-based), or None.

    The camera file (see camera_calibration.py) holds full-sensor maps in ADU: offset,
    read-noise variance and gain (ADU per photoelectron, map or scalar). The image is
    placed on the sensor with cfg.camera_offset_x/y.
    """
    if not cfg.camera_file:
        return None
    if cfg.camera_file not in _CAMERAS:
        with np.load(cfg.camera_file) as z:
            cam = {k: np.asarray(z[k], dtype=float) for k in ('offset', 'variance', 'gain')}
            cam['roi'] = np.asarray(z['roi'], dtype=int) if 'roi' in z.files else np.zeros(2, int)
        _CAMERAS[cfg.camera_file] = cam
    cam = _CAMERAS[cfg.camera_file]
    ox = int(round(cfg.camera_offset_x))-int(cam['roi'][0])
    oy = int(round(cfg.camera_offset_y))-int(cam['roi'][1])
    h, w = cam['offset'].shape
    if lo[0]+ox < 0 or lo[1]+oy < 0 or hi[0]+ox >= w or hi[1]+oy >= h:
        return None     # outside the calibrated sensor area: fall back to least squares
    ys, xs = slice(lo[1]+oy, hi[1]+oy+1), slice(lo[0]+ox, hi[0]+ox+1)
    pick = lambda a: (a[ys, xs].ravel() if a.ndim == 2 else np.full((hi[1]-lo[1]+1)*(hi[0]-lo[0]+1), float(a)))
    return pick(cam['offset']), np.maximum(pick(cam['variance']), 1e-6), np.maximum(pick(cam['gain']), 1e-6)


def _poisson_residuals(compute, data, cam):
    """Deviance residuals of the sCMOS Poisson model (Huang et al. 2013), with Jacobian.

    Pixel values are converted to photoelectrons plus gamma = read variance / gain^2, which
    makes Poisson + Gaussian read noise approximately Poisson; sum(r^2) is then twice the
    negative log-likelihood ratio, so least squares on r is maximum likelihood.
    """
    from scipy.special import xlogy
    off, var, g = cam
    gamma = var/g**2
    d = np.maximum((data-off)/g+gamma, 0.)

    def evaluate(p):
        m, J = compute(p)
        raw = (m-off)/g+gamma
        floor = raw < 1e-3          # non-physical expectation during an iteration: no gradient there
        mu = np.maximum(raw, 1e-3)
        dev = np.maximum(2*(mu-d-xlogy(d, mu)+xlogy(d, d)), 0.)
        r = np.sign(mu-d)*np.sqrt(dev)
        drdmu = np.where(np.abs(r) > 1e-8, (1-d/mu)/np.where(np.abs(r) > 1e-8, r, 1.), 1/np.sqrt(mu))
        scale = np.where(floor, 0., drdmu/g)
        if hasattr(J, 'multiply'):
            from scipy import sparse
            J = sparse.diags(scale)@J
        else:
            J = J*scale[:, None]
        return r, J
    return evaluate


def fit_gaussians(image, lo, hi, seeds, bounds, cfg):
    """Jointly fit free circular Gaussians seeded at `seeds` (absolute xy) in a window.

    `bounds` gives each seed's allowed centre displacement. Least squares by default; with
    cfg.noise_model == 'poisson' and a camera file, maximum likelihood under the sCMOS
    noise model. Returns (params (k,4) with absolute centres, at_bound flags (k,), residual
    RMS, background plane, covariance of the centres (2k x 2k, px^2)) or None on failure.
    """
    patch = image[lo[1]:hi[1]+1, lo[0]:hi[0]+1].astype(float)
    yy, xx = np.indices(patch.shape)
    origin = (lo+hi)/2
    x, y = (xx+lo[0]-origin[0]).ravel(), (yy+lo[1]-origin[1]).ravel()
    background = np.percentile(patch, 25)
    p0, lower, upper = [], [], []
    for (cx, cy), cb in zip(seeds, bounds):
        iy = int(np.clip(round(cy), 0, image.shape[0]-1))
        ix = int(np.clip(round(cx), 0, image.shape[1]-1))
        amplitude = max(float(image[iy, ix])-background, 1.)
        cx, cy = cx-origin[0], cy-origin[1]
        p0.extend([amplitude, cx, cy, 2.2])
        lower.extend([0, cx-cb, cy-cb, cfg.min_sigma])
        upper.extend([np.inf, cx+cb, cy+cb, cfg.max_sigma])
    p0.extend([background, 0, 0])
    lower.extend([-np.inf]*3)
    upper.extend([np.inf]*3)
    data = patch.ravel()
    cache = {}
    # Large windows: a dense Jacobian is mostly zeros; evaluate each Gaussian only near its seed.
    large = len(p0) > cfg.lsmr_min_params
    compute = (_SparseModel(x, y, np.asarray(seeds)-origin, bounds, cfg) if large
               else (lambda p: model_jac(p, x, y)))
    cam = camera_patch(cfg, lo, hi) if cfg.noise_model == 'poisson' else None
    if cam is not None:
        # Remove each pixel's own offset deviation so one smooth background plane is a valid
        # model (Huang et al. 2013); the mean offset stays inside the fitted background.
        off, var, g = cam
        data = data-(off-off.mean())
        cam = (np.full_like(off, off.mean()), var, g)
    if cfg.fit_solver == 'lm' and cam is None:
        fit = _levenberg_marquardt(compute, data, p0, np.asarray(lower), np.asarray(upper), cfg.max_nfev)
        fit.jac = compute(fit.x)[1]
    else:
        residual = _poisson_residuals(compute, data, cam) if cam is not None else \
            (lambda p: (lambda m, J: (m-data, J))(*compute(p)))
        def evaluate(p):
            key = p.tobytes()
            if key not in cache:
                cache.clear()
                cache[key] = residual(p)
            return cache[key]
        try:
            fit = least_squares(lambda p: evaluate(p)[0], p0,
                                jac=lambda p: evaluate(p)[1], bounds=(lower, upper),
                                max_nfev=cfg.max_nfev, ftol=1e-5, xtol=1e-5, gtol=1e-5, x_scale='jac',
                                tr_solver='lsmr' if large else 'exact')
        except ValueError:
            return None
    if fit.status < 0:
        return None
    params = fit.x[:-3].reshape(-1, 4).copy()
    shift = np.abs(params[:, 1:3]-(np.asarray(seeds)-origin)).max(axis=1)
    at_bound = ((shift >= np.asarray(bounds)*.99) | (params[:, 3] <= cfg.min_sigma*1.01) |
                (params[:, 3] >= cfg.max_sigma*.99))
    params[:, 1:3] += origin
    background = np.r_[fit.x[-3:], origin]   # b0, bx, by, origin_x, origin_y (zero-based px)
    model = None if cam is not None else data+fit.fun     # least squares: residual = model - data
    return params, at_bound, float(np.sqrt(np.mean(fit.fun**2))), background, \
        _centre_covariance(fit.jac, fit.fun, len(p0), len(params), cam is not None, model)


def _centre_covariance(J, residual, n_params, n_gauss, likelihood, model=None):
    """Covariance of the fitted centres (x, y per Gaussian).

    Maximum likelihood (deviance residuals): the inverse Fisher information (J^T J)^-1.
    Least squares: the sandwich (J^T J)^-1 J^T S J (J^T J)^-1 with a per-pixel variance
    S_i = a + b*model_i fitted to the squared residuals of this window. Shot noise makes
    the lobe pixels noisier than the background, so the textbook s^2 (J^T J)^-1 (one
    variance for all pixels, dominated by background) understated the scatter 1.5-2x.
    """
    JtJ = J.T@J
    JtJ = JtJ.toarray() if hasattr(JtJ, 'toarray') else np.asarray(JtJ)
    try:
        inv = np.linalg.pinv(JtJ, hermitian=True)
    except np.linalg.LinAlgError:
        return np.full((2*n_gauss, 2*n_gauss), np.nan)
    if likelihood:
        cov = inv
    else:
        dof = max(len(residual)-n_params, 1)
        r2 = residual**2*len(residual)/dof                 # degrees-of-freedom corrected
        var = np.full(len(residual), float(r2.mean()))
        if model is not None and np.ptp(model) > 0:
            # variance model a + b*model (shot noise grows with the signal), b >= 0
            A = np.column_stack((np.ones_like(model), model-model.min()))
            coef, *_ = np.linalg.lstsq(A, r2, rcond=None)
            if coef[1] > 0:
                var = np.maximum(A@coef, .1*float(r2.mean()))
        Jw = J.multiply(var[:, None]) if hasattr(J, 'multiply') else J*var[:, None]
        meat = J.T@Jw
        meat = meat.toarray() if hasattr(meat, 'toarray') else np.asarray(meat)
        cov = inv@meat@inv
    idx = np.ravel([[4*k+1, 4*k+2] for k in range(n_gauss)])
    return cov[np.ix_(idx, idx)]


def pair_precision(params, cov, a, b):
    """(sigma x_mid, sigma y_mid, sigma angle in degrees) of a lobe pair from the centre covariance."""
    ia, ib = [2*a, 2*a+1], [2*b, 2*b+1]
    Vaa, Vbb, Vab = cov[np.ix_(ia, ia)], cov[np.ix_(ib, ib)], cov[np.ix_(ia, ib)]
    mid = (Vaa+Vbb+Vab+Vab.T)/4
    d = params[b, 1:3]-params[a, 1:3]
    grad = np.array([-d[1], d[0]])/max(float(d@d), 1e-9)
    var_angle = float(grad@(Vaa+Vbb-Vab-Vab.T)@grad)
    return (float(np.sqrt(max(mid[0, 0], 0))), float(np.sqrt(max(mid[1, 1], 0))),
            float(np.degrees(np.sqrt(max(var_angle, 0)))))


def add_signals(row, image, background, cfg, shared_xy=None):
    """Fill backgroundLevel, roiSignal and lobeSignal1/2 (camera counts) for one localization.

    roiSignal: sum over pixels within roi_radius_px of either lobe centre of (image - fitted
    affine background). lobeSignal: integrated fitted Gaussian, 2*pi*A*sigma^2. For a shared
    lobe (coincident with a neighbour's lobe, `shared_xy` zero-based) the lobe signal is
    not separable and is NaN; the ROI then also contains the neighbour's light.
    """
    b0, bx, by, ox, oy = background
    lobes = np.array([[row[0], row[3]], [row[1], row[4]]])-1
    mid = lobes.mean(axis=0)
    row[C['backgroundLevel']] = b0+bx*(mid[0]-ox)+by*(mid[1]-oy)
    r = cfg.roi_radius_px
    lo = np.maximum(np.floor(lobes.min(axis=0)-r), 0).astype(int)
    hi = np.minimum(np.ceil(lobes.max(axis=0)+r), [image.shape[1]-1, image.shape[0]-1]).astype(int)
    yy, xx = np.mgrid[lo[1]:hi[1]+1, lo[0]:hi[0]+1]
    mask = np.zeros(xx.shape, bool)
    for cx, cy in lobes:
        mask |= (xx-cx)**2+(yy-cy)**2 <= r*r
    plane = b0+bx*(xx-ox)+by*(yy-oy)
    row[C['roiSignal']] = float(np.sum((image[lo[1]:hi[1]+1, lo[0]:hi[0]+1]-plane)[mask]))
    for k, (amp, sigma) in enumerate(((C['amplitude1'], C['sigma1']), (C['amplitude2'], C['sigma2']))):
        shared = shared_xy is not None and np.hypot(*(lobes[k]-shared_xy)) < 1.5
        row[C['lobeSignal1']+k] = np.nan if shared else 2*np.pi*row[amp]*row[sigma]**2
    return row


def pair_geometry(p1, p2):
    """Canonical lobe order (nonnegative dy), angle in [0,180), separation."""
    xy = np.array([p1[1:3], p2[1:3]])
    delta = xy[1]-xy[0]
    swap = delta[1] < 0 or (delta[1] == 0 and delta[0] < 0)
    if swap:
        xy, delta = xy[::-1], -delta
    angle = float(np.degrees(np.arctan2(delta[1], delta[0])) % 180)
    return xy, angle, float(np.hypot(*delta)), swap


def pair_snr(p1, p2, noise):
    """Matched-filter SNR of a fitted lobe pair: sqrt(sum over pixels of the model^2) / noise.

    For a circular Gaussian of amplitude A and width s, sum(model^2) = pi A^2 s^2. Unlike the
    weaker lobe's amplitude alone, it uses the light of both lobes and their widths.
    """
    return float(np.sqrt(np.pi*(p1[0]**2*p1[3]**2+p2[0]**2*p2[3]**2))/max(noise, 1e-9))


def pair_cost(p1, p2, cfg, model, noise=None):
    """Pair plausibility cost (lower is better) or inf when a hard gate fails."""
    _, angle, sep, _ = pair_geometry(p1, p2)
    a1, a2 = p1[0], p2[0]
    if not (cfg.min_separation <= sep <= cfg.max_separation) or min(a1, a2) <= 0:
        return np.inf, np.nan
    ratio = max(a1, a2)/min(a1, a2)
    if ratio > cfg.max_ratio:
        return np.inf, np.nan
    cost = (np.log(ratio)/cfg.ratio_log_sd)**2 + ((p1[3]-p2[3])/cfg.sigma_sd)**2
    mid = (p1[1:3]+p2[1:3])/2+1   # one-based
    residual = model.separation_residual(angle, sep, mid[0], mid[1]) if model is not None else np.nan
    if np.isfinite(residual):
        sd, gate = cfg.sep_sd_px, cfg.sep_gate_px
        if cfg.sep_sd_mode == 'precision' and noise:
            # expected scatter of the separation from the lobes' localization precision
            # (~0.8 noise/A px per lobe for these lobe widths) plus a floor for calibration error
            s1, s2 = .8*noise/a1, .8*noise/a2
            sd = float(np.sqrt(cfg.sep_sd_floor_px**2+s1*s1+s2*s2))
            gate = min(cfg.sep_gate_px, max(1.2, 6*sd))
        if abs(residual) > gate:
            return np.inf, residual
        cost += (residual/sd)**2
    else:
        # No calibration yet (or angle outside it): weak prior on the observed 16-23 px range.
        cost += (max(15.-sep, 0, sep-24.)/cfg.sep_sd_px)**2
    return cost, residual


def match_lobes(params, usable, cfg, model, noise=None):
    """Global maximum-weight matching of fitted lobes into pairs.

    With min_pair_snr > 0 (and the pixel noise), a pair must reach that matched-filter SNR
    (see pair_snr); each lobe then only needs lobe_floor_snr, instead of min_snr each.
    """
    idx = np.flatnonzero(usable)
    if len(idx) < 2:
        return []
    tree = cKDTree(params[idx, 1:3])
    graph = nx.Graph()
    info = {}
    for i, j in tree.query_pairs(cfg.max_separation):
        a, b = idx[i], idx[j]
        if cfg.min_pair_snr > 0 and noise and pair_snr(params[a], params[b], noise) < cfg.min_pair_snr:
            continue
        cost, residual = pair_cost(params[a], params[b], cfg, model, noise)
        if cost <= cfg.pair_gate_cost:
            graph.add_edge(a, b, weight=cfg.pair_gate_cost+1-cost)
            info[(min(a, b), max(a, b))] = (cost, residual)
    matching = nx.max_weight_matching(graph) if graph.number_of_edges() else set()
    return [(min(a, b), max(a, b), *info[(min(a, b), max(a, b))]) for a, b in matching]


def _split_positions(image, params, k, background):
    """Two sub-positions for a too-wide fitted lobe k, along its elongation axis (or None).

    The neighbouring Gaussians and the background plane are subtracted; second moments
    of the remaining light within 2.5 sigma give the axis. Two point-like lobes 2a apart
    add a^2 to the variance along the axis: a = sqrt(l1 - l2).
    """
    amp, cx, cy, s = params[k]
    r = int(np.ceil(3*s))
    x0, x1 = max(int(cx)-r, 0), min(int(cx)+r+1, image.shape[1])
    y0, y1 = max(int(cy)-r, 0), min(int(cy)+r+1, image.shape[0])
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(float)
    b0, bx, by, ox, oy = background
    light = image[y0:y1, x0:x1]-(b0+bx*(xx-ox)+by*(yy-oy))
    for j, (a, px, py, sj) in enumerate(params):
        if j != k:
            light = light-a*np.exp(-((xx-px)**2+(yy-py)**2)/(2*sj*sj))
    w = np.clip(light, 0, None)*(((xx-cx)**2+(yy-cy)**2) <= (2.5*s)**2)
    total = w.sum()
    if total <= 0:
        return None
    mx, my = (w*xx).sum()/total, (w*yy).sum()/total
    cov = np.array([[(w*(xx-mx)**2).sum(), (w*(xx-mx)*(yy-my)).sum()],
                    [(w*(xx-mx)*(yy-my)).sum(), (w*(yy-my)**2).sum()]])/total
    (l2, l1), vectors = np.linalg.eigh(cov)
    if l1 < 1.3*l2:
        return None
    a = np.sqrt(max(l1-l2, 0.))
    axis = vectors[:, 1]
    return np.array([[mx, my]+a*axis, [mx, my]-a*axis])


def _pairing_score(pairs, shared):
    """Higher is better: more beads first, then lower total pairing cost."""
    return (len(pairs)+len(shared), -sum(p[2] for p in pairs+shared))


def _local_split_fit(image, work, lo, hi, params, k, halves, cfg):
    """Refit only the neighbourhood of lobe k, with k replaced by the two `halves`.

    Gaussians within split_reach_sigmas x max_sigma of lobe k are refitted with the halves;
    all others stay fixed and are subtracted from the data. The fit box is these lobes plus
    fit_padding, inside the window. A split only changes its own neighbourhood, so refitting
    the whole window (up to ~40 Gaussians) per attempt cost a third of the run time on 20x
    frames and over half on noisy ones. `work` is a writable float copy of `image`.
    Returns (indices refitted, fit_gaussians result) or None; the fit's Gaussians are the
    refitted ones in that order, then the two halves.
    """
    free = np.flatnonzero(np.hypot(*(params[:, 1:3]-params[k, 1:3]).T) <= cfg.split_reach_sigmas*cfg.max_sigma)
    free = free[free != k]
    pts = np.vstack((params[free, 1:3], halves))
    blo = np.maximum(np.floor(pts.min(axis=0)-cfg.fit_padding), lo).astype(int)
    bhi = np.minimum(np.ceil(pts.max(axis=0)+cfg.fit_padding), hi).astype(int)
    ys, xs = slice(blo[1], bhi[1]+1), slice(blo[0], bhi[0]+1)
    yy, xx = np.mgrid[ys, xs].astype(float)
    data = image[ys, xs].astype(float)
    for j in np.setdiff1d(np.arange(len(params)), np.r_[free, k]):
        a, px, py, s = params[j]
        data = data-a*np.exp(-((xx-px)**2+(yy-py)**2)/(2*s*s))
    work[ys, xs] = data
    try:
        fit = fit_gaussians(work, blo, bhi, pts, np.full(len(pts), cfg.center_bound), cfg)
    finally:
        work[ys, xs] = image[ys, xs]
    return None if fit is None else (free, fit)


def split_wide_lobes(image, lo, hi, params, usable, background, noise, pairs, shared, cfg, model, cov):
    """Try splitting too-wide lobes (two lobes merged into one Gaussian) and keep improvements.

    Two beads side by side can place two lobes ~5 px apart; they then merge into one wide
    Gaussian and force a crossed or shared-lobe pairing. A lobe wider than
    split_sigma_ratio x the median width of the other usable lobes in the window is
    replaced by two Gaussians along its elongation axis, its neighbourhood is refitted
    (_local_split_fit) and the window re-paired; the result is kept only if the pairing
    score improves by a clear margin. At most split_max_tries refits per window.
    Returns (params, usable, background, pairs, shared, n_splits, centre covariance).
    """
    splits, tries, work = 0, 0, None
    for _ in range(2):
        widths = params[usable, 3]
        if usable.sum() < 3:
            break
        wide = [k for k in np.flatnonzero(usable)
                if params[k, 3] > cfg.split_sigma_ratio*np.median(np.delete(widths, np.flatnonzero(np.flatnonzero(usable) == k)))]
        improved = False
        for k in sorted(wide, key=lambda k: -params[k, 3]):
            if tries >= cfg.split_max_tries:
                break
            halves = _split_positions(image, params, k, background)
            if halves is None:
                continue
            tries += 1
            if work is None:
                work = np.array(image, dtype=float)
            local = _local_split_fit(image, work, lo, hi, params, k, halves, cfg)
            if local is None:
                continue
            free, (pf, at_bound_f, _, _, cov_f) = local
            # new Gaussians: all but k in their order (the free ones refitted), then the two halves
            keep = np.delete(np.arange(len(params)), k)
            p2 = np.vstack((params[keep], pf[len(free):]))
            usable2 = np.r_[usable[keep], np.zeros(2, bool)]
            refit = np.r_[np.searchsorted(keep, free), len(keep), len(keep)+1]
            p2[refit] = pf
            inside = ((pf[:, 1] >= 0) & (pf[:, 1] <= image.shape[1]-1) & (pf[:, 2] >= 0) & (pf[:, 2] <= image.shape[0]-1))
            usable2[refit] = ~at_bound_f & inside & (pf[:, 0] >= lobe_threshold(cfg)*noise)
            # centre covariance: unchanged Gaussians keep theirs, refitted ones take the local fit's
            # (covariances between the two groups are dropped)
            ix = lambda idx: np.ravel([[2*i, 2*i+1] for i in idx]).astype(int)
            cov2 = np.zeros((2*len(p2), 2*len(p2)))
            same = np.setdiff1d(np.arange(len(keep)), refit)
            cov2[np.ix_(ix(same), ix(same))] = cov[np.ix_(ix(keep[same]), ix(keep[same]))]
            cov2[np.ix_(ix(refit), ix(refit))] = cov_f
            pairs2 = match_lobes(p2, usable2, cfg, model, noise)
            shared2 = shared_lobe_pairs(p2, usable2, pairs2, cfg, model, noise)
            old, new = _pairing_score(pairs, shared), _pairing_score(pairs2, shared2)
            if new[0] > old[0] or (new[0] == old[0] and new[1] > old[1]+cfg.split_min_gain):
                params, usable, pairs, shared, cov = p2, usable2, pairs2, shared2, cov2
                splits += 1
                improved = True
                break
        if not improved:
            break
    return params, usable, background, pairs, shared, splits, cov


def lobe_threshold(cfg):
    """Minimum lobe amplitude (x pixel noise) for a fitted Gaussian to take part in pairing."""
    return cfg.lobe_floor_snr if cfg.min_pair_snr > 0 else cfg.min_snr


def shared_lobe_pairs(params, usable, pairs, cfg, model, noise=None):
    """Pair leftover lobes with an already-paired lobe that can hide two coincident lobes.

    Two beads whose lobes coincide show three spots with a bright middle one. After the
    matching, a usable unpaired lobe may take an already-paired lobe as its partner if
    the geometry fits the calibrated separation-angle curve. The shared lobe's position
    is the fitted centroid of the coincident lobes; the amplitude-ratio test is skipped
    for it. Leftovers on the side-lobe ring of a much brighter lobe are not eligible.
    """
    if model is None or not pairs:
        return []
    paired = sorted({i for a, b, *_ in pairs for i in (a, b)})
    partner = {a: b for a, b, *_ in pairs} | {b: a for a, b, *_ in pairs}
    taken = set(paired)
    out = []
    for u in sorted(set(np.flatnonzero(usable))-taken, key=lambda k: -params[k, 0]):
        if noise and params[u, 0] < cfg.min_snr*noise:
            continue   # a shared pair rests on its own lobe alone: it must be clearly above noise
        near_bright = [v for v in range(len(params)) if v != u and
                       np.hypot(*(params[v, 1:3]-params[u, 1:3])) < cfg.ring_shadow_px and
                       params[u, 0] < cfg.ring_shadow_ratio*params[v, 0]]
        if near_bright:
            continue
        best = None
        for v in paired:
            # Two coincident lobes: the shared spot carries extra amplitude beyond its own
            # partner's, and the leftover is not a faint ring arc of the shared spot.
            excess = params[v, 0]-params[partner[v], 0]
            if params[u, 0] < cfg.ring_shadow_ratio*params[v, 0] or excess < cfg.shared_excess*params[u, 0]:
                continue
            proxy = params[v].copy()
            proxy[0] = params[u, 0]   # the hidden lobe's own amplitude is unobservable
            cost, residual = pair_cost(params[u], proxy, cfg, model, noise)
            if np.isfinite(residual) and cost <= cfg.pair_gate_cost and (best is None or cost < best[2]):
                best = (u, v, cost, residual)
        if best is not None:
            out.append(best)
            taken.add(u)
    return out


def make_row(p1, p2, frame, rms, snr, joint, cost, residual, recovered, precision=(np.nan, np.nan, np.nan)):
    xy, angle, sep, swap = pair_geometry(p1, p2)
    q1, q2 = (p2, p1) if swap else (p1, p2)
    a, b = xy+1  # MATLAB pixels are one-based
    row = np.full(NCOL, np.nan)
    row[:10] = [a[0], b[0], (a[0]+b[0])/2, a[1], b[1], (a[1]+b[1])/2, angle, np.nan, frame, 0]
    row[10:] = [rms, snr, joint, sep, q1[0], q2[0], q1[3], q2[3], cost, recovered, 1, residual, 0, np.nan, np.nan,
                np.nan, np.nan, np.nan, np.nan, *precision, np.nan]
    return row


def seed_image(path, index, cfg):
    """Gaussian-weighted running average of neighbouring frames (or z-planes), for seeding only.

    Beads move slowly, so averaging +-2*seed_sigma_frames frames raises lobe contrast about
    1.9x (sigma=1) without smearing positions beyond the fit's +-3 px seed tolerance.
    Fits are always made on the single raw frame. Returns None when disabled.
    """
    sigma = cfg.seed_sigma_frames
    if sigma <= 0:
        return None
    n = frame_count(path)
    reach = int(np.ceil(2*sigma))
    offsets = [k for k in range(-reach, reach+1) if 0 <= index+k < n]
    weights = np.exp(-np.square(offsets)/(2*sigma*sigma))
    total = np.zeros(stack_reader(path).shape[-2:], np.float32)
    for k, w in zip(offsets, weights):
        total += w*read_frame(path, index+k)
    return total/weights.sum()


def localize_image(image, cfg, model, frame=0, seed=None):
    """All accepted pairs in one image. Returns (rows, stats).

    `seed`: optional smoothed image (see seed_image) used only to find candidate lobes;
    fitting, acceptance and signals always use `image`.
    """
    image = np.asarray(image, dtype=np.float32)
    noise = pixel_noise(image)
    xy, values, strong = find_peaks(image if seed is None else seed, cfg)
    stats = dict(frame=frame, peaks=len(xy), pixel_noise=noise, strong_threshold=float(strong))
    keep = seed_mask(xy, values, strong, cfg, model)
    xy, values = xy[keep], values[keep]
    stats['seed_peaks'] = len(xy)
    windows = candidate_groups(xy, values, strong, image.shape, cfg, model)
    rows = []
    for lo, hi, core in windows:
        seeds = xy[_inside(xy, np.r_[lo, hi])]
        if len(seeds) > 2*cfg.max_group_gaussians:
            stats['crowded_windows'] = stats.get('crowded_windows', 0)+1
            continue
        fit = fit_gaussians(image, lo, hi, seeds, np.full(len(seeds), cfg.center_bound), cfg)
        if fit is None:
            stats['failed_windows'] = stats.get('failed_windows', 0)+1
            continue
        params, at_bound, rms, background, cov = fit
        inside =((params[:, 1] >= 0) & (params[:, 1] <= image.shape[1]-1) &
                  (params[:, 2] >= 0) & (params[:, 2] <= image.shape[0]-1))
        usable = ~at_bound & inside & (params[:, 0] >= lobe_threshold(cfg)*noise)
        pairs = match_lobes(params, usable, cfg, model, noise)
        shared = shared_lobe_pairs(params, usable, pairs, cfg, model, noise)
        if model is not None:
            params, usable, background, pairs, shared, splits, cov = split_wide_lobes(
                image, lo, hi, params, usable, background, noise, pairs, shared, cfg, model, cov)
            stats['split_lobes'] = stats.get('split_lobes', 0)+splits
        in_core =lambda a, b: bool(_inside(((params[a, 1:3]+params[b, 1:3])/2)[None], core)[0])
        pairs = [p for p in pairs if in_core(p[0], p[1])]
        shared = [p for p in shared if in_core(p[0], p[1])]
        joint = len(pairs)+len(shared)
        for a, b, cost, residual in pairs:
            row = make_row(params[a], params[b], frame, rms, min(params[a, 0], params[b, 0])/noise,
                           joint, cost, residual, 0, pair_precision(params, cov, a, b))
            rows.append(add_signals(row, image, background, cfg))
        for u, v, cost, residual in shared:
            row = make_row(params[u], params[v], frame, rms, params[u, 0]/noise, joint, cost, residual, 0,
                           pair_precision(params, cov, u, v))
            row[C['sharedLobe']] = 1
            rows.append(add_signals(row, image, background, cfg, shared_xy=params[v, 1:3]))
        stats['shared_lobe_pairs'] = stats.get('shared_lobe_pairs', 0)+len(shared)
    rows = np.asarray(rows).reshape(-1, NCOL)
    rows, duplicates = reject_duplicate_lobes(rows, cfg, noise)
    stats['duplicate_lobe_rejected'] = duplicates
    rows, shadowed = reject_ring_shadows(rows, cfg)
    # Two windows can share a lobe region only in the crowded fallback; keep the better copy.
    rows = deduplicate(rows)
    stats.update(windows=len(windows), accepted=len(rows), ring_shadow_rejected=shadowed)
    return rows, stats


def reject_duplicate_lobes(rows, cfg, noise, distance=3.):
    """Drop phantom pairs that reuse a lobe of a brighter accepted pair.

    A fit can place two Gaussians on one spot (e.g. a ring seed drifting onto a lobe);
    pairing the copy with a ring arc or a neighbour's lobe gives a persistent phantom
    next to the true bead. A pair sharing one lobe survives only as a validated
    shared-lobe pair whose own lobe is strong; a pair sharing both lobes never does.
    """
    if len(rows) < 2:
        return rows, 0
    strength = rows[:, [C['amplitude1'], C['amplitude2']]].min(axis=1)
    kept = []
    for i in np.argsort(-strength, kind='stable'):
        mine = np.array([[rows[i, 0], rows[i, 3]], [rows[i, 1], rows[i, 4]]])
        if kept:
            other = np.vstack([np.array([[rows[j, 0], rows[j, 3]], [rows[j, 1], rows[j, 4]]]) for j in kept])
            coincident = (cdist(mine, other) <= distance).any(axis=1)
            if coincident.all():
                continue
            if coincident.any():
                own = rows[i, C['amplitude1'] if coincident[1] else C['amplitude2']]
                if rows[i, C['sharedLobe']] != 1 or own < cfg.shared_min_snr*noise:
                    continue
        kept.append(i)
    kept = sorted(kept)
    return rows[kept], len(rows)-len(kept)


def reject_ring_shadows(rows, cfg):
    """Drop faint pairs made of the side-lobe rings of a much brighter pair.

    The ring arcs of the two lobes of a bright bead lie parallel to it at a similar
    separation, so a ring artefact has one lobe near EACH lobe of the bright pair.
    """
    if len(rows) < 2:
        return rows, 0
    mids = rows[:, [2, 5]]
    tree = cKDTree(mids)
    amp = rows[:, [C['amplitude1'], C['amplitude2']]]
    drop = np.zeros(len(rows), bool)
    for i in range(len(rows)):
        mine = np.array([[rows[i, 0], rows[i, 3]], [rows[i, 1], rows[i, 4]]])
        for j in tree.query_ball_point(mids[i], cfg.ring_shadow_px):
            if j == i or amp[i].max() > cfg.ring_shadow_ratio*amp[j].min():
                continue
            other = np.array([[rows[j, 0], rows[j, 3]], [rows[j, 1], rows[j, 4]]])
            d = min(np.linalg.norm(mine-other, axis=1).max(), np.linalg.norm(mine-other[::-1], axis=1).max())
            if d <= cfg.ring_shadow_px:
                drop[i] = True
    return rows[~drop], int(drop.sum())


def deduplicate(rows, distance=3.):
    """Within each frame keep one localization per position (prefer detected, then higher SNR)."""
    if len(rows) < 2:
        return rows
    keep = np.ones(len(rows), bool)
    order = np.lexsort((-rows[:, C['minLobeSNR']], rows[:, C['recovered']]))
    for frame in np.unique(rows[:, 8]):
        members = order[rows[order, 8] == frame]
        tree = cKDTree(rows[members][:, [2, 5]])
        for rank, i in enumerate(members):
            if not keep[i]:
                continue
            for j in tree.query_ball_point(rows[i, [2, 5]], distance):
                if members[j] != i and keep[members[j]] and rank < np.flatnonzero(members == members[j])[0]:
                    keep[members[j]] = False
    return rows[keep]


def fit_frame(task):
    path, index, cfg, model = task
    return localize_image(read_frame(path, index), cfg, model, frame=index+1, seed=seed_image(path, index, cfg))


FRAME_FIELDS = ['seed_sigma_frames', 'dog_small','dog_large', 'peak_window', 'threshold_sigma', 'weak_fraction', 'threshold', 'local_noise_tile_px', 'local_noise_min_ratio',
                'min_separation', 'max_separation', 'seed_max_ratio', 'max_ratio', 'sep_sd_px', 'sep_gate_px',
                'ratio_log_sd', 'sigma_sd', 'pair_gate_cost', 'center_bound', 'min_sigma', 'max_sigma',
                'fit_padding', 'max_group_gaussians', 'max_nfev', 'min_snr', 'ring_shadow_ratio',
                'ring_shadow_px', 'shared_excess', 'shared_min_snr', 'roi_radius_px', 'split_sigma_ratio',
                'split_min_gain', 'split_max_tries', 'split_reach_sigmas', 'lsmr_min_params', 'fit_solver', 'noise_model', 'camera_file',
                'camera_offset_x', 'camera_offset_y', 'sep_sd_mode', 'sep_sd_floor_px', 'calibration_pruning',
                'prune_slack_px', 'min_pair_snr', 'lobe_floor_snr']


def cache_stamp(path, cfg, model):
    """Identity of a per-frame cache: input file, per-frame fitting settings, calibration."""
    settings = asdict(cfg)
    return {'path': str(path.resolve()), 'size': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns,
            'frame_config': {k: settings[k] for k in FRAME_FIELDS}, 'version': VERSION, 'columns': QUALITY,
            'fit_revision': FIT_REVISION,
            'model': model.fingerprint() if model is not None else None}


def cached_frame_count(path, output, cfg, model, planes=None):
    """How many of the wanted frames already have a valid per-frame fit in localize_stack's cache."""
    manifest = Path(output)/'fit_manifest.json'
    try:
        if not manifest.exists() or json.loads(manifest.read_text()) != cache_stamp(path, cfg, model):
            return 0
    except (OSError, ValueError):
        return 0
    count = frame_count(path)
    wanted = range(count) if planes is None else [i for i in planes if 0 <= i < count]
    return sum((Path(output)/f'frame_{i+1:04d}.npz').exists() for i in wanted)


def localize_stack(path, output, cfg, workers, model, tag, progress=None, planes=None):
    """Fit every frame (or only `planes`, zero-based indices) of a stack, with a per-frame cache."""
    output.mkdir(parents=True, exist_ok=True)
    stamp = cache_stamp(path, cfg, model)
    manifest = output/'fit_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != stamp:
        # Per-frame fits are derived data: rebuild them when the input, fitting settings or
        # calibration changed, instead of silently mixing old and new fits.
        stale = list(output.glob('frame_*.npz'))
        print(f'{tag}: input/settings/calibration changed; refitting {len(stale)} cached frames', flush=True)
        for f in stale:
            f.unlink()
    manifest.write_text(json.dumps(stamp, indent=2))
    count = frame_count(path)
    wanted = list(range(count)) if planes is None else [i for i in planes if 0 <= i < count]
    missing = [i for i in wanted if not (output/f'frame_{i+1:04d}.npz').exists()]
    start = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fit_frame, (str(path), i, cfg, model)) for i in missing]
        for done, future in enumerate(as_completed(futures), 1):
            rows, stats = future.result()
            np.savez_compressed(output/f"frame_{stats['frame']:04d}.npz", rows=rows, stats=json.dumps(stats))
            print(f"{tag}: frame {stats['frame']}/{count}: {len(rows)} pairs; {time.time()-start:.0f}s", flush=True)
            if progress is not None and (done % 4 == 0 or done == len(missing)):
                PROGRESS.update(tag, progress[0]+progress[1]*done/len(missing),
                                f'fitting frames: {len(wanted)-len(missing)+done}/{len(wanted)}')
    arrays, diagnostics = [], []
    for i in wanted:
        with np.load(output/f'frame_{i+1:04d}.npz') as z:
            arrays.append(z['rows'])
            diagnostics.append(json.loads(str(z['stats'])))
    return np.concatenate(arrays), diagnostics


# ----------------------------------------------------------------------------- tracking

def lap(cost, alternative):
    """Jaqaman augmented assignment with explicit birth/death; gated entries are inf."""
    n, m = cost.shape
    finite = np.isfinite(cost)
    if not finite.any():
        return []
    big = 1e12
    c = np.where(finite, cost, big)
    augmented = np.full((n+m, m+n), big)
    augmented[:n, :m] = c
    augmented[np.arange(n), m+np.arange(n)] = alternative
    augmented[n+np.arange(m), np.arange(m)] = alternative
    augmented[n:, m:] = c.T
    r, col = linear_sum_assignment(augmented)
    return [(int(a), int(b)) for a, b in zip(r, col) if a < n and b < m and finite[a, b]]


def axial_difference(a, b):
    return (np.asarray(a)-b+90) % 180-90


def _link_cost(rows, a, b, gate, cfg, z=None, z_sd=None, dt=None):
    cost = cdist(rows[a][:, [2, 5]], rows[b][:, [2, 5]], 'sqeuclidean')
    gated = cost > gate**2
    turn = axial_difference(rows[a, 6][:, None], rows[b, 6][None, :])
    cost = cost + (turn/cfg.angle_cost_deg)**2
    if z is not None:
        # A bead cannot jump in z by much more than its measurement noise between frames.
        dz = np.abs(z[a][:, None]-z[b][None, :])
        tol = np.maximum(cfg.link_z_um, cfg.link_z_sigma*np.hypot(z_sd[a][:, None], z_sd[b][None, :]))
        if dt is not None:
            tol = tol+cfg.link_z_rate_um*np.maximum(dt-1, 0)
        gated |= np.isfinite(dz) & (dz > tol)
    cost[gated] = np.inf
    return cost


def track_depths(rows, model):
    """z and its error bar per row for z-aware linking (NaN where unknown)."""
    if model is None or not len(rows):
        return None, None
    sep = rows[:, C['lobeSeparationPixels']]-model.field_separation(rows[:, 2], rows[:, 5])
    z = model.angle_to_z(rows[:, 6], sep)[0]
    sd = rows[:, C['anglePrecisionDeg']]/np.maximum(np.abs(model.slope(z)), 1e-6)
    sd = np.where(np.isfinite(sd), sd, np.nanmedian(sd) if np.isfinite(sd).any() else 1.)
    return z, sd


def track(rows, cfg, link_distance=None, gap_distance=None, max_frame_gap=None, method=None, model=None):
    """Assign track_number (column 9) with the configured tracker; no split/merge.

    'lap': two-stage LAP (adjacent frames, then global gap closing). Cost is squared XY
    distance plus a small rotation penalty; the non-link alternative is the squared
    gate, so every gated link is preferred to a birth/death pair. With a calibration
    `model`, links that would jump in z beyond the measurement noise are forbidden.
    'kalman': frame-by-frame LAP against constant-velocity Kalman predictions; tracks
    coast through up to max_frame_gap-1 missed frames. 'nearest': greedy nearest neighbour.
    """
    rows = rows.copy()
    if not len(rows):
        return rows
    link_distance = cfg.link_distance_px if link_distance is None else link_distance
    gap_distance = cfg.gap_distance_px if gap_distance is None else gap_distance
    max_frame_gap = cfg.max_frame_gap if max_frame_gap is None else max_frame_gap
    method = cfg.tracker if method is None else method
    if method == 'kalman':
        return _track_kalman(rows, cfg, link_distance, gap_distance, max_frame_gap)
    if method == 'nearest':
        return _track_nearest(rows, cfg, link_distance, gap_distance, max_frame_gap)
    if method != 'lap':
        raise ValueError(f'Unknown tracker {method!r}')
    parent = np.arange(len(rows))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    frames = rows[:, 8].astype(int)
    z, z_sd = track_depths(rows, model)
    by_frame = {f: np.flatnonzero(frames == f) for f in np.unique(frames)}
    for frame in range(int(frames.min()), int(frames.max())):
        a, b = by_frame.get(frame, []), by_frame.get(frame+1, [])
        if len(a) and len(b):
            for i, j in lap(_link_cost(rows, a, b, link_distance, cfg, z, z_sd), link_distance**2):
                parent[root(b[j])] = root(a[i])
    segments = {}
    for i in range(len(rows)):
        segments.setdefault(root(i), []).append(i)
    starts = np.asarray([min(s, key=lambda i: frames[i]) for s in segments.values()])
    ends = np.asarray([max(s, key=lambda i: frames[i]) for s in segments.values()])
    dt = frames[starts][None, :]-frames[ends][:, None]
    if z is not None:
        # compare segment depths (median of the last / first 5 localizations), not single noisy points
        zs, zsd = z.copy(), z_sd.copy()
        for members in segments.values():
            members = sorted(members, key=lambda i: frames[i])
            head, tail = members[:5], members[-5:]
            fin = lambda v: np.median(v[np.isfinite(v)]) if np.isfinite(v).any() else np.nan   # segment may lack z
            zs[members[0]], zs[members[-1]] = fin(z[head]), fin(z[tail])
            zsd[members[0]], zsd[members[-1]] = np.median(z_sd[head]), np.median(z_sd[tail])
        cost = _link_cost(rows, ends, starts, gap_distance, cfg, zs, zsd, dt)
    else:
        cost = _link_cost(rows, ends, starts, gap_distance, cfg)
    # dt == 1 re-joins adjacent-frame breaks made by the per-frame z gate on a single noisy z,
    # now judged on segment depths (and with the adjacent-frame distance gate)
    cost[(dt < 1) | (dt > max_frame_gap) | ((dt == 1) & ((z is None) | (cost > link_distance**2)))] = np.inf
    for i, j in lap(cost, gap_distance**2):
        parent[root(starts[j])] = root(ends[i])
    roots = [root(i) for i in range(len(rows))]
    labels = {v: i+1 for i, v in enumerate(dict.fromkeys(roots))}
    rows[:, 9] = [labels[r] for r in roots]
    return rows


def _gate(misses, link_distance, gap_distance):
    return link_distance if misses == 0 else gap_distance


def _track_kalman(rows, cfg, link_distance, gap_distance, max_frame_gap):
    """Constant-velocity Kalman filter per track, LAP assignment to predicted positions."""
    F = np.eye(4); F[0, 2] = F[1, 3] = 1.
    H = np.eye(2, 4)
    q = cfg.kalman_process_noise**2
    Q = q*np.array([[.25, 0, .5, 0], [0, .25, 0, .5], [.5, 0, 1, 0], [0, .5, 0, 1]])
    R = cfg.kalman_measurement_noise**2*np.eye(2)
    P0 = np.diag([R[0, 0], R[1, 1], cfg.kalman_initial_velocity**2, cfg.kalman_initial_velocity**2])
    frames = rows[:, 8].astype(int)
    labels = np.zeros(len(rows), int)
    active = []   # dicts: x (4), P (4x4), label, misses, angle
    next_label = 1
    for frame in range(int(frames.min()), int(frames.max())+1):
        for t in active:
            t['x'] = F@t['x']; t['P'] = F@t['P']@F.T+Q
        dets = np.flatnonzero(frames == frame)
        matched_t, matched_d = set(), set()
        if len(active) and len(dets):
            pred = np.array([t['x'][:2] for t in active])
            cost = cdist(pred, rows[dets][:, [2, 5]], 'sqeuclidean')
            gates = np.array([_gate(t['misses'], link_distance, gap_distance) for t in active])
            gated = cost > gates[:, None]**2
            turn = axial_difference(np.array([t['angle'] for t in active])[:, None], rows[dets, 6][None, :])
            cost = cost+(turn/cfg.angle_cost_deg)**2
            cost[gated] = np.inf
            for i, j in lap(cost, gap_distance**2):
                t, d = active[i], dets[j]
                S = H@t['P']@H.T+R
                K = t['P']@H.T@np.linalg.inv(S)
                t['x'] = t['x']+K@(rows[d, [2, 5]]-H@t['x'])
                t['P'] = (np.eye(4)-K@H)@t['P']
                t['misses'], t['angle'] = 0, rows[d, 6]
                labels[d] = t['label']
                matched_t.add(i); matched_d.add(j)
        for i, t in enumerate(active):
            if i not in matched_t:
                t['misses'] += 1
        active = [t for t in active if t['misses'] < max_frame_gap]
        for j, d in enumerate(dets):
            if j not in matched_d:
                active.append(dict(x=np.r_[rows[d, [2, 5]], 0., 0.], P=P0.copy(), label=next_label,
                                   misses=0, angle=rows[d, 6]))
                labels[d] = next_label
                next_label += 1
    rows[:, 9] = labels
    return rows


def _track_nearest(rows, cfg, link_distance, gap_distance, max_frame_gap):
    """Greedy nearest-neighbour linking to each track's last position (baseline)."""
    frames = rows[:, 8].astype(int)
    labels = np.zeros(len(rows), int)
    active = []   # dicts: xy, label, misses
    next_label = 1
    for frame in range(int(frames.min()), int(frames.max())+1):
        dets = np.flatnonzero(frames == frame)
        used_t, used_d = set(), set()
        if active and len(dets):
            d2 = cdist(np.array([t['xy'] for t in active]), rows[dets][:, [2, 5]], 'sqeuclidean')
            for flat in np.argsort(d2, axis=None):
                i, j = np.unravel_index(flat, d2.shape)
                if i in used_t or j in used_d:
                    continue
                if d2[i, j] > _gate(active[i]['misses'], link_distance, gap_distance)**2:
                    continue
                active[i]['xy'], active[i]['misses'] = rows[dets[j], [2, 5]], -1
                labels[dets[j]] = active[i]['label']
                used_t.add(i); used_d.add(j)
        for t in active:
            t['misses'] += 1
        active = [t for t in active if t['misses'] < max_frame_gap]
        for j, d in enumerate(dets):
            if j not in used_d:
                active.append(dict(xy=rows[d, [2, 5]], label=next_label, misses=0))
                labels[d] = next_label
                next_label += 1
    rows[:, 9] = labels
    return rows


# ----------------------------------------------------------------------------- recovery

def _aligned(l0, l1):
    """Reorder lobes l1 (2x2) to best match l0 (labels can exchange at the axial wrap)."""
    direct = np.linalg.norm(l0-l1, axis=1).sum()
    return l1 if direct <= np.linalg.norm(l0-l1[::-1], axis=1).sum() else l1[::-1]


def recovery_requests(rows, cfg, nframes):
    """Predicted lobe positions (zero-based) for frames a persistent track is missing."""
    frames = rows[:, 8].astype(int)
    tracks = {}
    for i, t in enumerate(rows[:, 9].astype(int)):
        tracks.setdefault(t, {})[frames[i]] = i
    motion = {}
    def local_shift(g, f, xy):
        key = (g, f)
        if key not in motion:
            common = [(v[g], v[f]) for v in tracks.values() if g in v and f in v]
            if common:
                ig, jf = np.array(common).T
                motion[key] = (cKDTree(rows[ig][:, [2, 5]]), rows[jf][:, [2, 5]]-rows[ig][:, [2, 5]])
            else:
                motion[key] = None
        if motion[key] is None:
            return np.zeros(2)
        tree, disp = motion[key]
        near = tree.query_ball_point(xy, 200.)
        if len(near) < 3:
            near = tree.query(xy, k=min(5, len(disp)))[1]
        return np.median(disp[np.atleast_1d(near)], axis=0)
    requests = {}
    for t, members in tracks.items():
        if len(members) < cfg.recover_min_track:
            continue
        observed = np.array(sorted(members))
        detected = [i for i in members.values() if rows[i, C['recovered']] != 1] or list(members.values())
        reference = float(np.median(rows[detected][:, [C['amplitude1'], C['amplitude2']]].min(axis=1)))
        for f in range(1, nframes+1):
            if f in members or np.min(np.abs(observed-f)) > cfg.recover_reach_frames:
                continue
            before, after = observed[observed < f], observed[observed > f]
            rb = rows[members[before[-1]]] if len(before) else None
            ra = rows[members[after[0]]] if len(after) else None
            if rb is not None and ra is not None:
                lb = np.array([[rb[0], rb[3]], [rb[1], rb[4]]])
                la = _aligned(lb, np.array([[ra[0], ra[3]], [ra[1], ra[4]]]))
                w = (f-before[-1])/(after[0]-before[-1])
                lobes = (1-w)*lb+w*la
            else:
                g, r = (before[-1], rb) if rb is not None else (after[0], ra)
                lobes = np.array([[r[0], r[3]], [r[1], r[4]]]) + local_shift(g, f, r[[2, 5]])
            requests.setdefault(f, []).append((t, lobes-1, reference))
    return requests


def recover_frame(task):
    """Refit predicted beads in one frame with the prediction as prior. Returns new rows."""
    path, index, cfg, model, requests, existing = task
    image = read_frame(path, index)
    noise = pixel_noise(image)
    xy, values, strong = find_peaks(image, cfg)
    xy = xy[values >= cfg.weak_fraction*strong*1.5]  # nuisance seeds only; the prediction supplies the bead
    existing_lobes = np.vstack((existing[:, [0, 3]], existing[:, [1, 4]]))-1 if len(existing) else np.empty((0, 2))
    existing_mid = existing[:, [2, 5]]-1 if len(existing) else np.empty((0, 2))
    out = []
    for track_id, lobes, reference in requests:
        predicted_angle = pair_geometry(np.r_[1, lobes[0], 2], np.r_[1, lobes[1], 2])[1]
        lo = np.maximum(np.floor(lobes.min(axis=0)-cfg.fit_padding), 0).astype(int)
        hi = np.minimum(np.ceil(lobes.max(axis=0)+cfg.fit_padding), [image.shape[1]-1, image.shape[0]-1]).astype(int)
        if np.any(hi-lo < 4):
            continue
        box = np.r_[lo, hi]
        others = existing_lobes[_inside(existing_lobes, box)]
        # A predicted lobe on top of an existing lobe is a shared (coincident) lobe: the
        # existing lobe is its fitted centroid, so do not add a degenerate second Gaussian.
        shared = np.zeros(2, bool)
        if len(others):
            dist = cdist(lobes, others)
            shared = dist.min(axis=1) <= 3
            others = others[dist.min(axis=0) > 3]
        peaks = xy[_inside(xy, box)]
        known = np.vstack((lobes, others))
        peaks = peaks[np.min(cdist(peaks, known), axis=1) > 3] if len(peaks) else peaks
        seeds = np.vstack((lobes, others, peaks))
        if len(seeds) > 2*cfg.max_group_gaussians or shared.all():
            continue
        bounds = np.r_[[cfg.recover_center_bound]*2, [cfg.center_bound]*(len(seeds)-2)]
        fit = fit_gaussians(image, lo, hi, seeds, bounds, cfg)
        if fit is None:
            continue
        params, at_bound, rms, background, cov = fit
        p1, p2 = params[0].copy(), params[1].copy()
        own = [p for p, s in zip((p1, p2), shared) if not s]
        floor = max(cfg.recover_min_snr*noise, cfg.recover_amplitude_fraction*reference)
        if shared.any():
            floor = max(floor, cfg.shared_min_snr*noise)
        if at_bound[0] or at_bound[1] or min(p[0] for p in own) < floor:
            continue
        if abs(axial_difference(pair_geometry(p1, p2)[1], predicted_angle)) > cfg.recover_max_turn_deg:
            continue
        if not all(0 <= p[1] <= image.shape[1]-1 and 0 <= p[2] <= image.shape[0]-1 for p in (p1, p2)):
            continue
        if shared.any():  # the hidden lobe's own amplitude is unobservable
            (p2 if shared[1] else p1)[0] = own[0][0]
        cost, residual = pair_cost(p1, p2, cfg, model, noise)
        if not cost <= cfg.pair_gate_cost:
            continue
        mid = (p1[1:3]+p2[1:3])/2
        if len(existing_mid) and np.min(np.linalg.norm(existing_mid-mid, axis=1)) < 4:
            continue
        row = make_row(p1, p2, index+1, rms, min(p[0] for p in own)/noise, 1, cost, residual, 1,
                       pair_precision(params, cov, 0, 1))
        row[9] = track_id
        row[C['sharedLobe']] = float(shared.any())
        shared_xy = (p2 if shared[1] else p1)[1:3] if shared.any() else None
        out.append(add_signals(row, image, background, cfg, shared_xy))
    return np.asarray(out).reshape(-1, NCOL)


def on_trend(t, v, tk, vk, cut, noise):
    """Does a point agree with a constant-velocity motion of its neighbours?

    Fits a straight line in time (constant velocity) to the neighbours (t, v): the estimate a Kalman
    smoother with a constant-velocity model gives over this window. If the neighbours lie on it (scatter
    within max(3 x noise, cut/3)) and the point is within `cut` of it, the point is part of a smooth
    motion (a bead being pushed, starting or stopping) and is not an outlier however far it is from the
    neighbours' median. Neighbours that do not fit a line (a bead flipping between two depths, or a
    short excursion and back) give False, and the median / majority rules decide as before.
    """
    t, v = np.asarray(t, float), np.asarray(v, float)
    ok = np.isfinite(v)
    t, v = t[ok], v[ok]
    if len(v) < 4 or not np.isfinite(vk) or np.ptp(t) == 0:
        return False
    noise = noise if np.isfinite(noise) else 0.
    tol = max(3*noise, cut/3)
    # robust: neighbours that are themselves off the line (e.g. a wrong-branch z at the edge of the
    # calibrated range) are set aside, as long as most of them (>= 4 and >= 2/3) are on it
    # start from a Theil-Sen line (median of the pairwise slopes), which one or two bad points cannot
    # tilt, then refine by least squares on the points near it
    i, j = np.triu_indices(len(t), 1)
    dt = t[j]-t[i]
    slope = np.median((v[j]-v[i])[dt != 0]/dt[dt != 0])
    coef = np.array([np.median(v-slope*(t-tk)), slope])
    keep = None
    for _ in range(3):
        res = np.abs(v-(coef[0]+coef[1]*(t-tk)))
        new = res <= max(cut, 3*tol)
        if new.sum() < max(4, int(np.ceil(2*len(v)/3))) or np.ptp(t[new]) == 0:
            return False
        if keep is not None and np.array_equal(new, keep):
            break
        keep = new
        A = np.column_stack((np.ones(keep.sum()), t[keep]-tk))
        coef = np.linalg.lstsq(A, v[keep], rcond=None)[0]
    res = np.abs(v-(coef[0]+coef[1]*(t-tk)))
    scatter = np.sqrt(np.sum(res[keep]**2)/max(keep.sum()-2, 1))
    return bool(scatter <= tol and abs(vk-coef[0]) <= cut)


def jump_outliers(rows, cfg, model=None):
    """Localizations that jump away from their own track and back (e.g. a flipped pairing).

    Each localization is compared with the median position, angle and (with a calibration
    model) z of the same track in the surrounding +-jump_window frames. It is an outlier if
    it lies more than jump_px (plus twice the neighbours' own spread) from them, is rotated
    more than jump_angle_deg, or its z differs by more than the z cut. The z test matters
    where lobes rotate slowly with depth (10x: ~0.7 deg/um), so a flipped pairing changes
    the angle little but z a lot. The window is centred, so a track's first frames are judged
    against the frames after them. The z cut follows the track's own noise:
    max(jump_z_um, jump_z_sigma x its frame-to-frame z noise) (3 x the local scatter for short tracks),
    so a 4 µm spike is caught on a quiet 5 ms bead (noise 0.7 µm) but not on a 1 ms one.
    The most outlying localization is removed first and the rest judged again without it, until none
    is left, so alternating stretches resolve to the majority state and a wrong point does not take
    its good neighbours with it. A localization on the constant-velocity line through its neighbours
    (on_trend: a bead really moving) is never an outlier.
    """
    bad = np.zeros(len(rows), bool)
    z = row_depths(rows, model)
    order = np.lexsort((rows[:, 8], rows[:, 9]))
    tracks = np.split(order, np.flatnonzero(np.diff(rows[order, 9]))+1)
    for track_rows in tracks:
        # worst first: remove the most outlying localization, then judge the others again without it,
        # so a wrong point does not drag its neighbours' reference and take good frames with it
        for _ in range(len(track_rows)):
            idx = track_rows[~bad[track_rows]]
            if len(idx) < 4:
                break
            f, xy, angle = rows[idx, 8], rows[idx][:, [2, 5]], rows[idx, 6]
            # the track's single-localization z noise (robust, from frame-to-frame steps)
            steps = np.diff(z[idx])
            steps = steps[np.isfinite(steps)]
            track_sd = (1.4826*np.median(np.abs(steps-np.median(steps)))/np.sqrt(2)
                        if len(steps) >= 6 else np.inf)
            worst, worst_score = None, 1.
            for k in range(len(idx)):
                near = np.flatnonzero((np.abs(f-f[k]) <= cfg.jump_window) & (np.arange(len(idx)) != k))
                if len(near) < 2:
                    continue
                ref = np.median(xy[near], axis=0)
                spread = np.median(np.linalg.norm(xy[near]-ref, axis=1))
                turn = np.median(axial_difference(angle[near], angle[k]))
                zn = z[idx[near]]
                zn = zn[np.isfinite(zn)]
                dz, z_cut = 0., cfg.jump_z_um
                if len(zn) >= 2 and np.isfinite(z[idx[k]]):
                    dz = abs(z[idx[k]]-np.median(zn))
                    # noisy (dim) tracks scatter in z by themselves: scale the cut with that noise
                    # (the track's, from ~all its frames; a +-5-frame window is too few to estimate it)
                    z_cut = max(cfg.jump_z_um, cfg.jump_z_sigma*track_sd) if np.isfinite(track_sd) else \
                        max(cfg.jump_z_um, 3*1.4826*np.median(np.abs(zn-np.median(zn))))
                xy_cut = cfg.jump_px+2*spread
                xy_off = np.linalg.norm(xy[k]-ref)
                if xy_off <= xy_cut and abs(turn) <= cfg.jump_angle_deg and dz <= z_cut:
                    continue
                # far from the neighbours' median, and not on the constant-velocity line through them
                # (a bead that is really moving, e.g. pushed by the indenter, stays on that line). The line
                # uses the 6 nearest frames of the track (up to 3 windows away), so a track's last frames,
                # or a track split into interleaved pieces, still have enough of them.
                others = np.flatnonzero((np.arange(len(idx)) != k) & (np.abs(f-f[k]) <= 3*cfg.jump_window))
                tn = others[np.argsort(np.abs(f[others]-f[k]), kind='stable')[:6]]
                score = 0.
                if xy_off > xy_cut and not (on_trend(f[tn], xy[tn, 0], f[k], xy[k, 0], cfg.jump_px, .1)
                                            and on_trend(f[tn], xy[tn, 1], f[k], xy[k, 1], cfg.jump_px, .1)):
                    score = max(score, xy_off/xy_cut)
                if abs(turn) > cfg.jump_angle_deg and not \
                        on_trend(f[tn], axial_difference(angle[tn], angle[k]), f[k], 0., cfg.jump_angle_deg, 1.):
                    score = max(score, abs(turn)/cfg.jump_angle_deg)
                if dz > z_cut and not on_trend(f[tn], z[idx[tn]], f[k], z[idx[k]], z_cut, track_sd):
                    score = max(score, dz/z_cut)
                if score > worst_score:
                    worst, worst_score = k, score
            if worst is None:
                break
            bad[idx[worst]] = True
    return bad


def row_depths(rows, model):
    if model is None or not len(rows):
        return np.full(len(rows), np.nan)
    sep = rows[:, C['lobeSeparationPixels']]-model.field_separation(rows[:, 2], rows[:, 5])
    return model.angle_to_z(rows[:, 6], sep)[0]


def z_jitter(rows, z):
    """Single-localization z noise (µm) of this movie, from frame-to-frame steps within tracks."""
    order = np.lexsort((rows[:, 8], rows[:, 9]))
    r, zz = rows[order], z[order]
    step = (np.diff(r[:, 9]) == 0) & (np.diff(r[:, 8]) == 1)
    d = np.diff(zz)[step]
    d = d[np.isfinite(d)]
    return 1.4826*np.median(np.abs(d-np.median(d)))/np.sqrt(2) if len(d) >= 20 else np.nan


def alternating_states(rows, cfg, model=None, twin_px=3., z=None):
    """Localizations in the minority of a bead whose fit flips between two depths.

    Crowded beads (a lobe overlapping another bead's lobe, or chains of lobes where two
    pairings fit) can be fitted in two states that differ by several µm in z, alternating
    from frame to frame. The per-frame z gate then splits them into interleaved tracks, so
    the per-track jump test cannot see it. Here the reference is everything localized at the
    same place (centre within twin_px) in the surrounding +-jump_window frames, in any track.
    Its dominant z state is the densest cluster (not the median, which falls between two
    states); a localization farther than max(state_z_um, 4 x the movie's z noise) from it and
    in a smaller cluster is flagged. Places with two beads present in the same frame are skipped.
    """
    bad = np.zeros(len(rows), bool)
    z = row_depths(rows, model) if z is None else z
    ok = np.isfinite(z)
    if ok.sum() < 10:
        return bad
    sd = z_jitter(rows, z)
    cut = max(cfg.state_z_um, 4*sd) if np.isfinite(sd) else cfg.state_z_um
    frames = rows[:, 8].astype(int)
    tree = cKDTree(rows[:, [2, 5]])
    for i in np.flatnonzero(ok):
        near = np.asarray(tree.query_ball_point(rows[i, [2, 5]], twin_px))
        if np.any((frames[near] == frames[i]) & (near != i)):
            continue
        near = near[(np.abs(frames[near]-frames[i]) <= cfg.jump_window) & (frames[near] != frames[i]) & ok[near]]
        if len(near) < 4:
            continue
        zn = z[near]
        support = (np.abs(zn[:, None]-zn[None, :]) <= cut/2).sum(axis=1)
        best = np.argmax(support)
        mode = np.mean(zn[np.abs(zn-zn[best]) <= cut/2])
        own = np.sum(np.abs(zn-z[i]) <= cut/2)
        if abs(z[i]-mode) > cut and support[best] >= 3 and own < support[best] and \
                not on_trend(frames[near], zn, frames[i], z[i], cut, sd):   # a smooth push, not two states
            bad[i] = True
    return bad


def sandwiched_tracks(rows, cfg, model=None, twin_px=3., z=None):
    """Track ids that sit, at the same place, between two tracks at another depth.

    The segment-scale version of a jump outlier: a crowded bead whose fit settles in a second
    state for a while is split by the z gate into before / during / after tracks. If the
    tracks just before and just after (within 2 x max_frame_gap frames, centres within
    twin_px of its ends) agree with each other in z but the middle one differs from both by
    more than max(state_z_um, 4 x the movie's z noise), the middle one is the wrong state.
    """
    z = row_depths(rows, model) if z is None else z
    if not len(rows) or not np.isfinite(z).any():
        return set()
    sd = z_jitter(rows, z)
    cut = max(cfg.state_z_um, 4*sd) if np.isfinite(sd) else cfg.state_z_um
    frames = rows[:, 8].astype(int)
    ends = {}
    for t in np.unique(rows[:, 9]):
        idx = np.flatnonzero(rows[:, 9] == t)
        idx = idx[np.argsort(frames[idx])]
        head, tail = idx[:5], idx[-5:]
        ends[t] = dict(first=frames[idx[0]], last=frames[idx[-1]], n=len(idx),
                       head_xy=np.median(rows[head][:, [2, 5]], axis=0), tail_xy=np.median(rows[tail][:, [2, 5]], axis=0),
                       head_z=np.nanmedian(z[head]) if np.isfinite(z[head]).any() else np.nan,
                       tail_z=np.nanmedian(z[tail]) if np.isfinite(z[tail]).any() else np.nan)
    reach = 2*cfg.max_frame_gap
    out = set()
    for t, e in ends.items():
        before = [u for u, o in ends.items() if u != t and 0 < e['first']-o['last'] <= reach
                  and np.linalg.norm(o['tail_xy']-e['head_xy']) <= twin_px]
        after = [u for u, o in ends.items() if u != t and 0 < o['first']-e['last'] <= reach
                 and np.linalg.norm(o['head_xy']-e['tail_xy']) <= twin_px]
        for a in before:
            for b in after:
                za, zb = ends[a]['tail_z'], ends[b]['head_z']
                if (abs(za-zb) <= cut and abs(e['head_z']-za) > cut and abs(e['tail_z']-zb) > cut
                        and e['n'] < ends[a]['n']+ends[b]['n']):
                    out.add(t)
    return out


def ghost_tracks(rows, cfg, z=None, window=8, local_px=150.):
    """Track ids whose 'bead' is assembled from other beads' lobes or side lobes.

    A ghost's two lobes both sit (within ghost_lobe_px) on lobes of OTHER tracks at nearby
    times (+-window frames) in at least ghost_fraction of its frames, and either it is much
    dimmer than those lenders (<= ghost_dim_ratio: side-lobe ghosts) or its z is more than
    ghost_z_um from theirs while also being the outlier relative to the tracks around it
    (crossed pairings of two neighbouring beads). A real bead in a three-spot overlap
    borrows at most one lobe and is not affected.
    """
    if not len(rows):
        return set()
    z = rows[:, 7] if z is None else z
    frames = rows[:, 8].astype(int)
    tid = rows[:, 9].astype(int)
    amp = rows[:, [C['amplitude1'], C['amplitude2']]].min(axis=1)
    lobes = np.stack((rows[:, [0, 3]], rows[:, [1, 4]]), axis=1)          # (n, 2, 2)
    ids = np.unique(tid)
    centre = {t: np.median(rows[tid == t][:, [2, 5]], axis=0) for t in ids}
    tz = {t: np.nanmedian(z[tid == t]) if np.isfinite(z[tid == t]).any() else np.nan for t in ids}
    stats = {t: [0, 0, [], [], []] for t in ids}   # frames, both-borrowed, own amp, lender amp, lender z
    for f in np.unique(frames):
        near = np.flatnonzero(np.abs(frames-f) <= window)
        pts = lobes[near].reshape(-1, 2)
        owner = np.repeat(near, 2)
        tree = cKDTree(pts)
        for i in np.flatnonzero(frames == f):
            lenders = []
            for k in (0, 1):
                cand = [owner[j] for j in tree.query_ball_point(lobes[i, k], cfg.ghost_lobe_px) if tid[owner[j]] != tid[i]]
                if not cand:
                    break
                # the lender's localization nearest in time
                lenders.append(min(cand, key=lambda j: abs(frames[j]-f)))
            s = stats[tid[i]]
            s[0] += 1
            if len(lenders) == 2:
                s[1] += 1
                s[2].append(amp[i]); s[3].append(np.mean(amp[lenders])); s[4].append(np.nanmean(z[lenders]))
    ghosts = set()
    tree = cKDTree(np.array([centre[t] for t in ids]))
    for t in ids:
        n, borrowed, own, lend, lz = stats[t]
        if n < 1 or borrowed < cfg.ghost_fraction*n:
            continue
        dim = np.median(np.asarray(own)/np.maximum(lend, 1e-9)) <= cfg.ghost_dim_ratio
        zfar = False
        if np.isfinite(tz[t]) and np.isfinite(lz).any() and abs(tz[t]-np.nanmedian(lz)) > cfg.ghost_z_um:
            around = [tz[ids[j]] for j in tree.query_ball_point(centre[t], local_px) if ids[j] != t and np.isfinite(tz[ids[j]])]
            if around:
                local = np.median(around)
                zfar = abs(tz[t]-local) > abs(np.nanmedian(lz)-local)
        if dim or zfar:
            ghosts.add(t)
    return ghosts


def recover(rows, path, cfg, model, workers, tag, progress=None):
    """Iterate jump removal, track-guided recovery and re-tracking until nothing changes.

    Returns (rows, history, removed jump outliers). progress: (output, start, span) for the status file.
    """
    nframes = frame_count(path)
    rows = track(rows, cfg, model=model)
    removed = []
    history = []
    for round_ in range(cfg.recover_rounds):
        if progress:
            out, start, span = progress
            PROGRESS.update(tag, start+span*round_/cfg.recover_rounds,
                            f'checking tracks and refitting missed frames (round {round_+1})', output=out)
        bad = jump_outliers(rows, cfg, model) | alternating_states(rows, cfg, model)
        bad |= np.isin(rows[:, 9], list(sandwiched_tracks(rows, cfg, model)))
        # ghosts (pairs assembled from other beads' lobes / side lobes) free their lobes, so the
        # real beads can be refitted from their own tracks below
        z, _ = track_depths(rows, model)
        ghosts = ghost_tracks(rows, cfg, z)
        bad |= np.isin(rows[:, 9], list(ghosts))
        if bad.any():
            removed.append(rows[bad])
            rows = rows[~bad]
            print(f'{tag}: removed {int(bad.sum())} jump outliers / ghost localizations ({len(ghosts)} ghost tracks); '
                  f'refitting from their tracks below', flush=True)
        requests = recovery_requests(rows, cfg, nframes)
        tasks = [(str(path), f-1, cfg, model, reqs, rows[rows[:, 8] == f]) for f, reqs in sorted(requests.items())]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            new = [r for r in pool.map(recover_frame, tasks) if len(r)]
        added = np.concatenate(new) if new else np.empty((0, NCOL))
        if len(added) and removed:
            # Never re-add what the jump test just removed (same frame, same place and angle);
            # otherwise noisy localizations cycle between recovery and removal.
            gone = np.vstack(removed)
            keep = np.ones(len(added), bool)
            for f in np.unique(added[:, 8]):
                g = gone[gone[:, 8] == f]
                if not len(g):
                    continue
                sel = np.flatnonzero(added[:, 8] == f)
                d, j = cKDTree(g[:, [2, 5]]).query(added[sel][:, [2, 5]])
                same = (d < 1.) & (np.abs(axial_difference(added[sel, 6], g[j, 6])) < 3.)
                keep[sel[same]] = False
            added = added[keep]
        n_req = sum(len(v) for v in requests.values())
        history.append(dict(round=round_+1, requests=n_req, recovered=len(added), jump_outliers=int(bad.sum())))
        print(f'{tag}: recovery round {round_+1}: {len(added)}/{n_req} predicted beads recovered', flush=True)
        if not len(added) and not bad.any():
            break
        rows = track(deduplicate(np.vstack((rows, added))), cfg, model=model)
    removed = np.vstack(removed) if removed else np.empty((0, NCOL))
    return rows, history, removed


def estimate_drift(rows, nframes, z_smooth=0.):
    """Rigid per-frame shake (dx, dy px; dz µm) relative to its median over the movie.

    Referenced to the median rather than to frame 1: frame 1 is a single sample of the shake (in
    the 10x 1 ms movie it lies at the 8th percentile of x), so a frame-1 reference offsets the whole
    curve and puts stabilized positions at one shaken frame instead of each bead's typical position.
    The reference is a constant, so displacements are the same either way.

    Robust two-way model per coordinate: position(bead b, frame f) = c_b + D(f) + noise,
    solved by median polish (alternating medians over frames and over beads). D is thus
    estimated directly in every frame from all beads -- its error does not accumulate
    over the movie, unlike summing median frame-to-frame steps (a random walk of about
    sigma*sqrt(N*pi/3M) after N frames with M beads). Medians ignore local deformation
    affecting a minority of beads, but a deformation shared by most of the field would be
    absorbed too, so stabilized coordinates are an optional view. The stage shakes
    laterally from frame to frame, but focus (z) drifts smoothly, so with z_smooth > 0
    D_z is also Gaussian-smoothed over time (sigma in frames).
    """
    drift = _direct_drift(rows, nframes)
    if z_smooth > 0 and nframes > 1:
        drift[:, 2] = gaussian_filter1d(drift[:, 2], z_smooth, mode='nearest')
        drift[:, 2] -= np.median(drift[:, 2])
    return drift


def _direct_drift(rows, nframes, iterations=6):
    drift = np.zeros((nframes, 3))
    if not len(rows):
        return drift
    frames = rows[:, 8].astype(int)-1
    ok = (frames >= 0) & (frames < nframes)
    tracks = np.unique(rows[ok, 9], return_inverse=True)[1]
    frames = frames[ok]
    for k, col in enumerate((2, 5, 7)):
        v = rows[ok, col]
        good = np.isfinite(v)
        f, t, v = frames[good], tracks[good], v[good]
        if not len(v):
            continue
        D = np.zeros(nframes)
        c = np.zeros(tracks.max()+1)
        for _ in range(iterations):
            c = _group_median(v-D[f], t, len(c))
            D = _group_median(v-c[t], f, nframes)
        # frames without beads keep the drift of the nearest frame that has some
        have = np.isfinite(D)
        if not have.any():
            continue
        D = np.interp(np.arange(nframes), np.flatnonzero(have), D[have])
        drift[:, k] = D-np.median(D)          # reference: the movie's median position (see estimate_drift)
    return drift


def _group_median(values, groups, n):
    """Median of `values` per group index 0..n-1 (NaN for empty groups)."""
    out = np.full(n, np.nan)
    order = np.argsort(groups, kind='stable')
    g, v = groups[order], values[order]
    starts = np.flatnonzero(np.r_[True, np.diff(g) != 0])
    for s, e in zip(starts, np.r_[starts[1:], len(g)]):
        out[g[s]] = np.median(v[s:e])
    return out


REJECT_REASONS = {1: 'transient: track shorter than min_track_length',
                  2: 'low quality: track median lobe SNR and template score both below thresholds',
                  3: 'satellite: follows a much brighter bead at the same angle within a few px (ring/phantom copy)',
                  4: 'removed during recovery: a jump outlier (jumped away from its own track and back, e.g. a '
                     'flipped lobe pairing), the minority state of a bead whose fit alternates between two depths, '
                     'or a ghost (see 5); the frame was refitted from the real track where possible',
                  5: 'ghost: both lobes borrowed from other beads (side lobes or a crossed pairing of two beads), '
                     'much dimmer than the lenders or at an implausible z'}


def satellite_tracks(rows, cfg):
    """Track ids that shadow a much brighter track: close, parallel and dimmer in most frames."""
    frames = rows[:, 8].astype(int)
    strength = rows[:, [C['amplitude1'], C['amplitude2']]].min(axis=1)
    ids = np.unique(rows[:, 9])
    median_strength = {t: np.median(strength[rows[:, 9] == t]) for t in ids}
    votes = {t: [0, 0] for t in ids}   # [shadowed frames, frames]
    for f in np.unique(frames):
        idx = np.flatnonzero(frames == f)
        tree = cKDTree(rows[idx][:, [2, 5]])
        for k, i in enumerate(idx):
            t = rows[i, 9]
            votes[t][1] += 1
            for j in tree.query_ball_point(rows[i, [2, 5]], cfg.satellite_distance_px):
                o = idx[j]
                if o != i and median_strength[rows[o, 9]] >= cfg.satellite_brightness_ratio*median_strength[t] and \
                        abs(axial_difference(rows[i, 6], rows[o, 6])) <= cfg.satellite_max_turn_deg:
                    votes[t][0] += 1
                    break
    return [t for t, (shadowed, total) in votes.items() if total and shadowed >= cfg.satellite_fraction*total]


def split_transient(rows, cfg):
    """Separate likely-noise tracks. Returns (kept, rejected, reason per rejected row).

    Transient: fewer than min_track_length frames. Low quality (needs templateScore):
    median lobe SNR < reject_track_snr AND median template score < reject_track_template
    over the track's independently detected frames; real beads sit far from that corner.
    """
    if not len(rows):
        return rows, rows, np.zeros(0, int)
    ids, counts = np.unique(rows[:, 9], return_counts=True)
    reason = np.zeros(len(rows), int)
    reason[np.isin(rows[:, 9], ids[counts < cfg.min_track_length])] = 1
    if np.isfinite(rows[:, C['templateScore']]).any():
        track_snr = {}
        for t in ids:
            member = rows[:, 9] == t
            detected = member & (rows[:, C['recovered']] != 1)
            use = detected if detected.any() else member
            track_snr[t] = (member, use, np.median(rows[use, C['minLobeSNR']]))
        # SNR cut relative to this movie's typical track, so dim acquisitions keep their beads.
        typical = np.median([v[2] for v in track_snr.values()])
        snr_cut = min(cfg.reject_track_snr, cfg.reject_track_snr_fraction*typical)
        for t, (member, use, snr) in track_snr.items():
            if snr < snr_cut and np.nanmedian(rows[use, C['templateScore']]) < cfg.reject_track_template:
                reason[member & (reason == 0)] = 2
    candidates = reason == 0
    if candidates.any():
        for t in ghost_tracks(rows[candidates], cfg):
            reason[(rows[:, 9] == t) & (reason == 0)] = 5
    candidates = reason == 0
    if candidates.any():
        for t in satellite_tracks(rows[candidates], cfg):
            reason[(rows[:, 9] == t) & (reason == 0)] = 3
    keep, rejected = rows[reason == 0], rows[reason > 0]
    # Renumber the kept tracks 1..N in order of first appearance.
    order = dict.fromkeys(keep[np.lexsort((keep[:, 2], keep[:, 8]))][:, 9])
    labels = {t: i+1 for i, t in enumerate(order)}
    keep = keep.copy()
    keep[:, 9] = [labels[t] for t in keep[:, 9]]
    return keep, rejected, reason[reason > 0]


# ----------------------------------------------------------------------------- calibration

class CalibrationModel:
    """Angle->z inverse and separation-vs-z curve from the calibration stack."""

    def __init__(self, dense_z, dense_a, dense_sep, z_limits=None):
        self.dense_z, self.dense_a, self.dense_sep = map(np.asarray, (dense_z, dense_a, dense_sep))
        # z is reported only inside z_limits; the full curve still serves pairing.
        self.z_limits = (float(self.dense_z[0]), float(self.dense_z[-1])) if z_limits is None else tuple(map(float, z_limits))
        grid = np.arange(0, 180, .05)
        roots = [self.roots(a) for a in grid]
        width = max(1, max(len(r) for r in roots))
        self._grid_sep = np.full((len(grid), width), np.nan)
        for i, r in enumerate(roots):
            self._grid_sep[i, :len(r)] = np.interp(r, self.dense_z, self.dense_sep)
        self.field_coef, self.field_offset_px, self.sensor = None, (0., 0.), 2048

    def with_field(self, coef, offset, sensor):
        """Copy whose expected separation includes a movie-specific field map (see estimate_field_separation)."""
        import copy
        other = copy.copy(self)
        other.field_coef, other.field_offset_px, other.sensor = np.asarray(coef, float), tuple(offset), sensor
        return other

    def slope(self, z):
        """Calibrated rotation rate d(angle)/dz in degrees per µm at z."""
        rate = np.gradient(self.dense_a, self.dense_z)
        return np.interp(z, self.dense_z, rate)

    def field_separation(self, x, y):
        """Separation offset (px) at one-based image position(s) x, y; 0 without a field map."""
        if self.field_coef is None:
            return np.zeros(np.shape(x)) if np.ndim(x) else 0.
        value = _field_design(x, y, self.field_offset_px, self.sensor)@self.field_coef
        return value if np.ndim(x) else float(value[0])

    def fingerprint(self):
        # In stack coordinates (z_stack = -z_lab, see to_lab_z): identical to the fingerprint of the
        # same calibration before z was reported as lab z, so per-frame caches (which hold no z) stay valid.
        return [float(-self.dense_z[-1]), float(-self.dense_z[0]), float(np.sum(self.dense_a[::-1])),
                float(np.sum(self.dense_sep[::-1])), -self.z_limits[1], -self.z_limits[0]] + \
               ([] if self.field_coef is None else self.field_coef.round(6).tolist())

    def roots(self, angle):
        a, z = self.dense_a, self.dense_z
        found = []
        for k in range(int(np.floor((a.min()-angle)/180)), int(np.ceil((a.max()-angle)/180))+1):
            target = angle+180*k
            for j in np.flatnonzero((np.minimum(a[:-1], a[1:]) <= target) & (target <= np.maximum(a[:-1], a[1:]))):
                found.append(z[j] if abs(a[j+1]-a[j]) < 1e-10 else z[j]+(target-a[j])/(a[j+1]-a[j])*(z[j+1]-z[j]))
        unique = []
        for r in sorted(found):
            if not unique or abs(r-unique[-1]) > 1e-6:
                unique.append(r)
        return unique

    def separation_residual(self, angle, sep, x=None, y=None):
        """Observed minus expected separation at the best-matching inverse (nan if none).

        With a field map and a one-based position (x, y), the expectation includes the
        movie's field-dependent separation offset.
        """
        predicted = self._grid_sep[int(round(angle/.05)) % len(self._grid_sep)]
        predicted = predicted[np.isfinite(predicted)]
        if not len(predicted):
            return np.nan
        if x is not None:
            sep = sep-self.field_separation(x, y)
        return float((sep-predicted)[np.argmin(np.abs(sep-predicted))])

    def separation_residuals(self, angles, seps, x=None, y=None):
        """Vectorized separation_residual for arrays (nan where the angle has no inverse)."""
        angles, seps = np.asarray(angles, float), np.asarray(seps, float)
        if not len(angles):
            return np.zeros(0)
        predicted = self._grid_sep[np.round(angles/.05).astype(int) % len(self._grid_sep)]
        if x is not None:
            seps = seps-self.field_separation(np.asarray(x, float), np.asarray(y, float))
        diff = seps[:, None]-predicted
        best = np.argmin(np.where(np.isfinite(diff), np.abs(diff), np.inf), axis=1)
        out = diff[np.arange(len(diff)), best]
        return np.where(np.isfinite(predicted).any(axis=1), out, np.nan)

    def angle_to_z(self, angles, seps=None, sep_margin=1.5):
        """z per angle. Status 0 unique, 1 out of range, 2 ambiguous, 3 resolved by separation."""
        z = np.full(len(angles), np.nan)
        status = np.ones(len(angles), dtype=int)
        lo, hi = self.z_limits
        for i, angle in enumerate(angles):
            r = [v for v in self.roots(angle) if lo <= v <= hi]
            if len(r) == 1:
                z[i], status[i] = r[0], 0
            elif len(r) > 1:
                status[i] = 2
                if seps is not None:
                    res = np.abs(seps[i]-np.interp(r, self.dense_z, self.dense_sep))
                    best = np.argsort(res)
                    if res[best[1]]-res[best[0]] >= sep_margin:
                        z[i], status[i] = r[best[0]], 3
        return z, status


def _smooth_curve(z, y, scatter, support):
    """Penalized cubic smoothing spline through per-plane medians y(z) (GCV smoothness).

    Weights 1/SE^2 with SE the standard error of a median, 1.2533*scatter/sqrt(n); without a
    per-plane scatter, weights are the bead counts n. Falls back to PCHIP for short curves.
    """
    z, y = np.asarray(z, float), np.asarray(y, float)
    n = np.maximum(np.asarray(support, float), 1.)
    if scatter is not None:
        s = np.asarray(scatter, float)
        s = np.where(np.isfinite(s) & (s > 0), s, np.nanmedian(s[s > 0]) if np.any(s > 0) else 1.)
        w = n/(1.2533*s)**2
    else:
        w = n
    if len(z) < 8:
        return PchipInterpolator(z, y, extrapolate=False)
    from scipy.interpolate import make_smoothing_spline
    try:
        return make_smoothing_spline(z, y, w=w/np.mean(w))
    except Exception:
        return PchipInterpolator(z, y, extrapolate=False)


def _register_tilt(beads, consensus, z, cfg):
    """Remove the calibration sample's tilt: the beads of a tilted slide sit at different depths.

    Each bead's angles are compared with a provisional curve; its depth offset delta_b is the
    least-squares fit of angle residual = slope(z)*delta (a depth offset, as opposed to a constant
    angle offset, changes the residual with the local rotation rate). A plane
    delta(x, y) = c1*(x - xc) + c2*(y - yc) is fitted robustly across the image and each bead's
    angles and separations are moved to where the curve puts them without the tilt, i.e. as if the
    bead were at the image centre's depth. The 10x calibration slide was tilted ~1.1 deg along x
    (~12 um across the field); left in, it blurred the curve and dominated the bead-to-bead spread.
    Returns (registered beads, dict(gradient_um_per_px, r2, beads)).
    """
    info = dict(gradient_um_per_px=[0., 0.], r2=0., beads=0, centre_px=[0., 0.])
    prov, support, scatter, seps = consensus(beads)
    valid = np.flatnonzero(np.isfinite(prov) & np.isfinite(seps))
    if len(valid) < 10:
        return beads, info
    unwrapped = np.degrees(np.unwrap(np.radians(prov[valid]*2)))/2
    a_curve = _smooth_curve(z[valid], unwrapped, scatter[valid], support[valid])
    s_curve = _smooth_curve(z[valid], seps[valid], None, support[valid])
    lo, hi = z[valid[0]], z[valid[-1]]
    deltas, xs, ys = [], [], []
    for b in beads:
        zb = z[b[:, 8].astype(int)-1]
        inside = (zb > lo+3) & (zb < hi-3)
        d = axial_difference(b[inside, 6], a_curve(zb[inside]))
        slope = a_curve.derivative()(zb[inside]) if hasattr(a_curve, 'derivative') else np.gradient(a_curve(zb[inside]), zb[inside])
        good = np.abs(d-np.median(d)) < 6 if len(d) else d.astype(bool)
        den = np.sum(slope[good]**2)
        deltas.append(np.sum(d[good]*slope[good])/den if good.sum() >= 10 and den > 0 else np.nan)
        xs.append(b[:, 2].mean()); ys.append(b[:, 5].mean())
    deltas, xs, ys = np.asarray(deltas), np.asarray(xs), np.asarray(ys)
    ok = np.isfinite(deltas)
    if ok.sum() < 20:
        return beads, info
    xc, yc = np.median(xs[ok]), np.median(ys[ok])
    A = np.column_stack((np.ones_like(xs), xs-xc, ys-yc))
    keep = ok.copy()
    for _ in range(4):   # least squares with 3-robust-SD clipping
        coef = np.linalg.lstsq(A[keep], deltas[keep], rcond=None)[0]
        r = deltas-A@coef
        s = 1.4826*np.median(np.abs(r[ok]-np.median(r[ok]))) or 1e-9
        keep = ok & (np.abs(r) <= 3*s)
    fit = A@coef
    r2 = 1-np.var(deltas[keep]-fit[keep])/max(np.var(deltas[keep]), 1e-12)
    info = dict(gradient_um_per_px=[float(coef[1]), float(coef[2])], r2=float(r2), beads=int(keep.sum()),
                centre_px=[float(xc), float(yc)])
    out = []
    for b, x, y in zip(beads, xs, ys):
        tilt = coef[1]*(x-xc)+coef[2]*(y-yc)          # this bead is `tilt` um deeper than the image centre
        zb = np.clip(z[b[:, 8].astype(int)-1], lo, hi)
        b = b.copy()
        b[:, 6] = (b[:, 6]-(a_curve(np.clip(zb+tilt, lo, hi))-a_curve(zb))) % 180
        b[:, C['lobeSeparationPixels']] -= s_curve(np.clip(zb+tilt, lo, hi))-s_curve(zb)
        out.append(b)
    return out, info


def calibrate(rows, nplanes, cfg, offset=(0., 0.)):
    """Robust per-plane axial and separation medians from long, consistently rotating beads."""
    tracks = track(rows, cfg, link_distance=4., gap_distance=5., max_frame_gap=3, method='lap')
    eligible = []
    for tid in np.unique(tracks[:, 9]):
        r = tracks[tracks[:, 9] == tid]
        r = r[np.argsort(r[:, 8])]
        if len(r) >= cfg.calibration_min_planes:
            eligible.append(r)
    if len(eligible) < cfg.calibration_min_beads_per_plane:
        raise RuntimeError(f'Only {len(eligible)} long calibration bead tracks; insufficient support')
    def consensus(beads):
        angles, support, scatter, seps = [], [], [], []
        for f in range(1, nplanes+1):
            obs = np.asarray([row for bead in beads for row in bead if int(row[8]) == f]).reshape(-1, NCOL)
            v = obs[:, 6]
            if len(v) < cfg.calibration_min_beads_per_plane:
                angles.append(np.nan); support.append(len(v)); scatter.append(np.nan); seps.append(np.nan); continue
            reference = np.degrees(np.angle(np.mean(np.exp(2j*np.radians(v)))))/2
            med = reference+np.median(axial_difference(v, reference))
            keep = np.abs(axial_difference(v, med)) <= cfg.calibration_max_residual_deg
            v = v[keep]
            ok = len(v) >= cfg.calibration_min_beads_per_plane
            angles.append((med+np.median(axial_difference(v, med))) % 180 if ok else np.nan)
            seps.append(float(np.median(obs[keep, C['lobeSeparationPixels']])) if ok else np.nan)
            support.append(len(v))
            scatter.append(1.4826*np.median(np.abs(axial_difference(v, angles[-1]))) if len(v) else np.nan)
        return np.asarray(angles), np.asarray(support), np.asarray(scatter), np.asarray(seps)
    z = (np.arange(nplanes)-(nplanes-1)/2)*cfg.z_step_um
    eligible, tilt = _register_tilt(eligible, consensus, z, cfg)
    provisional = consensus(eligible)[0]
    selected, residuals = [], []
    for bead in eligible:
        residual = float(np.nanmedian(np.abs(axial_difference(bead[:, 6], provisional[bead[:, 8].astype(int)-1]))))
        residuals.append(residual)
        if residual <= cfg.calibration_max_residual_deg/2:
            selected.append(bead)
    if len(selected) < cfg.calibration_min_beads_per_plane:
        raise RuntimeError('Too few mutually consistent calibration beads')
    angles, support, scatter, seps = consensus(selected)
    valid = np.isfinite(angles) & np.isfinite(seps)
    runs = np.split(np.flatnonzero(valid), np.flatnonzero(np.diff(np.flatnonzero(valid)) != 1)+1)
    run = max(runs, key=len)
    if len(run) < 10:
        raise RuntimeError('Fewer than 10 contiguous supported calibration planes')
    unwrapped = np.degrees(np.unwrap(np.radians(angles[run]*2)))/2
    # Smooth curves through the per-plane medians (weights 1/SE^2, SE = 1.2533*scatter/sqrt(n),
    # smoothness by generalized cross-validation). Interpolating the medians exactly (PCHIP)
    # would put their noise into z (~0.2 um SD, up to 0.7 um) and make dtheta/dz jagged.
    angle_curve, sep_curve = _smooth_curve(z[run], unwrapped, scatter[run], support[run]), \
        _smooth_curve(z[run], seps[run], None, support[run])
    z_zero = 'middle plane of the calibration stack'
    if str(cfg.z_zero).lower() != 'middle':
        if str(cfg.z_zero).lower() == 'auto':
            # Middle-plane lobe direction decides: horizontal (0 deg) or vertical (90 deg).
            middle = run[np.argmin(np.abs(z[run]))]
            target = 0. if abs(axial_difference(angles[middle], 0.)) <= 45 else 90.
            how = f'auto: middle-plane angle {angles[middle]:.1f} deg is closer to {"horizontal" if target == 0 else "vertical"}'
        else:
            target, how = float(cfg.z_zero), 'user setting'
        z0 = _angle_crossing(z[run], angle_curve(z[run]), target)
        z = z-z0
        shift = z0
        z_zero = (f'plane where the lobe angle is {target:g} deg ({how}); {z0:+.2f} um from the stack middle '
                  f'along the stack, plane {(z0/cfg.z_step_um)+(nplanes+1)/2:.2f}')
    else:
        shift = 0.
    # The curve spans every supported plane (pairing needs the separation at every angle);
    # z is reported only within z_limits.
    z_limits = (float(z[run[0]]), float(z[run[-1]]))
    if cfg.z_range_um:
        z_limits = (max(z_limits[0], -cfg.z_range_um), min(z_limits[1], cfg.z_range_um))
        if z_limits[1]-z_limits[0] < 10*cfg.z_step_um:
            raise RuntimeError(f'Fewer than 10 calibration planes within +-{cfg.z_range_um} um of z = 0')
    dense_z = np.linspace(z[run[0]], z[run[-1]], (len(run)-1)*100+1)
    dense_a = angle_curve(dense_z+shift)        # the curves were fitted before z = 0 was moved
    dense_sep = sep_curve(dense_z+shift)
    in_range = [b[(z[b[:, 8].astype(int)-1] >= z_limits[0]) & (z[b[:, 8].astype(int)-1] <= z_limits[1])] for b in selected]
    lateral_coef, lateral_sd = fit_lateral_model([b for b in in_range if len(b) > 2], z, offset, cfg.sensor_size_px)
    cal = dict(z=z, z_zero=z_zero, z_limits=np.asarray(z_limits), angles=angles, separations=seps,
               support=support, scatter=scatter,
               valid_planes=run, unwrapped=unwrapped, dense_z=dense_z, dense_a=dense_a,
               dense_sep=dense_sep, selected_track_ids=[int(b[0, 9]) for b in selected],
               eligible_tracks=len(eligible), selected_tracks=len(selected), tracks=tracks,
               median_track_residuals_deg=residuals, lateral_coef=lateral_coef,
               sample_tilt_um_per_px=np.asarray(tilt['gradient_um_per_px']), sample_tilt_r2=tilt['r2'],
               sample_tilt_beads=tilt['beads'], sample_tilt_centre_px=np.asarray(tilt['centre_px']),
               lateral_residual_sd_px=lateral_sd, calibration_offset=np.asarray(offset, float),
               sensor_size_px=cfg.sensor_size_px)
    return to_lab_z(cal)


# The stack index of the calibration z-scans runs down into the sample, away from the indenter
# (which comes from above): an indentation moves beads to larger stack z. Reported z is lab z,
# height, positive upwards (toward the indenter), so indentation is negative z. Everything is
# fitted in stack coordinates (ascending plane order) and converted once, by to_lab_z.
Z_CONVENTION = 'up'
Z_CONVENTION_TEXT = ('z is height: positive upwards, toward the indenter (which comes from above), '
                     'so indentation moves beads to negative z; the calibration stack index runs downwards')


def to_lab_z(cal):
    """Stack-coordinate calibration -> lab z (z_lab = -z_stack). Curves stay ascending in z (they are
    reversed), limits are negated and swapped, the odd-power terms of the lateral model and the slide
    tilt change sign. Per-plane arrays keep the plane order (so z per plane is descending)."""
    if cal.get('z_convention') == Z_CONVENTION:
        return cal
    out = dict(cal)
    out['z'] = -np.asarray(cal['z'], float)
    for k in ('dense_z', 'dense_a', 'dense_sep'):                               # reversed: z ascending
        if k in cal:
            out[k] = (-1. if k == 'dense_z' else 1.)*np.asarray(cal[k], float)[::-1].copy()
    if 'z_limits' in cal:
        lo, hi = map(float, cal['z_limits'])
        out['z_limits'] = np.asarray([-hi, -lo])
    if 'lateral_coef' in cal:                   # (older calibrations may lack some of these)
        coef = np.array(cal['lateral_coef'], float)
        odd = [i for i, term in enumerate(LATERAL_TERMS) if 'z^2' not in term]      # z, z*u, z*v
        if coef.shape[-1] != len(LATERAL_TERMS):
            raise ValueError(f'lateral model with {coef.shape[-1]} terms; expected {LATERAL_TERMS}')
        coef[..., odd] *= -1
        out['lateral_coef'] = coef
    if 'sample_tilt_um_per_px' in cal:
        out['sample_tilt_um_per_px'] = -np.asarray(cal['sample_tilt_um_per_px'], float)
    out['z_convention'] = Z_CONVENTION
    return out


def _angle_crossing(z, unwrapped, angle):
    """z where the unwrapped calibration angle equals `angle` (mod 180); the crossing nearest z = 0."""
    crossings = []
    for k in range(int(np.floor((unwrapped.min()-angle)/180)), int(np.ceil((unwrapped.max()-angle)/180))+1):
        target = angle+180*k
        for i in np.flatnonzero((unwrapped[:-1]-target)*(unwrapped[1:]-target) <= 0):
            a0, a1 = unwrapped[i], unwrapped[i+1]
            crossings.append(z[i] if a1 == a0 else z[i]+(target-a0)/(a1-a0)*(z[i+1]-z[i]))
    if not crossings:
        raise RuntimeError(f'The calibrated angle never reaches {angle} deg (mod 180); cannot set z = 0 there')
    return float(min(crossings, key=abs))


LATERAL_TERMS = ['z', 'z*u', 'z*v', 'z^2', 'z^2*u', 'z^2*v']


FIELD_TERMS = ['1', 'u', 'v', 'u^2', 'u*v', 'v^2']


def _field_design(x, y, offset=(0., 0.), sensor=2048):
    centre, half = (sensor+1)/2., sensor/2.
    u = (np.atleast_1d(np.asarray(x, float))+offset[0]-centre)/half
    v = (np.atleast_1d(np.asarray(y, float))+offset[1]-centre)/half
    return np.column_stack((np.ones_like(u), u, v, u*u, u*v, v*v))


def _field_sample(task):
    path, index, cfg = task
    rows, _ = localize_image(read_frame(path, index), cfg, None, index+1, seed=seed_image(path, index, cfg))
    return rows


def estimate_field_separation(path, cfg, model, workers, offset, frames=12, min_snr=10., isolation=40.,
                              max_offset_px=6.):
    """Movie-specific map of lobe separation relative to the calibration: offset(u, v), px.

    The PSF in a sample can differ from the calibration stack across the field (e.g. an
    off-axis aberration makes lobes closer in two corners). Isolated bright beads are
    localized on a few frames WITHOUT the calibration prior (geometric gates only), their
    separation is compared with the calibrated separation at their angle, and a quadratic
    in field position is fitted robustly. Returns (coef (6,), stats dict).
    """
    loose = Config(**{**asdict(cfg), 'min_separation': min(cfg.min_separation, 9.)})
    indices = np.unique(np.linspace(0, frame_count(path)-1, frames).astype(int))
    with ProcessPoolExecutor(max_workers=min(workers, len(indices))) as pool:
        rows = [r for r in pool.map(_field_sample, [(str(path), i, loose) for i in indices]) if len(r)]
    rows = np.vstack(rows) if rows else np.empty((0, NCOL))
    keep = []
    for f in np.unique(rows[:, 8]):
        idx = np.flatnonzero(rows[:, 8] == f)
        tree = cKDTree(rows[idx][:, [2, 5]])
        crowded = np.array([len(tree.query_ball_point(p, isolation)) > 1 for p in rows[idx][:, [2, 5]]])
        keep.extend(idx[~crowded & (rows[idx, C['minLobeSNR']] >= min_snr)])
    rows = rows[keep]
    residual = np.array([model.separation_residual(a, s) for a, s in rows[:, [6, C['lobeSeparationPixels']]]])
    ok = np.isfinite(residual)
    rows, residual = rows[ok], residual[ok]
    if len(rows) < 20:
        return np.zeros(len(FIELD_TERMS)), dict(beads=len(rows), note='too few isolated beads; no field map')
    X_full = _field_design(rows[:, 2], rows[:, 5], offset, cfg.sensor_size_px)
    before = 1.4826*np.median(np.abs(residual-np.median(residual)))
    # Model order grows with the evidence: constant (<60 beads), linear (<200), quadratic.
    # A map implying offsets beyond +-max_offset_px anywhere in the image falls back an order.
    orders = [1, 3, 6][:1+(len(rows) >= 60)+(len(rows) >= 200)]
    for terms in reversed(orders):
        X = X_full[:, :terms]
        use = np.ones(len(rows), bool)
        for _ in range(5):
            c = np.linalg.lstsq(X[use], residual[use], rcond=None)[0]
            err = residual-X@c
            scale = 1.4826*np.median(np.abs(err[use]))+1e-6
            use = np.abs(err) <= 3*scale
        coef = np.r_[c, np.zeros(len(FIELD_TERMS)-terms)]
        span = _span(coef, path, offset, cfg.sensor_size_px)
        if max(abs(span[0]), abs(span[1])) <= max_offset_px:
            break
    else:
        coef = np.zeros(len(FIELD_TERMS))
        span, terms = (0., 0.), 0
    return coef, dict(beads=int(len(rows)), inliers=int(use.sum()), fitted_terms=FIELD_TERMS[:terms],
                      frames=[int(i)+1 for i in indices],
                      residual_mad_before_px=float(before), residual_mad_after_px=float(scale),
                      range_over_image_px=[float(v) for v in span])


def _span(coef, path, offset, sensor):
    h, w = stack_reader(path).shape[-2:]
    yy, xx = np.mgrid[1:h:32, 1:w:32]
    values = _field_design(xx.ravel(), yy.ravel(), offset, sensor)@coef
    return values.min(), values.max()


def image_offset(path, sensor):
    """Default sensor offset (x, y) of an image: assumed centred on the sensor (0 for full frames)."""
    height, width = stack_reader(path).shape[-2:]
    return ((sensor-width)/2., (sensor-height)/2.)


def _lateral_design(z, x, y, offset=(0., 0.), sensor=2048):
    """Design matrix in field coordinates u, v in -1..1 across the full sensor (x, y one-based image px)."""
    centre, half = (sensor+1)/2., sensor/2.
    u = (np.asarray(x)+offset[0]-centre)/half
    v = (np.asarray(y)+offset[1]-centre)/half
    z = np.asarray(z, dtype=float)
    return np.column_stack((z, z*u, z*v, z*z, z*z*u, z*z*v))


def fit_lateral_model(beads, z_grid, offset=(0., 0.), sensor=2048):
    """Apparent midpoint shift with z: dx, dy = f(z, field position), from calibration beads.

    Each bead contributes its own fixed offset (demeaned per bead), so only the change
    with z is modelled. u, v are full-sensor field coordinates (-1..1); `offset` places a
    cropped calibration image on the sensor. Returns coefficients (2, 6) and residual
    SDs (px). The radial terms (z*u for x, z*v for y) capture the defocus-dependent
    magnification; the pure z terms a uniform tilt (optical or stage).

    Robust: Huber-weighted iteratively reweighted least squares (a few beads with a wrong
    pairing or a neighbour must not steer it), and the stage's random plane-to-plane
    wobble (the median residual of all beads in a plane, minus its smooth trend in z) is
    estimated and removed -- it is specific to the calibration scan and does not transfer
    to movies, while a smooth uniform tilt with z does. Residual SDs are robust (MAD).
    """
    X, Y, bead_id, plane = [], [], [], []
    for k, bead in enumerate(beads):
        zz = z_grid[bead[:, 8].astype(int)-1]
        design = _lateral_design(zz, np.full(len(bead), bead[:, 2].mean()), np.full(len(bead), bead[:, 5].mean()),
                                 offset, sensor)
        X.append(design)
        Y.append(bead[:, [2, 5]])
        bead_id.append(np.full(len(bead), k))
        plane.append(bead[:, 8].astype(int)-1)
    X, Y, bead_id, plane = np.vstack(X), np.vstack(Y), np.concatenate(bead_id), np.concatenate(plane)
    nb, npl = int(bead_id.max())+1, len(z_grid)

    def demean(a):
        """Subtract each bead's (weighted) mean -- its fixed offset."""
        a = np.asarray(a, float)
        sums = np.zeros((nb,)+a.shape[1:])
        np.add.at(sums, bead_id, a)
        means = sums/np.bincount(bead_id, minlength=nb).reshape((-1,)+(1,)*(a.ndim-1))
        return a-means[bead_id]

    Xd = demean(X)
    coef = np.zeros((2, X.shape[1]))
    wobble = np.zeros((npl, 2))
    zq = np.column_stack((np.ones(npl), z_grid, z_grid**2))
    for _ in range(6):
        Yc = demean(Y-wobble[plane])
        R = np.zeros_like(Yc)
        for c in range(2):
            w = np.ones(len(Yc))
            for _ in range(3):   # Huber IRLS (k = 1.345 robust SDs)
                sw = np.sqrt(w)
                coef[c] = np.linalg.lstsq(Xd*sw[:, None], Yc[:, c]*sw, rcond=None)[0]
                r = Yc[:, c]-Xd@coef[c]
                s = 1.4826*np.median(np.abs(r-np.median(r))) or 1e-9
                w = np.minimum(1., 1.345*s/np.maximum(np.abs(r), 1e-12))
            R[:, c] = r
        # plane wobble: median residual per plane, minus its smooth (quadratic in z) trend
        med = np.full((npl, 2), np.nan)
        for f in np.unique(plane):
            med[f] = np.median(R[plane == f]+wobble[f], axis=0)
        have = np.all(np.isfinite(med), axis=1)
        if have.sum() > 3:
            trend = zq[have]@np.linalg.lstsq(zq[have], med[have], rcond=None)[0]
            wobble = np.zeros((npl, 2))
            wobble[have] = med[have]-trend
    sd = 1.4826*np.median(np.abs(R-np.median(R, axis=0)), axis=0)
    return coef, sd


def lateral_correction(rows, coef, offset=(0., 0.), sensor=2048):
    """(xCorrected, yCorrected): midpoints with the calibrated z-dependent shift removed.

    Rows without a valid z are left uncorrected. `offset` places this image on the sensor.
    """
    z = np.where(np.isfinite(rows[:, 7]), rows[:, 7], 0.)
    shift = _lateral_design(z, rows[:, 2], rows[:, 5], offset, sensor)@np.asarray(coef).T
    return rows[:, 2]-shift[:, 0], rows[:, 5]-shift[:, 1]


TEMPLATE_HALF = 20


def build_templates(cal, path, isolation=45.):
    """Per-plane median PSF of isolated selected calibration beads, centred on the midpoint."""
    from scipy.ndimage import shift as subpixel_shift
    stack = stack_reader(path)
    tracks = cal['tracks']
    selected = tracks[np.isin(tracks[:, 9], cal['selected_track_ids'])]
    h = TEMPLATE_HALF
    templates = np.full((stack.shape[0], 2*h+1, 2*h+1), np.nan)
    counts = np.zeros(stack.shape[0], int)
    for plane in range(1, stack.shape[0]+1):
        everyone = tracks[tracks[:, 8] == plane]
        tree = cKDTree(everyone[:, [2, 5]])
        image = np.asarray(stack[plane-1], dtype=np.float32)
        crops = []
        for r in selected[selected[:, 8] == plane]:
            if len(tree.query_ball_point(r[[2, 5]], isolation)) > 1:
                continue
            cx, cy = r[2]-1, r[5]-1
            ix, iy = int(round(cx)), int(round(cy))
            if ix-h-2 < 0 or iy-h-2 < 0 or ix+h+3 > image.shape[1] or iy+h+3 > image.shape[0]:
                continue
            crop = image[iy-h-2:iy+h+3, ix-h-2:ix+h+3].astype(float)
            crop = subpixel_shift(crop, (iy-cy, ix-cx), order=1)[2:-2, 2:-2]
            crop -= np.median(np.r_[crop[0], crop[-1], crop[:, 0], crop[:, -1]])
            crops.append(crop/crop.sum())
        if len(crops) >= 3:
            templates[plane-1] = np.median(crops, axis=0)
            counts[plane-1] = len(crops)
    return templates, counts


def template_scores(image, rows, templates, z_grid):
    """Zero-normalized cross-correlation of each localization's patch with the template at its z.

    Rows without a valid z are scored against every plane (best score kept). Pixels
    outside the image are ignored. Neighbouring beads inside the patch lower the score.
    """
    from scipy.ndimage import shift as subpixel_shift
    h = TEMPLATE_HALF
    valid_planes = np.flatnonzero(np.isfinite(templates[:, 0, 0]))
    t = templates[valid_planes]
    scores = np.full(len(rows), np.nan)
    padded = np.pad(np.asarray(image, dtype=np.float32), h+3, constant_values=np.nan)
    for i, r in enumerate(rows):
        cx, cy = r[2]-1+h+3, r[5]-1+h+3
        ix, iy = int(round(cx)), int(round(cy))
        crop = padded[iy-h-2:iy+h+3, ix-h-2:ix+h+3].astype(float)
        mask = np.isfinite(crop)
        crop = subpixel_shift(np.where(mask, crop, np.nanmedian(crop)), (iy-cy, ix-cx), order=1)[2:-2, 2:-2]
        mask = mask[2:-2, 2:-2]
        if np.isfinite(r[7]):
            plane = np.argmin(np.abs(z_grid[valid_planes]-r[7]))
            choices = t[max(plane-1, 0):plane+2]
        else:
            choices = t
        c = crop[mask]-crop[mask].mean()
        best = -1.
        for tpl in choices:
            v = tpl[mask]-tpl[mask].mean()
            denominator = np.sqrt((c*c).sum()*(v*v).sum())
            if denominator > 0:
                best = max(best, float((c*v).sum()/denominator))
        scores[i] = best
    return scores


def score_frame(task):
    path, index, rows, templates, z_grid = task
    image = read_frame(path, index)
    return index+1, template_scores(image, rows, templates, z_grid)


def angle_to_z(angles, cal, seps=None):
    """Compatibility wrapper around CalibrationModel.angle_to_z for a calibration dict."""
    return CalibrationModel(cal['dense_z'], cal['dense_a'], cal.get('dense_sep', np.zeros_like(cal['dense_z'])),
                            cal.get('z_limits')).angle_to_z(angles, seps)


# ----------------------------------------------------------------------------- outputs

def save_payload(path, rows, rejected, cal, cfg, diagnostics, history, drift, reasons,
                 calibration_path=CALIBRATION, movie_offset=(0., 0.), movie_path=None, dz_raw=None):
    metadata = dict(configuration=asdict(cfg), columns=COLUMNS, quality_columns=QUALITY, version=VERSION,
        calibration_source=str(calibration_path), movie_source=str(movie_path) if movie_path else None,
        pixel_size_um=cfg.pixel_size_um, sensor_offset_px=list(map(float, movie_offset)),
        calibration_selected_track_ids=cal['selected_track_ids'],
        calibration_supported_z_um=[float(v) for v in cal['z_limits']],
        endpoint_risk_margin_um=4.,
        xy_units='one-based image pixels; +x right, +y down',
        angle_convention='atan2(y2-y1,x2-x1), degrees modulo 180',
        z_origin=str(cal.get('z_zero', 'middle plane of the calibration stack')),
        z_convention=Z_CONVENTION, z_direction=Z_CONVENTION_TEXT,
        tracking='XY+rotation two-stage LAP: adjacent frames then global segment gap closing; no split/merge',
        recovery='track-guided refit of missed frames (recovered=1); each frame fitted independently, no averaging',
        transient_rejection=f'tracks shorter than {cfg.min_track_length} frames moved to *_rejected.csv',
        z_status={'0': 'valid unique inverse', '1': 'outside supported calibration',
                  '2': 'ambiguous inverse', '3': 'ambiguous angle resolved by lobe separation'},
        minLobeSNR='minimum fitted lobe amplitude / camera pixel noise SD of that frame',
        sharedLobe='1 when one lobe coincides with a lobe of a neighbouring bead (three-spot overlap)',
        templateScore='zero-normalized cross-correlation of the 41x41 patch with the calibration PSF at that z (1 = identical)',
        stabilization='x/y/zStabilized = xMean/yMean/zMicrons minus the median whole-field shake of that frame '
                      '(see *_drift.csv; the z drift is smoothed over time, sigma = drift_z_smooth_frames); raw columns are unchanged',
        recovery_history=history, diagnostics=diagnostics)
    # True depth: all processing uses calibration (stage) units; exported z, its precision, z drift
    # and the calibration's z axis are multiplied by axial_scale (e.g. 1.33 for an air objective
    # imaging a watery sample: refractive-index focal shift, paraxial n_sample/n_immersion).
    f = float(cfg.axial_scale)
    metadata['z_units'] = ('µm of stage displacement (calibration units)' if f == 1 else
                           f'true depth in the sample: stage µm x axial_scale {f:g} (refractive-index focal-shift correction)')
    metadata['axial_scale'] = f
    def corrected(r):
        return np.column_stack(lateral_correction(r, cal['lateral_coef'], movie_offset, cfg.sensor_size_px)).reshape(-1, 2)
    def stabilized(r):
        d = drift[r[:, 8].astype(int)-1] if len(r) else np.empty((0, 3))
        c = corrected(r)
        return np.column_stack((c[:, 0]-d[:, 0], c[:, 1]-d[:, 1], f*(r[:, 7]-d[:, 2]))).reshape(-1, 3)
    def scaled(r):
        r = r.copy()
        r[:, 7] *= f
        r[:, C['zPrecisionUm']] *= f
        return r
    drift_out = drift.copy()
    drift_out[:, 2] *= f
    dz_raw = None if dz_raw is None else f*np.asarray(dz_raw)
    metadata['reject_reasons'] = REJECT_REASONS
    metadata['lateral_correction'] = dict(
        description='x/yCorrected = xMean/yMean minus the calibrated apparent midpoint shift with z '
                    '(defocus magnification + uniform tilt); rows without valid z are uncorrected. '
                    'Stabilized columns start from the corrected coordinates.',
        terms=LATERAL_TERMS,
        field_coordinates=f'u=(x+offset_x-{(cfg.sensor_size_px+1)/2})/{cfg.sensor_size_px/2}, likewise v (one-based image px; offset places the image on the sensor)',
        coefficients_dx=np.asarray(cal['lateral_coef'])[0].tolist(), coefficients_dy=np.asarray(cal['lateral_coef'])[1].tolist(),
        residual_sd_px=np.asarray(cal['lateral_residual_sd_px']).tolist())
    savemat(path, dict(localizationMatrix=scaled(rows)[:, :10], columnNames=np.asarray(COLUMNS, dtype=object),
                       fitQualityMatrix=scaled(rows)[:, 10:], qualityNames=np.asarray(QUALITY, dtype=object),
                       zStatus=rows[:, C['zStatus']][:, None], correctedMatrix=corrected(rows),
                       stabilizedMatrix=stabilized(rows),
                       driftMatrix=np.column_stack((np.arange(1, len(drift)+1), drift_out)),
                       calibrationMatrix=np.column_stack((f*cal['z'], cal['angles'], cal['support'], cal['scatter'], cal['separations'])),
                       interpolatedCalibration=np.column_stack((f*cal['dense_z'], cal['dense_a'], cal['dense_sep'])),
                       metadataJSON=json.dumps(metadata)), do_compression=True)
    header = COLUMNS+QUALITY+['trackLength', 'xCorrected', 'yCorrected', 'xStabilized', 'yStabilized', 'zStabilized']
    for r, suffix, tail in ((rows, '_localizations.csv', []), (rejected, '_rejected.csv', ['rejectReason'])):
        lengths = dict(zip(*np.unique(r[:, 9], return_counts=True))) if len(r) else {}
        extra = np.array([lengths.get(t, 0) for t in r[:, 9]]).reshape(-1, 1)
        table = np.hstack((scaled(r), extra, corrected(r), stabilized(r))).reshape(-1, NCOL+6)
        if tail:
            table = np.hstack((table, np.asarray(reasons).reshape(-1, 1)))
        np.savetxt(path.with_name(path.name.replace('_payload.mat', suffix)), table, delimiter=',',
                   header=','.join(header+tail), comments='', fmt='%.6g')
    cols, head = [np.arange(1, len(drift)+1), drift_out], 'frame_number,dx,dy,dz'
    if dz_raw is not None:
        cols.append(dz_raw); head += ',dzRaw'   # dz is the time-smoothed z drift; dzRaw the per-frame one
    np.savetxt(path.with_name(path.name.replace('_payload.mat', '_drift.csv')),
               np.column_stack(cols), delimiter=',', header=head, comments='', fmt='%.6g')


def plots(out, cal, movies):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    for tid in cal['selected_track_ids']:
        r = cal['tracks'][cal['tracks'][:, 9] == tid]
        a = np.degrees(np.unwrap(np.radians(r[:, 6]*2)))/2
        midpoint = np.argmin(abs(r[:, 8]-(len(cal['z'])+1)/2))
        ref = np.interp(cal['z'][int(r[midpoint, 8])-1], cal['dense_z'], cal['dense_a'])
        a += 180*round((ref-a[midpoint])/180)
        zz = cal['z'][r[:, 8].astype(int)-1]
        ax[0].plot(zz, a, alpha=.15, lw=.6)
        ax[1].plot(zz, r[:, C['lobeSeparationPixels']], alpha=.15, lw=.6)
    ax[0].plot(cal['dense_z'], cal['dense_a'], 'k', lw=2)
    ax[0].set(xlabel='z (µm)', ylabel='Unwrapped axial angle (degrees)',
              title=f"Angle calibration\nz = 0: {str(cal.get('z_zero', 'stack middle'))[:70]}")
    for a in ax:
        a.axvspan(*cal['z_limits'], color='g', alpha=.06)
    ax[1].plot(cal['dense_z'], cal['dense_sep'], 'k', lw=2)
    ax[1].set(xlabel='Z (µm)', ylabel='Lobe separation (px)', title='Separation calibration')
    ax[2].plot(cal['z'], cal['support'])
    ax[2].set(xlabel='Z (µm)', ylabel='Beads supporting plane', title='Calibration support after rejection')
    fig.tight_layout(); fig.savefig(out/'calibration.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(len(movies), 2, figsize=(12, 5*len(movies)), squeeze=False)
    for axrow, (name, path, rows) in zip(axes, movies):
        middle = frame_count(path)//2
        image = read_frame(path, middle)
        frame = rows[rows[:, 8] == middle+1]
        rec = frame[:, C['recovered']] == 1
        for ax in axrow:
            ax.imshow(image, cmap='gray', vmin=np.percentile(image, 5), vmax=np.percentile(image, 99.8))
            ax.scatter(frame[~rec, 2]-1, frame[~rec, 5]-1, s=12, facecolors='none', edgecolors='cyan', linewidth=.6)
            ax.scatter(frame[rec, 2]-1, frame[rec, 5]-1, s=16, facecolors='none', edgecolors='magenta', linewidth=.8)
            for r in frame:
                ax.plot(r[[0, 1]]-1, r[[3, 4]]-1, color='orange', lw=.5)
            ax.set_title(f'{name}: frame {middle+1}, {len(frame)} beads ({rec.sum()} recovered, magenta)')
        cy, cx = image.shape[0]/2, image.shape[1]/2
        axrow[1].set_xlim(cx-256, cx+256); axrow[1].set_ylim(cy+256, cy-256)
    fig.tight_layout(); fig.savefig(out/'experimental_fits.png', dpi=160); plt.close(fig)


def field_plots(out, field_maps):
    """Field-dependent separation offset map with the measured per-bead separation residuals."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(field_maps), figsize=(6.5*len(field_maps), 5.5), squeeze=False)
    for ax, (name, path, model, rows) in zip(axes[0], field_maps):
        h, w = stack_reader(path).shape[-2:]
        yy, xx = np.mgrid[1:h+1:16, 1:w+1:16]
        grid = model.field_separation(xx.ravel(), yy.ravel()).reshape(xx.shape)
        im = ax.imshow(grid, extent=(1, w, h, 1), cmap='RdBu_r', vmin=-4, vmax=4)
        first = rows[rows[:, 8] == rows[:, 8].min()]
        raw = first[:, C['separationResidual']]+first[:, C['fieldSeparationOffset']]
        ax.scatter(first[:, 2], first[:, 5], c=raw, cmap='RdBu_r', vmin=-4, vmax=4, s=14, edgecolors='k', linewidths=.3)
        ax.set(title=f'{name}: lobe separation vs calibration (px)\nmap = fitted offset, dots = beads in first frame',
               xlabel='x (px)', ylabel='y (px)')
        plt.colorbar(im, ax=ax, shrink=.8)
    fig.tight_layout(); fig.savefig(out/'field_separation.png', dpi=120); plt.close(fig)


class Progress:
    """Optional JSON status file for front ends (analyze.py, the explorer's New analysis)."""

    def __init__(self):
        self.path, self.started, self.output = None, None, None

    def open(self, path):
        import datetime
        self.path = Path(path) if path else None
        self.started = datetime.datetime.now().isoformat(timespec='seconds')

    def update(self, stage, progress, message='', state='running', error=None, output=None):
        if self.path is None:
            return
        import datetime
        self.output = output or self.output
        info = dict(state=state, stage=stage, progress=round(float(min(max(progress, 0.), 1.)), 4), message=message,
                    started=self.started, updated=datetime.datetime.now().isoformat(timespec='seconds'),
                    output=str(self.output) if self.output else None, error=error)
        tmp = self.path.with_suffix('.tmp')
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(info, indent=1))
        except OSError:
            return
        # Windows refuses the replace while a reader (e.g. the explorer polling) has the file open.
        for attempt in range(25):
            try:
                os.replace(tmp, self.path)
                return
            except OSError:
                time.sleep(.01+.04*np.random.rand())


PROGRESS = Progress()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--status-file', type=Path, help='write JSON progress here (for front ends)')
    parser.add_argument('--calibration-cache', type=Path,
                        help='folder for the calibration fit cache, shared between runs using the same calibration '
                             '(default: the output folder)')
    parser.add_argument('--camera', type=Path,
                        help='camera.npz from camera_calibration.py: fit the movies by sCMOS maximum likelihood')
    parser.add_argument('--calibration-planes', metavar='A:B',
                        help='use only calibration planes A..B (1-based, inclusive); others are not fitted')
    parser.add_argument('--movie-config', help='JSON (file or inline) overriding Config for the MOVIES only, '
                                               'e.g. dim-data settings; the calibration keeps --config')
    parser.add_argument('--output', type=Path, default=None,
                        help='result folder (default: a new folder in the results folder chosen at setup, see '
                             'user_settings.py; analyze.py names it after the movie)')
    parser.add_argument('--workers', type=int, default=max(1, min(12, (os.cpu_count() or 2)-2)))
    parser.add_argument('--config', type=Path, help='JSON object overriding Config defaults')
    parser.add_argument('--matlab', default=None, help='matlab executable (default: found on PATH or in the '
                                                         'usual install folders, newest release)')
    parser.add_argument('--calibration-only', action='store_true')
    parser.add_argument('--no-matlab', action='store_true', help='Write interchange files; run export_tables.m separately to create native tables')
    parser.add_argument('--calibration', type=Path, default=CALIBRATION,
                        help='bead z-stack TIFF, one plane per z step (default: the 20x corrected stack)')
    parser.add_argument('--movie', action='append', default=[], metavar='[NAME=]PATH',
                        help='movie TIFF to analyse (repeatable; default: the two 20x indentation movies)')
    parser.add_argument('--pixel-size', type=float, help='µm per camera pixel (Config.pixel_size_um)')
    parser.add_argument('--z-step', type=float, help='µm between calibration planes (Config.z_step_um)')
    parser.add_argument('--sensor-size', type=int, help='full camera width in px (Config.sensor_size_px)')
    parser.add_argument('--z-zero', help="where z = 0 is: 'auto' (default: horizontal or vertical lobes, whichever the "
                                         "stack's middle plane is closer to), 'middle' (middle plane), or a lobe angle in degrees")
    parser.add_argument('--z-range', type=float, help='report z only within +- this many µm of z = 0 (calibration units)')
    parser.add_argument('--no-motion', action='store_true', help='skip the motion analysis of the tracks (track_analysis.py)')
    parser.add_argument('--frame-interval', type=float,
                        help='ms between movie frames: motion results in µm²/s and µm/s instead of per frame')
    parser.add_argument('--axial-scale', type=float,
                        help='multiply exported z by this (true depth). Air objective + watery sample: 1.33 '
                             '(paraxial n_sample/n_immersion; slightly larger at high NA). Default 1 (stage units)')
    parser.add_argument('--no-field-correction', action='store_true',
                        help='do not learn a per-movie field map of lobe separation (Config.field_correction)')
    parser.add_argument('--offset', action='append', default=[], metavar='NAME=X,Y',
                        help="sensor position of a cropped image's top-left pixel; NAME is 'calibration' or a "
                             "movie name (default: crops assumed centred on the sensor)")
    args = parser.parse_args(argv)
    PROGRESS.open(args.status_file)
    try:
        return _run(args, parser)
    except Exception as exc:
        PROGRESS.update('error', 1., str(exc), state='error', error=f'{type(exc).__name__}: {exc}')
        raise


def _run(args, parser):
    settings = json.loads(args.config.read_text()) if args.config else {}
    for key, value in (('pixel_size_um', args.pixel_size), ('z_step_um', args.z_step), ('sensor_size_px', args.sensor_size),
                       ('z_zero', args.z_zero), ('z_range_um', args.z_range), ('axial_scale', args.axial_scale)):
        if value is not None:
            settings[key] = value
    if 'z_step_um' not in settings:   # analyze.py always passes it; direct runs must not rely on the default silently
        print(f'WARNING: no --z-step given; using the default {Config().z_step_um} µm between calibration planes. '
              'A wrong z step scales every z value.', flush=True)
    if args.no_field_correction:
        settings['field_correction'] = False
    cfg = Config(**settings)
    movie_settings = {}
    if args.movie_config:
        text = args.movie_config
        movie_settings = json.loads(Path(text).read_text() if Path(text).is_file() else text)
    if args.camera:
        if not args.camera.is_file():
            parser.error(f'camera file not found: {args.camera}')
        movie_settings = {**movie_settings, 'noise_model': 'poisson', 'camera_file': str(args.camera.resolve())}
    # {"per_movie": {name: {...}}} inside the movie config: settings for single movies, e.g. a dim
    # 1 ms movie analysed together with a bright 5 ms one
    per_movie = movie_settings.pop('per_movie', {}) or {}
    cfg_movie = Config(**{**settings, **movie_settings})
    if str(cfg.z_zero).lower() not in ('auto', 'middle'):
        try:
            float(cfg.z_zero)
        except ValueError:
            parser.error("--z-zero must be 'auto', 'middle' or an angle in degrees")
    if args.workers < 1 or cfg.z_step_um <= 0 or cfg.min_separation >= cfg.max_separation:
        parser.error('Workers and Z step must be positive; separation bounds must be ordered')
    movie_paths = {}
    for spec in args.movie:
        name, _, path = spec.rpartition('=') if '=' in spec else ('', '', spec)
        path = Path(path)
        name = name or path.name.split('.')[0].replace('_MMStack_Pos0', '')
        movie_paths[name] = path
    movie_paths = movie_paths or dict(DEFAULT_MOVIES)
    for path in [args.calibration, *movie_paths.values()]:
        if not path.exists():
            parser.error(f'File not found: {path}')
    offsets = {}
    for spec in args.offset:
        name, _, xy = spec.partition('=')
        offsets[name.strip()] = tuple(float(v) for v in xy.split(','))
    def offset_of(name, path):
        return offsets.get(name, image_offset(path, cfg.sensor_size_px))
    if args.output is None:
        import user_settings
        args.output = user_settings.results_dir()/time.strftime('pipeline_%Y%m%d-%H%M%S')
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    calibration_offset = offset_of('calibration', args.calibration)
    if cfg.axial_scale <= 0:
        parser.error('--axial-scale must be positive')
    run_info = dict(version=VERSION, pixel_size_um=cfg.pixel_size_um, z_step_um=cfg.z_step_um, axial_scale=cfg.axial_scale,
                    z_convention=Z_CONVENTION, z_direction=Z_CONVENTION_TEXT, sensor_size_px=cfg.sensor_size_px,
                    calibration=dict(path=str(args.calibration.resolve()), offset=calibration_offset,
                                     shape=list(stack_reader(args.calibration).shape)),
                    movies={n: dict(path=str(p.resolve()), offset=offset_of(n, p), shape=list(stack_reader(p).shape))
                            for n, p in movie_paths.items()})
    run_info['movie_settings'] = movie_settings
    for name in per_movie:
        if name not in movie_paths:
            parser.error(f'--movie-config per_movie names an unknown movie: {name}')
        run_info['movies'][name]['settings'] = per_movie[name]
    (out/'run_info.json').write_text(json.dumps(run_info, indent=2))
    cal_dir = (args.calibration_cache or out).resolve()
    PROGRESS.update('calibration', 0., 'fitting calibration planes', output=out)
    planes = None
    if args.calibration_planes:
        a, _, b = args.calibration_planes.partition(':')
        planes = list(range(int(a)-1, int(b)))
        run_info['calibration']['planes'] = [int(a), int(b)]
        (out/'run_info.json').write_text(json.dumps(run_info, indent=2))
    # Progress bar: each step's share is its expected share of the time, so the percentage (and the time
    # left estimated from it) tracks the clock: the calibration planes still to fit (cached ones are free),
    # then per movie frames x image size; within a movie field map / fitting / recovery / writing take
    # about 20 / 45 / 30 / 5 % (timed 10x and 20x runs), and the motion analysis about 8 % of the movies.
    def mpx(p):
        h, w = stack_reader(p).shape[-2:]
        return h*w/1.05e6
    n_cal = len(planes) if planes else frame_count(args.calibration)
    cal_w = (n_cal-cached_frame_count(args.calibration, cal_dir/'cache_calibration', cfg, None, planes))*mpx(args.calibration)+2
    mov_w = {} if args.calibration_only else {n: 1.5*frame_count(p)*mpx(p) for n, p in movie_paths.items()}
    motion_w = 0 if (args.calibration_only or args.no_motion) else .08*sum(mov_w.values())
    share = lambda w: .985*w/(cal_w+sum(mov_w.values())+motion_w)
    rows, _ = localize_stack(args.calibration, cal_dir/'cache_calibration', cfg, args.workers, None, 'calibration',
                             progress=(0., .95*share(cal_w)), planes=planes)
    PROGRESS.update('calibration', .95*share(cal_w), 'building the calibration curve', output=out)
    cal = calibrate(rows, frame_count(args.calibration), cfg, calibration_offset)
    cal['templates'], cal['template_counts'] = build_templates(cal, args.calibration)
    np.savez_compressed(out/'calibration.npz', **cal)
    model = CalibrationModel(cal['dense_z'], cal['dense_a'], cal['dense_sep'], cal['z_limits'])
    print(f"Calibration: {cal['selected_tracks']} beads, z reported for {cal['z_limits'][0]:.1f}..{cal['z_limits'][1]:.1f} µm; z = 0: {cal['z_zero']}", flush=True)
    movies, summary = [], {'version': VERSION, 'calibration_selected_beads': cal['selected_tracks'],
        'lateral_model': dict(terms=LATERAL_TERMS, dx=np.round(cal['lateral_coef'][0], 5).tolist(),
                              dy=np.round(cal['lateral_coef'][1], 5).tolist(),
                              residual_sd_px=np.round(cal['lateral_residual_sd_px'], 3).tolist()),
        'calibration_supported_z_um': [float(v) for v in cal['z_limits']],
        'z_zero': cal['z_zero'], 'z_convention': Z_CONVENTION, 'z_direction': Z_CONVENTION_TEXT, 'axial_scale': cfg.axial_scale,
        'z_units': 'stage µm (calibration units)' if cfg.axial_scale == 1 else
                   f'true depth: stage µm x {cfg.axial_scale:g} in all exported z columns; summary values are in stage µm',
        'movies': {}}
    # the calibration slide's tilt, removed from the calibration (it concerns only that slide)
    gx, gy = cal['sample_tilt_um_per_px']
    h, w = stack_reader(args.calibration).shape[-2:]
    per_mm = 1000/cfg.pixel_size_um
    summary['calibration_sample_tilt'] = dict(
        um_per_mm_x=round(float(gx*per_mm), 2), um_per_mm_y=round(float(gy*per_mm), 2),
        degrees=round(float(np.degrees(np.arctan(np.hypot(gx, gy)/cfg.pixel_size_um))), 2),
        depth_range_across_image_um=round(float(abs(gx)*w+abs(gy)*h), 2), r2=round(float(cal['sample_tilt_r2']), 2),
        beads=int(cal['sample_tilt_beads']),
        note='calibration beads lie on a tilted plane; their depth differences were removed before building '
             'the calibration curve (in stage units). Movies are not affected.')
    field_maps = []
    cal_cfg = cfg
    if not args.calibration_only:
        base = share(cal_w)
        for i, (name, path) in enumerate(movie_paths.items()):
            if i:
                base += span
            span = share(mov_w[name])
            movie_offset = run_info['movies'][name]['offset']
            # movies use the movie settings; the camera maps are placed with this movie's sensor offset
            cfg = replace(cfg_movie, **per_movie.get(name, {}),
                          camera_offset_x=float(movie_offset[0]), camera_offset_y=float(movie_offset[1]))
            PROGRESS.update(name, base, 'measuring the lobe-separation field map', output=out)
            if cfg.field_correction:
                field_coef, field_stats = estimate_field_separation(path, cfg, model, args.workers, movie_offset)
            else:
                field_coef, field_stats = np.zeros(len(FIELD_TERMS)), dict(beads=0, note='field correction disabled')
            movie_model = model.with_field(field_coef, movie_offset, cfg.sensor_size_px)
            print(f"{name}: field separation map from {field_stats.get('beads')} isolated beads, "
                  f"offset range {field_stats.get('range_over_image_px')} px", flush=True)
            rows, diag = localize_stack(path, out/f'cache_{name}', cfg, args.workers, movie_model, name,
                                        progress=(base+.2*span, .45*span))
            detected = len(rows)
            PROGRESS.update(name, base+.65*span, 'tracking and track-guided recovery', output=out)
            if cfg.recover:
                rows, history, jumps = recover(rows, path, cfg, movie_model, args.workers, name,
                                               progress=(out, base+.65*span, .3*span))
            else:
                rows, history = track(rows, cfg, model=movie_model), []
                bad = jump_outliers(rows, cfg, movie_model)
                rows, jumps = rows[~bad], rows[bad]
            PROGRESS.update(name, base+.95*span, 'scoring and writing results', output=out)
            if len(jumps):
                jf = movie_model.field_separation(jumps[:, 2], jumps[:, 5])
                jumps[:, C['fieldSeparationOffset']] = jf
                jumps[:, 7], jumps[:, C['zStatus']] = movie_model.angle_to_z(jumps[:, 6], jumps[:, C['lobeSeparationPixels']]-jf)
            field = movie_model.field_separation(rows[:, 2], rows[:, 5])
            rows[:, C['fieldSeparationOffset']] = field
            rows[:, 7], rows[:, C['zStatus']] = movie_model.angle_to_z(rows[:, 6], rows[:, C['lobeSeparationPixels']]-field)
            rows[:, C['separationResidual']] = [movie_model.separation_residual(r[6], r[C['lobeSeparationPixels']], r[2], r[5])
                                                for r in rows]
            for r in (rows, jumps):   # z error bar = angle error / local rotation rate
                if len(r):
                    r[:, C['zPrecisionUm']] = r[:, C['anglePrecisionDeg']]/np.maximum(np.abs(movie_model.slope(r[:, 7])), 1e-6)
            frames = rows[:, 8].astype(int)
            tasks = [(str(path), f-1, rows[frames == f], cal['templates'], cal['z']) for f in np.unique(frames)]
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for f, s in pool.map(score_frame, tasks):
                    rows[frames == f, C['templateScore']] = s
            rows, rejected, reasons = split_transient(rows, cfg)
            rejected = np.vstack((rejected, jumps))
            reasons = np.r_[reasons, np.full(len(jumps), 4)]
            lateral = rows.copy()
            lateral[:, 2], lateral[:, 5] = lateral_correction(rows, cal['lateral_coef'], movie_offset, cfg.sensor_size_px)
            drift = estimate_drift(lateral, frame_count(path), cfg.drift_z_smooth_frames)
            dz_raw = estimate_drift(lateral, frame_count(path))[:, 2]
            save_payload(out/f'{name}_payload.mat', rows, rejected, cal, cfg, diag, history, drift, reasons,
                         args.calibration, movie_offset, path, dz_raw=dz_raw)
            movies.append((name, path, rows))
            lengths = np.unique(rows[:, 9], return_counts=True)[1]
            status = rows[:, C['zStatus']]
            per_frame = np.bincount(rows[:, 8].astype(int))[1:]
            summary['movies'][name] = dict(localizations=len(rows), per_frame_min_median_max=[int(per_frame.min()), float(np.median(per_frame)), int(per_frame.max())],
                detected_first_pass=detected, recovered=int(rows[:, C['recovered']].sum()),
                rejected_transient_localizations=int(np.sum(reasons == 1)),
                rejected_low_quality_localizations=int(np.sum(reasons == 2)),
                rejected_low_quality_tracks=len(np.unique(rejected[reasons == 2, 9])),
                rejected_satellite_tracks=len(np.unique(rejected[reasons == 3, 9])),
                jump_outliers_removed=int(np.sum(reasons == 4)),
                valid_z=int(np.sum(status == 0)), resolved_by_separation_z=int(np.sum(status == 3)),
                out_of_range_z=int(np.sum(status == 1)), ambiguous_z=int(np.sum(status == 2)),
                tracks=len(lengths), full_length_tracks=int(np.sum(lengths == len(per_frame))),
                median_track_length=float(np.median(lengths)), tracks_at_least_10_frames=int(np.sum(lengths >= 10)),
                endpoint_risk_localizations=int(np.sum((rows[:, 7] < cal['z_limits'][0]+4) | (rows[:, 7] > cal['z_limits'][1]-4))),
                jointly_fitted_localizations=int(np.sum(rows[:, C['jointEmitterCount']] > 1)),
                shared_lobe_localizations=int(np.nansum(rows[:, C['sharedLobe']])),
                template_score_median=float(np.nanmedian(rows[:, C['templateScore']])),
                max_abs_drift_px=float(np.abs(drift[:, :2]).max()), max_abs_drift_z_um=float(np.abs(drift[:, 2]).max()),
                field_separation=dict(terms=FIELD_TERMS, coef=np.round(field_coef, 4).tolist(), **field_stats),
                recovery_history=history)
            field_maps.append((name, path, movie_model, rows))
    plots(out, cal, movies) if movies else None
    field_plots(out, field_maps) if field_maps else None
    (out/'summary.json').write_text(json.dumps(summary, indent=2))
    if not args.calibration_only and not args.no_motion:
        # motion analysis of the tracks (track_analysis.py); optional, never fails the run
        PROGRESS.update('motion', share(cal_w+sum(mov_w.values())), 'classifying track motion and finding when beads move',
                        output=out)
        try:
            import track_analysis
            track_analysis.analyse(out, frame_interval_ms=args.frame_interval)
        except Exception as exc:
            summary['warnings'] = summary.get('warnings', [])+[f'motion analysis failed ({exc}); '
                                                               'run track_analysis.py or use the explorer to retry']
            (out/'summary.json').write_text(json.dumps(summary, indent=2))
            print(f'WARNING: motion analysis failed: {exc}', flush=True)
    if not args.calibration_only and not args.no_matlab:
        args.matlab = args.matlab or find_matlab()
        if args.matlab and Path(args.matlab).exists():
            PROGRESS.update('export', .985, 'exporting MATLAB tables', output=out)
            matlab_cmd = "addpath('"+str(ROOT).replace("'", "''")+"'); export_tables('"+str(out).replace("'", "''")+"');"
            # optional extra: all results are already written, so a MATLAB problem is only a warning
            try:
                subprocess.run([args.matlab, '-batch', matlab_cmd], check=True, timeout=900)
            except Exception as exc:
                summary['warnings'] = summary.get('warnings', []) + [f'MATLAB table export failed ({exc}); the CSV '
                                                                     f'and *_payload.mat results are complete']
                (out/'summary.json').write_text(json.dumps(summary, indent=2))
                print(f'WARNING: MATLAB table export failed: {exc}', flush=True)
        else:
            print(f'MATLAB not found{f" at {args.matlab}" if args.matlab else ""}; skipped the native table export', flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    PROGRESS.update('done', 1., 'finished', state='done', output=out)
    return summary


if __name__ == '__main__':
    main()
