"""One-step analysis: a calibration z-stack plus movie(s) -> tracked 3D bead localizations.

    python DHPSF_pipeline/analyze.py CALIBRATION.tif MOVIE.tif [MOVIE2.tif ...]
    python DHPSF_pipeline/analyze.py --calibration CAL.tif --movie MOVIE.tif [--pixel-size 0.63] [--open]
    python DHPSF_pipeline/analyze.py --inspect FILE.tif          # what the tool infers about a file

Everything else is decided automatically and recorded in the run's run_info.json:
  * pixel size from the objective in the file/folder name ('10x' -> 0.63 µm, '20x' -> 0.325 µm)
    unless --pixel-size is given; z step from the stage speed in the calibration's name ('25ums') times
    its frame interval, unless --z-step (without either, the run stops rather than guessing);
  * z reported within +-75 µm for 10x (the useful part of its long rotation range), full range otherwise;
  * crop position on the sensor from Micro-Manager *_metadata.txt ("ROI") when present, else centred;
  * dim movies (median lobe SNR < 6, e.g. 1 ms exposures) get the dim-data detection settings;
  * a calibration too dim to calibrate from is rejected early with advice;
  * the calibration fit is cached per calibration file (DHPSF_pipeline/calibrations/), so further
    movies with the same calibration skip it;
  * output goes to DHPSF_pipeline/runs/<movie>_<date-time>/, progress to <output>/status.json.
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import pipeline as P
import user_settings

ROOT = Path(__file__).resolve().parent
OBJECTIVE_PIXEL_UM = {10: .63, 20: .325}      # measured with a USAF target (READMEfile.txt)
DEFAULT_Z_RANGE = {10: 75.}                   # 10x rotates slowly over a very long range
DIM_SNR = 6.                                  # median lobe SNR below this -> dim-data settings
# dim data: accept pairs on their combined (matched-filter) SNR and a precision-aware separation prior;
# on the 1 ms movie this raised single-frame recall from 95.1 to 97.8 % with fewer false positives
DIM_SETTINGS = {'min_snr': 2., 'recover_min_snr': 1.5, 'min_pair_snr': 10., 'sep_sd_mode': 'precision'}
MIN_CALIBRATION_BEADS = 20
WATER_AIR_SCALE = 1.33                        # true depth / stage displacement, air objective into water (paraxial)


def keep_awake():
    """Stop Windows from going to sleep while this analysis runs (released automatically when the process
    ends). The screen may still turn off; closing a laptop lid can still sleep, depending on its settings."""
    if os.name == 'nt':
        try:
            import ctypes
            ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        except Exception:
            pass


def clean_name(path):
    """File name without its TIFF extensions ('cells_0.5ms_1.ome.tif' -> 'cells_0_5ms_1')."""
    stem = re.sub(r'(\.ome)?\.tiff?$', '', Path(path).name, flags=re.I).replace('_MMStack_Pos0', '')
    return re.sub(r'[^A-Za-z0-9_-]+', '_', stem).strip('_') or 'movie'


def mm_metadata(path):
    """Micro-Manager facts embedded in an OME-TIFF: frame interval, crop ROI, sensor width, exposure."""
    out = {}
    try:
        import tifffile
        with tifffile.TiffFile(str(path)) as t:
            if not t.is_micromanager:
                return out
            summary = (t.micromanager_metadata or {}).get('Summary', {}) or {}
            if summary.get('Interval_ms'):
                out['interval_ms'] = float(summary['Interval_ms'])
            tag = t.pages[0].tags.get('MicroManagerMetadata')
            frame = tag.value if tag is not None and isinstance(tag.value, dict) else {}
            m = re.fullmatch(r'(\d+)-(\d+)-(\d+)-(\d+)', str(frame.get('ROI', '')))
            if m:
                out['roi'] = [int(v) for v in m.groups()]
            for key, name in (('Camera-1-X-dimension', 'sensor_width_px'), ('Exposure-ms', 'exposure_ms')):
                try:
                    out[name] = float(frame[key])
                except (KeyError, TypeError, ValueError):
                    pass
    except Exception:
        pass
    return out


def guess_z_step(path, meta=None):
    """(µm per calibration plane, how it was found) from the stage speed in the name ('25ums') times the
    frame interval (embedded Micro-Manager metadata, or '0.04s' in the name); (None, reason) if unknown."""
    name = Path(path).name
    speed = re.search(r'(?<![\d.])(\d+(?:\.\d+)?)\s*um/?s', name, re.I)
    if not speed:
        return None, 'no stage speed (e.g. "25ums") in the file name'
    meta = mm_metadata(path) if meta is None else meta
    side = sidecar_interval_ms(path)
    if meta.get('interval_ms'):
        interval, src = meta['interval_ms']/1000, 'frame interval from the file metadata'
    elif side:   # e.g. a processed copy (*_corrected.tif) that lost the embedded metadata
        interval, src = side[0]/1000, f'frame interval from {side[1]}'
    else:
        m = re.search(r'(?<![\d.])(\d*\.?\d+)\s*s(?:_|$)', name)
        if not m:
            return None, 'no frame interval in the metadata or file name'
        interval, src = float(m.group(1)), 'frame interval from the file name'
    step = float(speed.group(1))*interval
    return round(step, 4), f'stage speed {speed.group(1)} µm/s from the file name × {src} ({interval*1000:g} ms)'


def guess_objective(path):
    """Magnification from '10x' / '20X' etc. in the file name or its folders (None if absent)."""
    for part in reversed(Path(path).resolve().parts):
        m = re.search(r'(?<![0-9])(4|10|20|40|60|63|100)\s*[xX](?![a-zA-Z0-9])', part) or \
            re.search(r'(?<![0-9])(4|10|20|40|60|63|100)[xX]', part)
        if m:
            return int(m.group(1))
    return None


def sidecar_metadata(path):
    """Micro-Manager *_metadata.txt files that belong to this TIFF (same base name before _MMStack / _crop)."""
    path = Path(path)
    base = re.split(r'_(MMStack|crop|cropped)', path.name.split('.')[0])[0]
    out = []
    for meta in sorted(path.parent.glob('*_metadata.txt')):
        mbase = re.split(r'_(MMStack|crop|cropped)', meta.name.replace('_metadata.txt', ''))[0]
        if mbase and (mbase == base or base.startswith(mbase) or mbase.startswith(base)):
            out.append(meta)
    return out


def sidecar_interval_ms(path):
    """(frame interval in ms, file name) from a matching *_metadata.txt, or None."""
    for meta in sidecar_metadata(path):
        with open(meta, encoding='utf-8', errors='ignore') as fh:
            for line in fh:
                m = re.search(r'"Interval_ms"\s*:\s*([0-9.]+)', line)
                if m and float(m.group(1)) > 0:
                    return float(m.group(1)), meta.name
    return None


def micromanager_roi(path, meta=None):
    """[x, y, w, h] of the camera ROI: embedded Micro-Manager metadata, else a matching *_metadata.txt, else None."""
    meta = mm_metadata(path) if meta is None else meta
    if meta.get('roi'):
        return meta['roi']
    for meta in sidecar_metadata(path):
        with open(meta, encoding='utf-8', errors='ignore') as fh:
            for line in fh:
                m = re.search(r'"ROI"\s*:\s*"(\d+)-(\d+)-(\d+)-(\d+)"', line)
                if m:
                    return [int(v) for v in m.groups()]
    return None


def suggested_planes(frames, objective, z_step=1., margin_um=5.):
    """Expected useful calibration planes (1-based, inclusive): the z range around the middle plus a margin."""
    z_range = DEFAULT_Z_RANGE.get(objective)
    if not z_range:
        return [1, int(frames)]
    half = int(np.ceil((z_range+margin_um)/z_step))
    middle = (frames+1)/2
    return [max(1, int(np.floor(middle-half))), min(int(frames), int(np.ceil(middle+half)))]


def objective_source(path):
    """Which file/folder name the magnification was read from (for display), or None."""
    for part in reversed(Path(path).resolve().parts):
        if re.search(r'(?<![0-9])(4|10|20|40|60|63|100)\s*[xX]', part):
            return part
    return None


def inspect(path):
    reader = P.stack_reader(path)
    frames, (h, w) = P.frame_count(path), reader.shape[-2:]
    objective = guess_objective(path)
    meta = mm_metadata(path)
    z_step, z_step_source = guess_z_step(path, meta)
    return dict(path=str(Path(path).resolve()), frames=int(frames), height=int(h), width=int(w),
                dtype=str(getattr(reader, 'dtype', 'uint16')), objective_guess=f'{objective}x' if objective else None,
                objective_source=objective_source(path) if objective else None,
                pixel_size_guess_um=OBJECTIVE_PIXEL_UM.get(objective), micromanager_roi=micromanager_roi(path, meta),
                sensor_width_px=meta.get('sensor_width_px'), exposure_ms=meta.get('exposure_ms'),
                z_step_guess_um=z_step, z_step_source=z_step_source,
                suggested_planes=suggested_planes(frames, objective, z_step or 1.))


def check_calibration(path, workers=3):
    """Is this a usable calibration z-stack? Beads and lobe angles on three planes (25/50/75 %).

    A z-stack must show enough beads and lobes that rotate between planes; a movie of beads that
    barely move in z shows (nearly) the same angle everywhere.
    """
    n = P.frame_count(path)
    picks = sorted({int(round(v)) for v in np.linspace(.25*(n-1), .75*(n-1), 3)})
    with ProcessPoolExecutor(max_workers=min(workers, len(picks))) as pool:
        results = list(pool.map(_probe_angles, [(str(path), i) for i in picks]))
    beads = [r[0] for r in results]
    snr = float(np.median([s for r in results for s in r[1]])) if any(r[1] for r in results) else 0.
    angles = [r[2] for r in results]
    # summed over the two half-steps, so a ~180° turn is not mistaken for none (angles wrap at 180°)
    rotation = float(sum(abs(P.axial_difference(b, a)) for a, b in zip(angles, angles[1:]))) \
        if len(angles) > 1 and all(np.isfinite(angles)) else float('nan')
    beads_per_plane = float(np.median(beads))
    if beads_per_plane < MIN_CALIBRATION_BEADS:
        verdict, message = 'too_dim', (f'Too dim: {beads_per_plane:.0f} beads per plane (at least {MIN_CALIBRATION_BEADS} '
                                       'needed). Choose a longer-exposure calibration of the same objective.')
    elif n < 10 or not np.isfinite(rotation) or rotation < 10:
        verdict, message = 'not_zscan', (f'This does not look like a z-scan: the lobes rotate only {rotation:.0f}° between '
                                         f'planes {picks[0]+1} and {picks[-1]+1}. Choose the bead calibration stack.')
    else:
        verdict, message = 'ok', (f'Good: about {beads_per_plane:.0f} beads per plane; the lobes rotate {rotation:.0f}° '
                                  f'between planes {picks[0]+1} and {picks[-1]+1}.')
    return dict(verdict=verdict, message=message, beads_per_plane=beads_per_plane, median_snr=snr,
                rotation_deg=rotation, planes=[p+1 for p in picks])


def calibration_cache_dir(calibration):
    key = hashlib.sha1(str(Path(calibration).resolve()).encode()).hexdigest()[:8]
    return ROOT/'calibrations'/f'{clean_name(calibration)}_{key}'


def calibration_cache_state(calibration):
    """('current', n) when the cached per-plane fits match today's fitting code and settings,
    ('stale', n) when they would be refitted, ('none', 0) without a cache."""
    key = hashlib.sha1(str(Path(calibration).resolve()).encode()).hexdigest()[:8]
    for d in sorted((ROOT/'calibrations').glob(f'*_{key}'), key=lambda p: p.stat().st_mtime, reverse=True):
        manifest = d/'cache_calibration'/'fit_manifest.json'
        if not manifest.is_file():
            continue
        n = len(list((d/'cache_calibration').glob('frame_*.npz')))
        try:
            stamp = P.cache_stamp(Path(calibration), P.Config(), None)
            saved = json.loads(manifest.read_text())
            current = all(saved.get(k) == json.loads(json.dumps(v)) for k, v in stamp.items())
        except Exception:
            current = False
        return ('current' if current else 'stale'), n
    return 'none', 0


def recent_calibrations(limit=8):
    """Calibration stacks used before (newest first), whether they still exist and whether their fit is cached."""
    seen = {}
    run_dirs = [d for d in dict.fromkeys((user_settings.results_dir(), ROOT/'runs')) if d.is_dir()]
    for s in [s for d in run_dirs for s in d.glob('*/analyze_settings.json')]:
        try:
            cal = json.loads(s.read_text()).get('calibration')
        except Exception:
            continue
        if cal:
            seen[cal] = max(seen.get(cal, 0), s.stat().st_mtime)
    for s in list(ROOT.glob('results*/run_info.json'))+[s for d in run_dirs for s in d.glob('*/run_info.json')]:
        try:
            cal = json.loads(s.read_text()).get('calibration', {}).get('path')
        except Exception:
            continue
        if cal:
            seen[cal] = max(seen.get(cal, 0), s.stat().st_mtime)
    for m in (ROOT/'calibrations').glob('*/cache_calibration/fit_manifest.json'):
        try:
            cal = json.loads(m.read_text()).get('path')
        except Exception:
            continue
        if cal:
            seen[cal] = max(seen.get(cal, 0), m.stat().st_mtime)
    out = []
    for cal, t in sorted(seen.items(), key=lambda kv: -kv[1])[:limit]:
        exists = Path(cal).is_file()
        state, planes = calibration_cache_state(cal) if exists else ('none', 0)
        obj = guess_objective(cal)
        out.append(dict(path=cal, name=Path(cal).name, last_used=t, exists=exists, cache=state, cached_planes=planes,
                        objective_guess=f'{obj}x' if obj else None))
    return out


def check_movie(path):
    """Quick brightness check of a movie on 3 frames: bright (standard settings) or dim (dim-data settings)."""
    beads, snr = signal_check(path, DIM_SETTINGS, frames=3, workers=3, crop=1024)
    dim = snr < DIM_SNR
    obj = guess_objective(path)
    if beads < 3:
        verdict, message = 'empty', f'Almost no beads found ({beads:.0f} per frame). Is this a bead movie?'
    elif dim:
        verdict, message = 'dim', f'Dim (median lobe SNR {snr:.1f}): analysed with the dim-data settings. ~{beads:.0f} beads per frame.'
    else:
        verdict, message = 'bright', f'Bright (median lobe SNR {snr:.1f}). ~{beads:.0f} beads per frame.'
    return dict(verdict=verdict, message=message, beads_per_frame=beads, median_snr=snr, dim=bool(dim),
                objective_guess=f'{obj}x' if obj else None)


def _probe_angles(task):
    path, index = task
    cfg = P.Config()
    rows, _ = P.localize_image(P.read_frame(path, index), cfg, None, index+1, seed=P.seed_image(path, index, cfg))
    if not len(rows):
        return 0, [], float('nan')
    # median axial angle (doubled-angle mean handles the 0/180 wrap)
    a = np.radians(2*rows[:, 6])
    angle = float(np.degrees(np.arctan2(np.median(np.sin(a)), np.median(np.cos(a))))/2 % 180)
    return len(rows), rows[:, P.C['minLobeSNR']].tolist(), angle


def preview(path, frame):
    """Detections on one frame with a permissive threshold; each bead carries its own lobe SNR.

    A front end filters by SNR instantly (e.g. a slider) instead of re-detecting.
    """
    index = int(frame)-1
    cfg = P.Config(min_snr=1.5)
    image = P.read_frame(path, index)
    rows, stats = P.localize_image(image, cfg, None, index+1, seed=P.seed_image(path, index, cfg))
    snr = rows[:, P.C['minLobeSNR']]
    beads = [dict(x1=float(r[0]), y1=float(r[3]), x2=float(r[1]), y2=float(r[4]), x=float(r[2]), y=float(r[5]),
                  angle=float(r[6]), sep=float(r[P.C['lobeSeparationPixels']]), snr=float(s)) for r, s in zip(rows, snr)]
    median = float(np.median(snr)) if len(snr) else 0.
    suggested = DIM_SETTINGS['min_snr'] if median < DIM_SNR else P.Config().min_snr
    return dict(frame=index+1, width=int(image.shape[1]), height=int(image.shape[0]), noise=float(stats['pixel_noise']),
                median_snr=median, beads=beads, suggested_min_snr=suggested, default_min_snr=P.Config().min_snr)


def _probe(task):
    path, index, settings, crop = task
    cfg = P.Config(**settings)
    image, seed = P.read_frame(path, index), P.seed_image(path, index, cfg)
    scale = 1.
    if crop and max(image.shape) > crop:   # central region only: enough to judge brightness, much faster
        h, w = image.shape
        y0, x0 = max((h-crop)//2, 0), max((w-crop)//2, 0)
        sl = (slice(y0, y0+crop), slice(x0, x0+crop))
        scale = h*w/float(image[sl].size)
        image, seed = image[sl], (seed[sl] if seed is not None else None)
    rows, _ = P.localize_image(image, cfg, None, index+1, seed=seed)
    return len(rows)*scale, rows[:, P.C['minLobeSNR']].tolist()


def signal_check(path, settings, frames=3, workers=3, crop=None):
    """(beads per frame, median lobe SNR) on a few frames, without calibration (optionally a central crop)."""
    n = P.frame_count(path)
    picks = sorted({int(round(v)) for v in np.linspace(.25*(n-1), .75*(n-1), frames)})
    with ProcessPoolExecutor(max_workers=min(workers, len(picks))) as pool:
        results = list(pool.map(_probe, [(str(path), i, settings, crop) for i in picks]))
    counts = [c for c, _ in results]
    snr = [s for _, v in results for s in v]
    return float(np.median(counts)), float(np.median(snr)) if snr else 0.


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('files', nargs='*', type=Path, help='CALIBRATION.tif MOVIE.tif [MOVIE2.tif ...]')
    ap.add_argument('--calibration', type=Path)
    ap.add_argument('--movie', action='append', type=Path, default=[])
    ap.add_argument('--inspect', type=Path, help='print what is inferred about a file and exit')
    ap.add_argument('--preview', type=Path, help='print one-frame detections (JSON, permissive threshold) and exit')
    ap.add_argument('--frame', type=int, help='frame for --preview (1-based; default: middle)')
    ap.add_argument('--check-calibration', type=Path, help='print whether a file is a usable calibration z-stack (JSON) and exit')
    ap.add_argument('--check-movie', type=Path, help='print a quick brightness check of a movie (JSON) and exit')
    ap.add_argument('--recent-calibrations', action='store_true', help='print calibration stacks used before (JSON) and exit')
    ap.add_argument('--calibration-planes', metavar='A:B',
                    help="calibration planes to use, 1-based inclusive (default: suggested from the objective; 'all')")
    ap.add_argument('--min-snr', type=float, help='minimum lobe SNR in the movies (overrides the automatic dim check)')
    ap.add_argument('--camera', type=Path, help='camera.npz (camera_calibration.py): sCMOS maximum-likelihood fitting')
    ap.add_argument('--pixel-size', type=float, help='µm per pixel (default: from the objective in the name)')
    ap.add_argument('--z-step', type=float, help='µm between calibration planes (default: stage speed in the name × '
                                                'frame interval, e.g. 25ums × 40 ms = 1 µm; else 1)')
    ap.add_argument('--z-range', type=float, help='report z within +- this many µm (default: 75 for 10x, else all)')
    ap.add_argument('--true-depth', action='store_true',
                    help=f'report z as true depth in a watery sample imaged with an air objective (z x {WATER_AIR_SCALE}; '
                         'refractive-index focal shift, paraxial). Default: stage units')
    ap.add_argument('--axial-scale', type=float, help='multiply reported z by this factor (overrides --true-depth)')
    ap.add_argument('--frame-interval', type=float, help='ms between movie frames (motion results per second instead of per frame)')
    ap.add_argument('--no-motion', action='store_true', help='skip the motion analysis of the tracks')
    ap.add_argument('--name', help='run name (default: first movie name)')
    ap.add_argument('--output', type=Path, help='output folder (default: DHPSF_pipeline/runs/<name>_<date-time>)')
    ap.add_argument('--status-file', type=Path, help='progress JSON (default: <output>/status.json)')
    ap.add_argument('--workers', type=int, default=max(1, min(16, (os.cpu_count() or 2)-4)))
    ap.add_argument('--no-matlab', action='store_true', help='skip the native MATLAB table export')
    ap.add_argument('--dim', choices=['auto', 'yes', 'no'], default='auto', help='dim-data settings (default: auto)')
    ap.add_argument('--open', action='store_true', help='open the results in the explorer when done')
    args = ap.parse_args(argv)

    if args.inspect:
        print(json.dumps(inspect(args.inspect)))
        return 0
    if args.preview:
        frame = args.frame or (P.frame_count(args.preview)+1)//2
        print(json.dumps(preview(args.preview, frame)))
        return 0
    if args.check_calibration:
        result = check_calibration(args.check_calibration)
        result['cache'], result['cached_planes'] = calibration_cache_state(args.check_calibration)
        print(json.dumps(result))
        return 0
    if args.check_movie:
        print(json.dumps(check_movie(args.check_movie)))
        return 0
    if args.recent_calibrations:
        print(json.dumps({'calibrations': recent_calibrations()}))
        return 0
    calibration = args.calibration or (args.files[0] if args.files else None)
    movies = args.movie or (args.files[1:] if args.calibration is None else args.files)
    if calibration is None or not movies:
        ap.error('give a calibration z-stack and at least one movie')
    keep_awake()
    for f in [calibration, *movies]:
        if not Path(f).is_file():
            ap.error(f'file not found: {f}')

    guesses = {Path(f).name: guess_objective(f) for f in [calibration, *movies]}
    known = {g for g in guesses.values() if g}
    if len(known) > 1 and not args.pixel_size:
        ap.error('the calibration and movies seem to come from different objectives ('
                 + ', '.join(f'{n}: {g}x' for n, g in guesses.items() if g)
                 + '); a calibration only applies to movies taken with the same objective. '
                   'Pass --pixel-size if this is intended.')
    objective = guess_objective(movies[0]) or guess_objective(calibration)
    pixel = args.pixel_size or OBJECTIVE_PIXEL_UM.get(objective)
    if pixel is None:
        ap.error('cannot infer the pixel size from the file names; pass --pixel-size (µm per pixel)')
    z_range = args.z_range if args.z_range is not None else DEFAULT_Z_RANGE.get(objective)
    cal_meta = mm_metadata(calibration)
    if args.z_step is None:
        guessed, why = guess_z_step(calibration, cal_meta)
        if not guessed:   # never assume: a wrong z step scales every z by the same wrong factor
            ap.error(f'cannot work out the z step between calibration planes ({why}); pass --z-step '
                     '(µm; stage speed x frame interval, e.g. 25 µm/s x 0.04 s = 1)')
        args.z_step = guessed
        z_step_note = why
    else:
        z_step_note = 'given'
    name = clean_name(args.name or movies[0])
    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M')
    out = (args.output or user_settings.results_dir()/f'{name}_{stamp}').resolve()
    out.mkdir(parents=True, exist_ok=True)
    status = args.status_file or out/'status.json'
    P.PROGRESS.open(status)
    P.PROGRESS.update('checking', 0., 'checking the calibration and movie signal', output=out)
    print(f'Output: {out}\nObjective: {objective or "unknown"}x, pixel size {pixel} µm, z step {args.z_step} µm '
          f'({z_step_note}), z range {"+-%g µm" % z_range if z_range else "full"}', flush=True)

    try:
        # The calibration must be a z-scan, bright enough with the standard settings.
        check = check_calibration(calibration)
        print(f'Calibration check: {check["message"]}', flush=True)
        if check['verdict'] == 'too_dim':
            raise RuntimeError(check['message'] + ' The angle-z relation does not depend on exposure time.')
        if check['verdict'] == 'not_zscan':
            raise RuntimeError(check['message'])
        movie_settings, per_movie, names = {}, {}, {}
        used = set()
        for movie in movies:
            mname = clean_name(movie)
            while mname in used:
                mname += '_2'
            used.add(mname)
            names[movie] = mname
        if args.min_snr is not None:
            movie_settings = dict(min_snr=args.min_snr, recover_min_snr=min(P.Config().recover_min_snr, .75*args.min_snr))
            print(f'Movie detection threshold set by user: min lobe SNR {args.min_snr:g}', flush=True)
        elif args.dim == 'yes':
            movie_settings = dict(DIM_SETTINGS)
        elif args.dim == 'auto':
            # each movie on its own: a dim 1 ms movie next to a bright 5 ms one needs its own threshold
            for movie in movies:
                _, movie_snr = signal_check(movie, DIM_SETTINGS, frames=3, workers=3)
                dim = movie_snr < DIM_SNR
                print(f'Movie check {names[movie]}: median lobe SNR {movie_snr:.1f}'
                      + (f' -> dim, using dim-data settings {DIM_SETTINGS}' if dim else ''), flush=True)
                if dim:
                    per_movie[names[movie]] = dict(DIM_SETTINGS)
            if per_movie and len(per_movie) == len(movies):
                movie_settings, per_movie = dict(DIM_SETTINGS), {}

        cal_cache = calibration_cache_dir(calibration)
        key = cal_cache.name.rsplit('_', 1)[-1]
        # reuse an existing cache of this stack even if it was named under an older naming rule
        older = [d for d in (ROOT/'calibrations').glob(f'*_{key}') if d != cal_cache]
        if not cal_cache.exists() and older:
            cal_cache = max(older, key=lambda d: d.stat().st_mtime)
        argv_p = ['--output', str(out), '--calibration', str(calibration), '--pixel-size', str(pixel),
                  '--z-step', str(args.z_step), '--status-file', str(status), '--calibration-cache', str(cal_cache),
                  '--workers', str(args.workers)]
        if z_range:
            argv_p += ['--z-range', str(z_range)]
        axial_scale = args.axial_scale or (WATER_AIR_SCALE if args.true_depth else 1.)
        if axial_scale != 1.:
            argv_p += ['--axial-scale', str(axial_scale)]
            print(f'z reported as true depth: stage µm x {axial_scale:g}', flush=True)
        cal_frames = P.frame_count(calibration)
        if args.calibration_planes and args.calibration_planes.lower() != 'all':
            planes = args.calibration_planes
        elif args.calibration_planes is None:
            a, b = suggested_planes(cal_frames, objective, args.z_step)
            planes = f'{a}:{b}' if (a, b) != (1, cal_frames) else None
        else:
            planes = None
        if planes:
            argv_p += ['--calibration-planes', planes]
            print(f'Calibration planes {planes} of {cal_frames}', flush=True)
        if movie_settings or per_movie:
            argv_p += ['--movie-config', json.dumps({**movie_settings, **({'per_movie': per_movie} if per_movie else {})})]
        if cal_meta.get('sensor_width_px'):
            argv_p += ['--sensor-size', str(int(cal_meta['sensor_width_px']))]
        if args.no_matlab:
            argv_p.append('--no-matlab')
        if args.no_motion:
            argv_p.append('--no-motion')
        if args.frame_interval:
            argv_p += ['--frame-interval', str(args.frame_interval)]
        if args.camera:
            argv_p += ['--camera', str(args.camera)]
            print(f'Camera noise model: {args.camera} (maximum-likelihood fitting)', flush=True)
        for movie in movies:
            mname = names[movie]
            argv_p += ['--movie', f'{mname}={movie}']
            roi = micromanager_roi(movie)
            h, w = P.stack_reader(movie).shape[-2:]
            if roi and roi[2] == w and roi[3] == h:
                argv_p += ['--offset', f'{mname}={roi[0]},{roi[1]}']
        roi = micromanager_roi(calibration)
        h, w = P.stack_reader(calibration).shape[-2:]
        if roi and roi[2] == w and roi[3] == h:
            argv_p += ['--offset', f'calibration={roi[0]},{roi[1]}']
        (out/'analyze_settings.json').write_text(json.dumps(dict(
            calibration=str(Path(calibration).resolve()), movies=[str(Path(m).resolve()) for m in movies],
            objective=objective, pixel_size_um=pixel, z_step_um=args.z_step, z_step_source=z_step_note, z_range_um=z_range,
            axial_scale=axial_scale,
            calibration_check=check, movie_settings=movie_settings, per_movie_settings=per_movie,
            calibration_cache=str(cal_cache), pipeline_arguments=argv_p), indent=2))
    except Exception as exc:
        P.PROGRESS.update('error', 1., str(exc), state='error', error=str(exc), output=out)
        print(f'ERROR: {exc}', file=sys.stderr, flush=True)
        return 1

    try:
        P.main(argv_p)
    except Exception as exc:   # pipeline.main already wrote the error status
        print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        return 1
    print(f'Done: {out}', flush=True)
    if args.open:
        port = 8780
        subprocess.Popen([sys.executable, str(ROOT/'explorer'/'app.py'), '--results', str(out), '--port', str(port)])
        import webbrowser
        webbrowser.open(f'http://127.0.0.1:{port}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
