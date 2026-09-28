"""Injection-recovery benchmark: real calibration-bead PSFs added to real movie frames.

Isolated calibration beads (known z plane, fitted midpoint) are background-subtracted,
scaled to movie brightness and added at random positions to experimental frames:
isolated, overlapping an existing bead (crowded), or near the image edge. Recall is
the fraction of injected beads localized within 2 px; z error uses the plane depth.
Run from DHPSF_pipeline:  python tools/benchmark_injection.py --results runs/<run folder>
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the pipeline folder (this script is in tools/)
import pipeline as P

HALF = 24


def build_bank(cal, stack, min_isolation=45.):
    tracks = cal['tracks']
    sel = tracks[np.isin(tracks[:, 9], cal['selected_track_ids'])]
    bank = []
    for plane in range(1, stack.shape[0]+1):
        allp = tracks[tracks[:, 8] == plane]
        tree = cKDTree(allp[:, [2, 5]])
        image = np.asarray(stack[plane-1], dtype=np.float32)
        for r in sel[sel[:, 8] == plane]:
            if len(tree.query_ball_point(r[[2, 5]], min_isolation)) > 1:
                continue
            cx, cy = r[2]-1, r[5]-1
            ix, iy = int(round(cx)), int(round(cy))
            if ix < HALF or iy < HALF or ix+HALF >= image.shape[1] or iy+HALF >= image.shape[0]:
                continue
            crop = image[iy-HALF:iy+HALF+1, ix-HALF:ix+HALF+1].copy()
            border = np.r_[crop[0], crop[-1], crop[:, 0], crop[:, -1]]
            crop -= np.median(border)
            bank.append(dict(crop=crop, dx=cx-ix, dy=cy-iy, z=float(cal['z'][plane-1]), peak=float(crop.max())))
    return bank


def inject(image, dets, bank, rng, n_iso, n_crowd, n_edge):
    h, w = image.shape
    out = image.astype(np.float32).copy()
    truth, placed = [], [p for p in dets]
    def free(x, y, rmin):
        return all((x-a)**2+(y-b)**2 >= rmin**2 for a, b in placed)
    def put(kind, x, y):
        item = bank[rng.integers(len(bank))]
        scale = np.exp(rng.uniform(np.log(25), np.log(450)))/item['peak']
        ix, iy = int(x), int(y)
        y0, y1, x0, x1 = iy-HALF, iy+HALF+1, ix-HALF, ix+HALF+1
        cy0, cx0 = max(0, -y0), max(0, -x0)
        cy1, cx1 = (2*HALF+1)-max(0, y1-h), (2*HALF+1)-max(0, x1-w)
        out[max(y0, 0):min(y1, h), max(x0, 0):min(x1, w)] += scale*item['crop'][cy0:cy1, cx0:cx1]
        truth.append((ix+item['dx'], iy+item['dy'], item['z'], scale*item['peak'], kind))
        placed.append((ix+item['dx'], iy+item['dy']))
    for _ in range(n_iso):
        for _ in range(200):
            x, y = rng.uniform(40, w-40), rng.uniform(40, h-40)
            if free(x, y, 45):
                put('isolated', x, y); break
    for _ in range(n_crowd):
        for _ in range(200):
            a, b = dets[rng.integers(len(dets))]
            ang, d = rng.uniform(0, 2*np.pi), rng.uniform(10, 26)
            x, y = a+d*np.cos(ang), b+d*np.sin(ang)
            others = [p for p in placed if (p[0]-a)**2+(p[1]-b)**2 > 1]
            if 30 < x < w-30 and 30 < y < h-30 and all((x-p[0])**2+(y-p[1])**2 >= 30**2 for p in others):
                put('crowded', x, y); break
    for _ in range(n_edge):
        for _ in range(200):
            side = rng.integers(4)
            t, e = rng.uniform(40, w-40), rng.uniform(4, 16)
            x, y = [(e, t), (w-1-e, t), (t, e), (t, h-1-e)][side]
            if free(x, y, 45):
                put('edge', x, y); break
    return out, truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', default='5,20,36,50,65')
    ap.add_argument('--results', type=Path, required=True, help='a run folder, e.g. runs/<name>')
    ap.add_argument('--config', type=Path)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()
    cfg = P.Config(**(json.loads(args.config.read_text()) if args.config else {}))
    with np.load(args.results/'calibration.npz') as z:
        cal = {k: z[k] for k in z.files}
    base_model = P.CalibrationModel(cal['dense_z'], cal['dense_a'], cal['dense_sep'], cal.get('z_limits'))
    info_path, summary_path = args.results/'run_info.json', args.results/'summary.json'
    info = json.loads(info_path.read_text()) if info_path.exists() else \
        dict(calibration=dict(path=str(P.CALIBRATION)), sensor_size_px=2048,
             movies={n: dict(path=str(p), offset=(0, 0)) for n, p in P.DEFAULT_MOVIES.items()})
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    bank = build_bank(cal, P.stack_reader(info['calibration']['path']))
    rng = np.random.default_rng(args.seed)
    results = []
    for name, movie in info['movies'].items():
        path = movie['path']
        field = summary.get('movies', {}).get(name, {}).get('field_separation', {}).get('coef')
        model = base_model.with_field(field, movie['offset'], info['sensor_size_px']) if field else base_model
        frames = P.frame_count(path)
        for f in [int(v) for v in args.frames.split(',') if int(v) <= frames]:
            image = P.read_frame(path, f-1)
            seed = P.seed_image(path, f-1, cfg)
            base = P.localize_image(image, cfg, model, f, seed=seed)[0]
            dets = [tuple(p) for p in base[:, [2, 5]]-1]
            injected, truth = inject(image, dets, bank, rng, 30, 30, 10)
            # Injected beads are static, like the slow real beads: present in every frame of the seed average.
            seed_injected = None if seed is None else seed+(injected-image)
            rows = P.localize_image(injected, cfg, model, f, seed=seed_injected)[0]
            xy, angle, sep = rows[:, [2, 5]]-1, rows[:, 6], rows[:, C_SEP]
            base_xy = dets
            z, st = model.angle_to_z(angle, sep)
            tree = cKDTree(xy)
            used = set()
            for tx, ty, tz, amp, kind in truth:
                d, j = tree.query([tx, ty])
                hit = d <= 2.
                results.append(dict(movie=name, frame=f, kind=kind, amp=amp, z=tz, hit=bool(hit),
                                    err=float(d), zerr=float(z[j]-tz) if hit and np.isfinite(z[j]) else np.nan))
                if hit:
                    used.add(j)
            # damage: previously found beads lost after injection; extras: unexplained new detections
            base_xy = np.asarray(base_xy).reshape(-1, 2)
            lost = np.sum(tree.query(base_xy)[0] > 2) if len(base_xy) else 0
            extra = [j for j in range(len(xy)) if j not in used]
            extra = np.sum(cKDTree(base_xy).query(xy[extra])[0] > 2) if len(extra) and len(base_xy) else 0
            results.append(dict(movie=name, frame=f, kind='_frame', lost=int(lost), extra=int(extra),
                                base=len(base_xy)))
    hits = [r for r in results if r['kind'] != '_frame']
    frames = [r for r in results if r['kind'] == '_frame']
    report = {'injected': len(hits),
              'recall': float(np.mean([r['hit'] for r in hits]))}
    for kind in ['isolated', 'crowded', 'edge']:
        k = [r for r in hits if r['kind'] == kind]
        report[f'recall_{kind}'] = float(np.mean([r['hit'] for r in k]))
    for lo, hi in [(25, 50), (50, 100), (100, 200), (200, 450)]:
        k = [r for r in hits if lo <= r['amp'] < hi]
        report[f'recall_peak_{lo}_{hi}'] = float(np.mean([r['hit'] for r in k]))
    for lo, hi in [(-29, -20), (-20, -10), (-10, 0), (0, 10), (10, 20), (20, 29.1)]:
        k = [r for r in hits if lo <= r['z'] < hi]
        report[f'recall_z_{lo}_{hi}'] = float(np.mean([r['hit'] for r in k])) if k else None
    ze = np.array([r['zerr'] for r in hits if r['hit']])
    ze = ze[np.isfinite(ze)]
    report['xy_error_median_px'] = float(np.median([r['err'] for r in hits if r['hit']]))
    report['z_error_median_abs_um'] = float(np.median(np.abs(ze)))
    report['z_error_p95_abs_um'] = float(np.percentile(np.abs(ze), 95))
    report['z_errors_over_5um'] = int(np.sum(np.abs(ze) > 5))
    report['existing_beads_lost_after_injection'] = int(sum(r['lost'] for r in frames))
    report['unexplained_new_detections'] = int(sum(r['extra'] for r in frames))
    report['baseline_detections'] = int(sum(r['base'] for r in frames))
    print(json.dumps(report, indent=2))
    misses = [r for r in hits if not r['hit']]
    print('misses (first 25):')
    for r in misses[:25]:
        print(' ', r)


C_SEP = P.C['lobeSeparationPixels']

if __name__ == '__main__':
    main()
