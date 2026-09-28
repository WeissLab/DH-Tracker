"""Re-reference the whole-field drift of existing result folders from frame 1 to the movie's median.

pipeline.estimate_drift now reports the drift relative to its median over the movie instead of
relative to frame 1 (frame 1 is one sample of the stage shake, so a frame-1 reference offsets the
curve and puts stabilized positions at one shaken frame). The reference is a constant per movie
and coordinate, so this script shifts the files written before the change, in place:

  *_drift.csv                                   dx, dy, dz, dzRaw   minus their median over frames
  *_localizations.csv, *_rejected.csv           x/y/zStabilized     plus the same medians
  *_payload.mat                                 driftMatrix, stabilizedMatrix, metadata note
  *_localizations.mat (native MATLAB tables)    regenerated with export_tables.m if MATLAB exists
  summary.json                                  max_abs_drift_px, max_abs_drift_z_um
  exports/*.csv, exports/*.mat                  x/y/zStabilized columns of track exports

Displacements, motion analysis (offsets and velocities) and deformation exports do not depend on
the reference and are left alone. Changed files are copied to <folder>/_drift_frame1_backup/ first;
a folder is marked with drift_reference.json and skipped if already done.

Usage:  python tools/rereference_drift.py FOLDER [FOLDER ...] [--no-matlab] [--dry-run]
"""
import argparse
import csv
import io
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.io

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the pipeline folder (this script is in tools/)
import pipeline as P

STAB = ('xStabilized', 'yStabilized', 'zStabilized')


def fmt(v):
    return '%.6g' % v if np.isfinite(v) else 'nan'


class Shifter:
    def __init__(self, folder, dry):
        self.folder, self.dry, self.changed = folder, dry, []

    def backup(self, path):
        dest = self.folder/'_drift_frame1_backup'/path.relative_to(self.folder)
        if not self.dry and not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
        self.changed.append(str(path.relative_to(self.folder)))

    def read_csv(self, path):
        rows = list(csv.reader(io.StringIO(path.read_text(encoding='utf-8'))))
        return rows[0], rows[1:]

    def write_csv(self, path, header, rows):
        self.backup(path)
        if not self.dry:
            out = io.StringIO()
            csv.writer(out, lineterminator='\n').writerows([header]+rows)
            path.write_text(out.getvalue(), encoding='utf-8')

    def drift(self, path):
        """Shift the drift table to median zero; returns the medians removed {dx, dy, dz, dzRaw}."""
        header, rows = self.read_csv(path)
        med = {}
        for c in ('dx', 'dy', 'dz', 'dzRaw'):
            if c not in header:
                continue
            i = header.index(c)
            v = np.array([float(r[i]) for r in rows])
            med[c] = float(np.nanmedian(v))
            for r, x in zip(rows, v-med[c]):
                r[i] = fmt(x)
        self.write_csv(path, header, rows)
        return med

    def stabilized_csv(self, path, med):
        header, rows = self.read_csv(path)
        idx = [(header.index(c), m) for c, m in zip(STAB, (med['dx'], med['dy'], med.get('dz', 0.))) if c in header]
        if not idx or not rows:
            return
        for r in rows:
            for i, m in idx:
                if i < len(r) and r[i].strip():
                    r[i] = fmt(float(r[i])+m)
        self.write_csv(path, header, rows)

    def payload(self, path, med):
        s = scipy.io.loadmat(path)
        keep = {k: v for k, v in s.items() if not k.startswith('__')}
        m = np.array([med['dx'], med['dy'], med.get('dz', 0.)])
        if 'driftMatrix' in keep and keep['driftMatrix'].size:
            keep['driftMatrix'][:, 1:4] -= m
        if 'stabilizedMatrix' in keep and keep['stabilizedMatrix'].size:
            keep['stabilizedMatrix'][:, :3] += m
        if 'metadataJSON' in keep:
            meta = json.loads(str(np.ravel(keep['metadataJSON'])[0]))
            meta['drift_reference'] = 'median over the movie'
            keep['metadataJSON'] = json.dumps(meta)
        for k in ('columnNames', 'qualityNames'):
            if k in keep:
                keep[k] = np.asarray([str(np.ravel(v)[0]).strip() if np.size(v) else '' for v in np.ravel(keep[k])], dtype=object)
        self.backup(path)
        if not self.dry:
            scipy.io.savemat(path, keep, do_compression=True)

    def export_mat(self, path, med):
        s = scipy.io.loadmat(path)
        hit = [c for c in STAB if c in s]
        if not hit:
            return
        keep = {k: v for k, v in s.items() if not k.startswith('__')}
        for c, m in zip(STAB, (med['dx'], med['dy'], med.get('dz', 0.))):
            if c in keep:
                keep[c] = keep[c]+m
        self.backup(path)
        if not self.dry:
            scipy.io.savemat(path, keep, do_compression=True)


def rereference(folder, dry=False, matlab=None):
    folder = Path(folder).resolve()
    if (folder/'drift_reference.json').is_file():
        print(f'{folder}: already referenced to the median, skipped')
        return {}
    drifts = sorted(folder.glob('*_drift.csv'))
    if not drifts:
        print(f'{folder}: no drift files')
        return {}
    c = Shifter(folder, dry)
    shifts = {}
    summary = json.loads((folder/'summary.json').read_text()) if (folder/'summary.json').is_file() else None
    for dp in drifts:
        movie = dp.name[:-len('_drift.csv')]
        med = c.drift(dp)
        shifts[movie] = med
        for suffix in ('_localizations.csv', '_rejected.csv', '_payload.csv'):
            p = folder/f'{movie}{suffix}'
            if p.is_file():
                c.stabilized_csv(p, med)
        if (folder/f'{movie}_payload.mat').is_file():
            c.payload(folder/f'{movie}_payload.mat', med)
        ex = folder/'exports'
        if ex.is_dir():
            for p in sorted(ex.glob(f'*_{movie}_*.csv')):
                c.stabilized_csv(p, med)
            for p in sorted(ex.glob(f'*_{movie}_*.mat')):
                c.export_mat(p, med)
        if summary and movie in summary.get('movies', {}):
            h, rows = c.read_csv(dp) if not dry else (None, None)
            if rows is not None:
                d = np.array([[float(r[h.index(k)]) for k in ('dx', 'dy', 'dz') if k in h] for r in rows])
                summary['movies'][movie]['max_abs_drift_px'] = float(np.abs(d[:, :2]).max())
                if d.shape[1] > 2:
                    summary['movies'][movie]['max_abs_drift_z_um'] = float(np.abs(d[:, 2]).max())
    if summary is not None:
        summary['drift_reference'] = 'median over the movie'
        c.backup(folder/'summary.json')
        if not dry:
            (folder/'summary.json').write_text(json.dumps(summary, indent=2))
    if not dry:
        (folder/'drift_reference.json').write_text(json.dumps(dict(
            reference='median over the movie', previous='frame 1', shifted=datetime.now().isoformat(timespec='seconds'),
            removed_medians=shifts, units='dx, dy px; dz, dzRaw µm', files=c.changed), indent=2))
        tables = sorted(folder.glob('*_localizations.mat'))
        if tables and any(folder.glob('*_payload.mat')):
            if matlab and Path(matlab).exists():
                for t in tables:
                    c.backup(t)
                cmd = "addpath('"+str(P.ROOT).replace("'", "''")+"'); export_tables('"+str(folder).replace("'", "''")+"');"
                subprocess.run([matlab, '-batch', cmd], check=True, timeout=900)
            else:
                print(f'{folder}: MATLAB not found; run export_tables.m to regenerate {len(tables)} native table file(s)')
    print(f"{folder}: {'would change' if dry else 'shifted'} {len(c.changed)} file(s); medians removed "
          + '; '.join(f"{m}: " + ', '.join(f'{k} {v:+.3f}' for k, v in s.items()) for m, s in shifts.items()))
    return shifts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('folders', nargs='+', type=Path)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--no-matlab', action='store_true')
    ap.add_argument('--matlab', default=None, help='matlab executable (default: found automatically)')
    a = ap.parse_args(argv)
    for f in a.folders:
        rereference(f, a.dry_run, None if a.no_matlab else (a.matlab or P.find_matlab()))


if __name__ == '__main__':
    main()
