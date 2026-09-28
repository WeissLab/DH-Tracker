"""Convert result folders written before z was reported as lab z (height, up +) to the new convention.

Earlier runs reported z along the calibration stack index, which runs down into the sample: an
indentation from above gave positive z. Now z is height, positive upwards (toward the indenter), so
an indentation is negative z (pipeline.to_lab_z). This script flips every z-signed quantity of a
result folder in place, after copying each file it changes to <folder>/_z_stack_backup/:

  *_localizations.csv, *_rejected.csv, *_payload.csv   zMicrons, zStabilized
  *_drift.csv                                           dz, dzRaw
  *_motion_states.csv                                   vz, refine_dz_um
  *_payload.mat                                         z columns, drift, calibration z, metadata
  *_localizations.mat (native MATLAB tables)            regenerated with export_tables.m if MATLAB exists
  calibration.npz                                       via pipeline.to_lab_z
  summary.json, validation.json, run_info.json          z ranges, lateral model, slide tilt, signed errors
  exports/*.csv, exports/*.mat                          zMicrons, zStabilized, zAnalysis (deformation exports
                                                        are already in lab z and are left alone)

Numbers are negated as text (no reformatting). A folder is marked with z_convention.json (and
run_info.json z_convention = "up") and is skipped if already converted.

Usage:  python tools/convert_z_up.py FOLDER [FOLDER ...] [--no-matlab] [--dry-run]
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

CSV_COLUMNS = {'_localizations.csv': ('zMicrons', 'zStabilized'), '_rejected.csv': ('zMicrons', 'zStabilized'),
               '_payload.csv': ('zMicrons', 'zStabilized'), '_drift.csv': ('dz', 'dzRaw'),
               '_motion_states.csv': ('vz', 'refine_dz_um')}
EXPORT_COLUMNS = ('zMicrons', 'zStabilized', 'zAnalysis')
ODD = [i for i, t in enumerate(P.LATERAL_TERMS) if 'z^2' not in t]


def neg_text(s):
    t = s.strip()
    try:
        v = float(t)
    except ValueError:
        return s
    if not np.isfinite(v) or v == 0:
        return s
    return t[1:] if t.startswith('-') else '-'+t.lstrip('+')


def convention(folder):
    for name in ('z_convention.json', 'run_info.json'):
        p = folder/name
        if p.is_file():
            try:
                c = json.loads(p.read_text()).get('z_convention')
                if c:
                    return c
            except (OSError, ValueError):
                pass
    return 'stack'


class Converter:
    def __init__(self, folder, dry):
        self.folder, self.dry, self.changed = folder, dry, []

    def backup(self, path):
        dest = self.folder/'_z_stack_backup'/path.relative_to(self.folder)
        if not self.dry and not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
        self.changed.append(str(path.relative_to(self.folder)))

    def csv_file(self, path, columns, skip_if=()):
        text = path.read_text(encoding='utf-8')
        rows = list(csv.reader(io.StringIO(text)))
        if not rows:
            return
        header = rows[0]
        if any(s in header for s in skip_if):
            return
        idx = [i for i, h in enumerate(header) if h in columns]
        if not idx:
            return
        for r in rows[1:]:
            for i in idx:
                if i < len(r):
                    r[i] = neg_text(r[i])
        self.backup(path)
        if not self.dry:
            out = io.StringIO()
            csv.writer(out, lineterminator='\n').writerows(rows)
            path.write_text(out.getvalue(), encoding='utf-8')

    def payload_mat(self, path):
        s = scipy.io.loadmat(path)
        keep = {k: v for k, v in s.items() if not k.startswith('__')}
        keep['localizationMatrix'][:, 7] *= -1
        if 'stabilizedMatrix' in keep and keep['stabilizedMatrix'].size:
            keep['stabilizedMatrix'][:, 2] *= -1
        if 'driftMatrix' in keep and keep['driftMatrix'].size:
            keep['driftMatrix'][:, 3] *= -1
        if 'calibrationMatrix' in keep:
            keep['calibrationMatrix'][:, 0] *= -1
        if 'interpolatedCalibration' in keep:
            ic = keep['interpolatedCalibration'][::-1].copy()
            ic[:, 0] *= -1
            keep['interpolatedCalibration'] = ic
        if 'metadataJSON' in keep:
            meta = json.loads(str(np.ravel(keep['metadataJSON'])[0]))
            keep['metadataJSON'] = json.dumps(self.metadata(meta))
        for k in ('columnNames', 'qualityNames'):
            if k in keep:
                keep[k] = np.asarray([str(np.ravel(v)[0]).strip() if np.size(v) else '' for v in np.ravel(keep[k])], dtype=object)
        self.backup(path)
        if not self.dry:
            scipy.io.savemat(path, keep, do_compression=True)

    def metadata(self, meta):
        if 'calibration_supported_z_um' in meta:
            lo, hi = meta['calibration_supported_z_um']
            meta['calibration_supported_z_um'] = [-hi, -lo]
        lc = meta.get('lateral_correction')
        if lc:
            for k in ('coefficients_dx', 'coefficients_dy'):
                if k in lc:
                    lc[k] = [-v if i in ODD else v for i, v in enumerate(lc[k])]
        meta['z_convention'] = P.Z_CONVENTION
        meta['z_direction'] = P.Z_CONVENTION_TEXT
        return meta

    def npz(self, path):
        cal = dict(np.load(path, allow_pickle=True))
        if str(cal.get('z_convention', '')) == P.Z_CONVENTION:
            return
        cal = {k: (v.item() if isinstance(v, np.ndarray) and v.dtype == object and v.shape == () else v) for k, v in cal.items()}
        out = P.to_lab_z(cal)
        self.backup(path)
        if not self.dry:
            np.savez_compressed(path, **out)

    def summary(self, path):
        s = json.loads(path.read_text())
        if 'calibration_supported_z_um' in s:
            lo, hi = s['calibration_supported_z_um']
            s['calibration_supported_z_um'] = [-hi, -lo]
        lm = s.get('lateral_model')
        if lm:
            for k in ('dx', 'dy'):
                if k in lm:
                    lm[k] = [-v if i in ODD else v for i, v in enumerate(lm[k])]
        t = s.get('calibration_sample_tilt')
        if t:
            for k in ('um_per_mm_x', 'um_per_mm_y'):
                if k in t:
                    t[k] = -t[k]
        s['z_convention'], s['z_direction'] = P.Z_CONVENTION, P.Z_CONVENTION_TEXT
        self.backup(path)
        if not self.dry:
            path.write_text(json.dumps(s, indent=2))

    def validation(self, path):
        v = json.loads(path.read_text())

        def walk(o):
            if isinstance(o, dict):
                return {k: (-x if k == 'median_signed_error_um' and isinstance(x, (int, float)) else walk(x)) for k, x in o.items()}
            if isinstance(o, list):
                return [walk(x) for x in o]
            return o
        self.backup(path)
        if not self.dry:
            path.write_text(json.dumps(walk(v), indent=2))

    def export_mat(self, path):
        s = scipy.io.loadmat(path)
        if 'uz' in s:                         # deformation field export: already lab z
            return
        hit = [k for k in EXPORT_COLUMNS if k in s]
        if not hit:
            return
        keep = {k: v for k, v in s.items() if not k.startswith('__')}
        for k in hit:
            keep[k] = -keep[k]
        if 'export_info_json' in keep:
            info = json.loads(str(np.ravel(keep['export_info_json'])[0]))
            info['z_convention'] = P.Z_CONVENTION
            keep['export_info_json'] = json.dumps(info)
        self.backup(path)
        if not self.dry:
            scipy.io.savemat(path, keep, do_compression=True)


def convert(folder, dry=False, matlab=None):
    folder = Path(folder).resolve()
    if convention(folder) == P.Z_CONVENTION:
        print(f'{folder}: already z up, skipped')
        return []
    c = Converter(folder, dry)
    for path in sorted(folder.glob('*.csv')):
        for suffix, cols in CSV_COLUMNS.items():
            if path.name.endswith(suffix):
                c.csv_file(path, cols)
    for path in sorted(folder.glob('*_payload.mat')):
        c.payload_mat(path)
    if (folder/'calibration.npz').is_file():
        c.npz(folder/'calibration.npz')
    if (folder/'summary.json').is_file():
        c.summary(folder/'summary.json')
    if (folder/'validation.json').is_file():
        c.validation(folder/'validation.json')
    ex = folder/'exports'
    if ex.is_dir():
        for path in sorted(ex.glob('*.csv')):
            c.csv_file(path, EXPORT_COLUMNS, skip_if=('uz_um',))
        for path in sorted(ex.glob('*.mat')):
            c.export_mat(path)
    stamp = dict(z_convention=P.Z_CONVENTION, z_direction=P.Z_CONVENTION_TEXT,
                 converted=datetime.now().isoformat(timespec='seconds'), converted_by='convert_z_up.py', files=c.changed)
    if not dry:
        (folder/'z_convention.json').write_text(json.dumps(stamp, indent=2))
        ri = folder/'run_info.json'
        if ri.is_file():
            info = json.loads(ri.read_text())
            if 'z_convention' not in info:
                c.backup(ri)
            info['z_convention'], info['z_direction'] = P.Z_CONVENTION, P.Z_CONVENTION_TEXT
            ri.write_text(json.dumps(info, indent=2))
        tables = sorted(folder.glob('*_localizations.mat'))
        if tables and any(folder.glob('*_payload.mat')):
            if matlab and Path(matlab).exists():
                for t in tables:
                    c.backup(t)
                cmd = "addpath('"+str(P.ROOT).replace("'", "''")+"'); export_tables('"+str(folder).replace("'", "''")+"');"
                subprocess.run([matlab, '-batch', cmd], check=True, timeout=900)
            else:
                print(f'{folder}: MATLAB not found; run export_tables.m to regenerate {len(tables)} native table file(s)')
    print(f"{folder}: {'would change' if dry else 'converted'} {len(c.changed)} file(s)")
    return c.changed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('folders', nargs='+', type=Path)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--no-matlab', action='store_true')
    ap.add_argument('--matlab', default=None, help='matlab executable (default: found automatically)')
    a = ap.parse_args(argv)
    for f in a.folders:
        convert(f, a.dry_run, None if a.no_matlab else (a.matlab or P.find_matlab()))


if __name__ == '__main__':
    main()
