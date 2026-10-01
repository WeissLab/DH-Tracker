"""Re-run analyses with the current code and time every stage.

Runs analyze.py (the same entry point as the explorer's New analysis) for each dataset, polls its
status.json once a second to time each stage (calibration fitting, per movie: frame fitting,
field map, tracking + recovery, scoring; motion analysis; MATLAB export), then times an extra
motion analysis with ordered stages (track_analysis --stages). Writes runs/timing_<date>.json and a
Markdown table runs/timing_<date>.md.

Usage:  python tools/time_runs.py [--only NAME ...] [--no-stages]
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # the pipeline folder (this script is in tools/)
DATA = ROOT.parent
PY = sys.executable

DATASETS = [
    dict(name='10x_5ms_and_1ms', calibration=DATA/'10X/3. Inter-Lobe Calibration/10x_0.04s_25ums_5ms_1_MMStack_Pos0.ome.tif',
         movies=[DATA/'10X/4. Indentation/10x_20um-indent_100ums_5ms_1_cropped.tif',
                 DATA/'10X/4. Indentation/10x_20um-indent_100ums_1ms_2_cropped.tif'],
         note='the 1 ms movie uses the 5 ms calibration z-scan (same optics; the angle-z relation does not depend on exposure)'),
    dict(name='10x_10ms', calibration=DATA/'10X/3. Inter-Lobe Calibration/10x_0.04s_25ums_10ms_1_MMStack_Pos0.ome.tif',
         movies=[DATA/'10X/4. Indentation/10x_20um-indent_100ums_10ms_1_cropped.tif']),
    dict(name='10x_cells_collagen', calibration=DATA/'10X/3. Inter-Lobe Calibration/10xPALD_0.04s_25ums_1_MMStack_Pos0.ome.tif',
         movies=[DATA/'10X/4. Indentation/10x_10ms_20um-indent_100ums_cells1_1_crop.tif',
                 DATA/'10X/4. Indentation/10x_10ms_20um-indent_100ums_collagen1_1_crop.tif']),
    dict(name='20x_cells_collagen', calibration=DATA/'20X/3. Inter-Lobe Calibration/20ms_0.1%beads_80um_10ums_2_MMStack_Pos0.ome_corrected.tif',
         movies=[DATA/'20X/4. Indentation/20x_10ms_20um-indent_100ums_cells1_1_crop.tif',
                 DATA/'20X/4. Indentation/20x_10ms_20um-indent_100ums_collagen1_2_crop.tif'],
         note='the _corrected calibration file'),
]


def run_timed(cmd, status_file, log_file):
    """Run cmd, polling status_file every second; returns (seconds, [(t, stage, message)], returncode)."""
    t0 = time.time()
    events, last = [], None
    with open(log_file, 'w', encoding='utf-8') as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT))
        while proc.poll() is None:
            time.sleep(1)
            try:
                s = json.loads(Path(status_file).read_text())
            except (OSError, ValueError):
                continue
            key = (s.get('stage'), s.get('message'))
            if key != last:
                events.append((round(time.time()-t0, 1), *key))
                last = key
    return round(time.time()-t0, 1), events, proc.returncode


def durations(events, total):
    """Seconds spent in each (stage, message) step, in order."""
    out = []
    for i, (t, stage, msg) in enumerate(events):
        end = events[i+1][0] if i+1 < len(events) else total
        out.append(dict(stage=stage, step=msg, seconds=round(end-t, 1)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--no-stages', action='store_true')
    a = ap.parse_args()
    stamp = datetime.now().strftime('%Y%m%d-%H%M')
    report = dict(started=datetime.now().isoformat(timespec='seconds'), machine=dict(
        cpu=platform.processor(), logical_cpus=os.cpu_count(), python=sys.version.split()[0]), runs=[])
    out_json = ROOT/'runs'/f'timing_{stamp}.json'
    for ds in DATASETS:
        if a.only and ds['name'] not in a.only:
            continue
        out = ROOT/'runs'/f"{ds['name']}_{stamp}"
        out.mkdir(parents=True, exist_ok=True)
        status = out/'status.json'
        cmd = [PY, '-W', 'ignore', str(ROOT/'analyze.py'), str(ds['calibration']), *map(str, ds['movies']),
               '--output', str(out), '--status-file', str(status)]
        print(f"[{datetime.now():%H:%M:%S}] {ds['name']}: analysing ...", flush=True)
        total, events, rc = run_timed(cmd, status, out/'analyze.log')
        entry = dict(name=ds['name'], output=str(out), calibration=str(ds['calibration']), movies=[str(m) for m in ds['movies']],
                     note=ds.get('note', ''), returncode=rc, total_seconds=total, steps=durations(events, total))
        try:
            info = json.loads((out/'run_info.json').read_text())
            entry['calibration_shape'] = info['calibration']['shape']
            entry['movie_shapes'] = {k: v['shape'] for k, v in info['movies'].items()}
            entry['workers'] = None
        except (OSError, ValueError, KeyError):
            pass
        print(f"[{datetime.now():%H:%M:%S}] {ds['name']}: {'done' if rc == 0 else f'FAILED ({rc})'} in {total:.0f} s", flush=True)
        if rc == 0 and not a.no_stages:
            t = time.time()
            r = subprocess.run([PY, '-W', 'ignore', str(ROOT/'track_analysis.py'), str(out), '--stages', 'indentation'],
                               cwd=str(ROOT), capture_output=True, text=True)
            entry['stages_motion_seconds'] = round(time.time()-t, 1)
            entry['stages_motion_output'] = (r.stdout or r.stderr).strip().splitlines()[-3:]
            print(f"[{datetime.now():%H:%M:%S}] {ds['name']}: indentation-stages motion analysis {entry['stages_motion_seconds']:.0f} s", flush=True)
        report['runs'].append(entry)
        out_json.write_text(json.dumps(report, indent=2))
    report['finished'] = datetime.now().isoformat(timespec='seconds')
    out_json.write_text(json.dumps(report, indent=2))
    # Markdown summary
    lines = [f"# Analysis timing, {report['started']}", '', f"Machine: {report['machine']}", '']
    for e in report['runs']:
        lines += [f"## {e['name']} ({'ok' if e['returncode'] == 0 else 'failed'}) — {e['total_seconds']/60:.1f} min total", '',
                  f"Calibration {e.get('calibration_shape')} · movies {e.get('movie_shapes')}" + (f" · {e['note']}" if e['note'] else ''), '',
                  '| stage | step | seconds |', '|---|---|---|']
        lines += [f"| {s['stage']} | {s['step']} | {s['seconds']:.0f} |" for s in e['steps']]
        if 'stages_motion_seconds' in e:
            lines.append(f"| motion | indentation stages (track_analysis --stages) | {e['stages_motion_seconds']:.0f} |")
        lines.append('')
    out_json.with_suffix('.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'timing report: {out_json.with_suffix(".md")}', flush=True)


if __name__ == '__main__':
    main()
