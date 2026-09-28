"""Compare tracker choices on the same v2 localizations (main + rejected tables).

Run from the project folder (the one holding DHPSF_pipeline):  python DHPSF_pipeline/tools/compare_trackers.py [--results DIR]
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the pipeline folder (this script is in tools/)
from pipeline import ROOT, Config, C, NCOL, track


def load(csv):
    data = np.genfromtxt(csv, delimiter=',', names=True)
    rows = np.full((len(data), NCOL), np.nan)
    for name, i in C.items():
        rows[:, i] = data[name]
    return rows


def describe(rows, nframes):
    ids, counts = np.unique(rows[:, 9], return_counts=True)
    long = counts[counts >= 10]
    swaps = 0
    for t in ids:
        b = rows[rows[:, 9] == t]
        b = b[np.argsort(b[:, 8])]
        swaps += int(np.sum(np.linalg.norm(np.diff(b[:, [2, 5]], axis=0), axis=1) > 6))
    return dict(tracks=len(ids), full_length=int(np.sum(counts == nframes)), at_least_10=len(long),
                transient_lt3=int(np.sum(counts < 3)), median_length=float(np.median(counts)),
                whole_movie_completeness_long=float(long.sum()/(nframes*len(long))) if len(long) else None,
                steps_over_6px=swaps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', type=Path, required=True, help='a run folder, e.g. runs/<name>')
    args = ap.parse_args()
    report = {}
    info = args.results/'run_info.json'
    names = list(json.loads(info.read_text())['movies']) if info.exists() else ['cells', 'collagen']
    for movie in names:
        rows = np.vstack([load(args.results/f'{movie}_{kind}.csv') for kind in ('localizations', 'rejected')])
        nframes = int(rows[:, 8].max())
        report[movie] = {}
        for method in ('lap', 'kalman', 'nearest'):
            start = time.time()
            tracked = track(rows, Config(tracker=method))
            report[movie][method] = describe(tracked, nframes) | {'seconds': round(time.time()-start, 2)}
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
