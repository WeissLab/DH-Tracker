"""Audit v2 exports, estimate calibration reproducibility and real-data track completeness.

Run from the project folder (the one holding DHPSF_pipeline):  python DHPSF_pipeline/tools/validate_results.py [--results DIR]
"""
import argparse
from pathlib import Path
import json
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.io import loadmat
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the pipeline folder (this script is in tools/)
from pipeline import ROOT, Config, C, CalibrationModel, axial_difference


def calibration_holdout(cal):
    ids = cal['selected_track_ids']
    planes = np.asarray(cal['valid_planes'])+1   # calibrated (one-based) planes only
    if 'z_limits' in cal:                          # ... inside the reported z range
        zl = cal['z'][planes-1]
        planes = planes[(zl >= cal['z_limits'][0]) & (zl <= cal['z_limits'][1])]
    tracks = cal['tracks'][np.isin(cal['tracks'][:, 8], planes)]
    train = tracks[np.isin(tracks[:, 9], ids[::2])]
    test = tracks[np.isin(tracks[:, 9], ids[1::2])]
    angles, seps = [], []
    for frame in planes:
        rows = train[train[:, 8] == frame]
        ref = np.degrees(np.angle(np.mean(np.exp(2j*np.radians(rows[:, 6])))))/2
        angles.append(ref+np.median(axial_difference(rows[:, 6], ref)))
        seps.append(np.median(rows[:, C['lobeSeparationPixels']]))
    angles = np.degrees(np.unwrap(np.radians(np.array(angles)*2)))/2
    zp = cal['z'][planes-1]
    order = np.argsort(zp)          # lab z runs opposite to the plane order (see pipeline.to_lab_z)
    zp, angles, seps = zp[order], angles[order], np.asarray(seps)[order]
    dense_z = np.linspace(zp[0], zp[-1], (len(zp)-1)*100+1)
    model = CalibrationModel(dense_z, PchipInterpolator(zp, angles)(dense_z), PchipInterpolator(zp, seps)(dense_z))
    predicted, status = model.angle_to_z(test[:, 6], test[:, C['lobeSeparationPixels']])
    truth = cal['z'][test[:, 8].astype(int)-1]
    valid = np.isfinite(predicted)
    # a tilted calibration slide puts beads at different depths: compare with each bead's own depth
    tilt = np.zeros(len(test))
    if 'sample_tilt_um_per_px' in cal:
        (gx, gy), (xc, yc) = cal['sample_tilt_um_per_px'], cal['sample_tilt_centre_px']
        for t in np.unique(test[:, 9]):
            m = test[:, 9] == t
            tilt[m] = gx*(test[m, 2].mean()-xc)+gy*(test[m, 5].mean()-yc)
    def metrics(mask, reference=truth):
        error = predicted[mask]-reference[mask]
        return dict(n=int(mask.sum()), median_absolute_error_um=float(np.median(abs(error))),
                    p90_absolute_error_um=float(np.percentile(abs(error), 90)),
                    rmse_um=float(np.sqrt(np.mean(error**2))), median_signed_error_um=float(np.median(error)),
                    errors_over_10um=int(np.sum(abs(error) > 10)))
    return dict(training_beads=len(ids[::2]), test_beads=len(ids[1::2]), held_out_localizations=len(test),
                unavailable_inverse=int((~valid).sum()), resolved_by_separation=int(np.sum(status == 3)),
                all_valid=metrics(valid),
                interior_4um_from_ends=metrics(valid & (truth >= zp[0]+4) & (truth <= zp[-1]-4)),
                tilt_corrected=metrics(valid, truth+tilt),
                # the z origin is a convention: without the constant offset between the two halves
                tilt_corrected_offset_removed=metrics(valid, truth+tilt+np.median((predicted-truth-tilt)[valid])),
                caveat='Bead split after consistency selection on the full stack; measures reproducibility '
                       'among selected beads, not independent absolute accuracy.')


def audit_movie(path, cfg):
    s = loadmat(path)
    r = np.hstack((s['localizationMatrix'], s['fitQualityMatrix']))
    names = [str(n).strip() for n in np.ravel(s['qualityNames'])] if 'qualityNames' in s else []
    assert r.shape[1] == len(C), (r.shape, names)
    nframes = int(r[:, 8].max())
    assert np.isfinite(r[:, [0, 1, 2, 3, 4, 5, 6, 8, 9]]).all()
    np.testing.assert_allclose(r[:, 2], r[:, :2].mean(axis=1))
    np.testing.assert_allclose(r[:, 5], r[:, 3:5].mean(axis=1))
    np.testing.assert_allclose(axial_difference(r[:, 6], np.degrees(np.arctan2(r[:, 4]-r[:, 3], r[:, 1]-r[:, 0]))), 0, atol=1e-9)
    assert np.all((r[:, 6] >= 0) & (r[:, 6] < 180))
    lengths, jumps, gaps, coverage = [], [], [], []
    for tid in np.unique(r[:, 9]):
        bead = r[r[:, 9] == tid]
        bead = bead[np.argsort(bead[:, 8])]
        df = np.diff(bead[:, 8])
        assert np.all(df > 0), f'track {tid} has two localizations in one frame'
        distance = np.linalg.norm(np.diff(bead[:, [2, 5]], axis=0), axis=1)
        lengths.append(len(bead)); jumps.extend(distance.tolist()); gaps.extend(df[df > 1].tolist())
        if len(bead) >= 10:
            span = bead[-1, 8]-bead[0, 8]+1
            coverage.append((len(bead), span))
    lengths = np.array(lengths)
    coverage = np.array(coverage)
    per_frame = np.bincount(r[:, 8].astype(int), minlength=nframes+1)[1:]
    recovered = r[:, C['recovered']] == 1
    score = r[:, C['templateScore']]
    report = dict(rows=len(r), frames=nframes, per_frame_min_median_max=[int(per_frame.min()), float(np.median(per_frame)), int(per_frame.max())],
        tracks=len(lengths), full_length_tracks=int(np.sum(lengths == nframes)),
        tracks_at_least_10_frames=int(np.sum(lengths >= 10)),
        # Persistent beads: fraction of frames inside each track's span that carry a localization.
        within_span_completeness=float(coverage[:, 0].sum()/coverage[:, 1].sum()) if len(coverage) else None,
        # Stricter: a bead of >=10 frames is assumed present in every frame of the movie.
        whole_movie_completeness=float(coverage[:, 0].sum()/(nframes*len(coverage))) if len(coverage) else None,
        recovered_localizations=int(recovered.sum()), gap_closing_links=len(gaps),
        max_gap_frames=int(max(gaps)) if gaps else 0,
        median_xy_step_pixels=float(np.median(jumps)), max_xy_step_pixels=float(np.max(jumps)),
        z_status_counts={str(k): int(np.sum(r[:, C['zStatus']] == k)) for k in range(4)},
        shared_lobe_localizations=int(np.nansum(r[:, C['sharedLobe']])),
        template_score_median_detected=float(np.nanmedian(score[~recovered])),
        template_score_median_recovered=float(np.nanmedian(score[recovered])) if recovered.any() else None,
        template_score_below_0p5=int(np.sum(score < .5)),
        median_lobe_snr=float(np.median(r[:, C['minLobeSNR']])), structural_checks='passed')
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', type=Path, required=True, help='a run folder, e.g. runs/<name>')
    args = ap.parse_args()
    out = args.results
    with np.load(out/'calibration.npz') as saved:
        cal = {key: saved[key] for key in saved.files}
    report = dict(calibration_holdout=calibration_holdout(cal), movies={})
    for path in sorted(out.glob('*_payload.mat')):
        report['movies'][path.stem.replace('_payload', '')] = audit_movie(path, Config())
    (out/'validation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
