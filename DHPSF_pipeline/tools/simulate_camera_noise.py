"""Simulation check: least squares vs sCMOS maximum likelihood, and error-bar calibration.

Synthetic DH beads (two Gaussian lobes) are rendered in photons, then passed through an
sCMOS camera model with per-pixel offset, read-noise variance (including a few noisy
pixels) and a gain in ADU/e-. Each bead is fitted with both noise models; the spread of
the estimates around the truth is compared with each other and with the fit's own
predicted precision (anglePrecisionDeg etc.).

Run from the project folder (the one holding DHPSF_pipeline):  python DHPSF_pipeline/tools/simulate_camera_noise.py
"""
import tempfile
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the pipeline folder (this script is in tools/)
import pipeline as P

Z = np.linspace(-29, 29, 5801)
MODEL = P.CalibrationModel(Z, np.linspace(5, 175, len(Z)), 16.5+(Z/12.)**2)


def camera(shape, rng, gain=0.8, read_sd_e=1.3, noisy_fraction=.02):
    offset = 100+rng.normal(0, 1.5, shape)
    variance = (read_sd_e*gain)**2*np.exp(rng.normal(0, .25, shape))
    noisy = rng.random(shape) < noisy_fraction
    variance[noisy] *= rng.uniform(5, 30, noisy.sum())      # sCMOS telegraph / hot pixels
    return offset, variance, gain


def render(beads, shape, cam, rng, background_e=3., sigma=2.1):
    y, x = np.indices(shape)
    photons = np.full(shape, background_e)
    for cx, cy, angle, amp_e in beads:
        sep = float(np.interp(angle, MODEL.dense_a, MODEL.dense_sep))
        d = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])*sep/2
        for s in (-1, 1):
            photons += amp_e*np.exp(-((x-cx-s*d[0])**2+(y-cy-s*d[1])**2)/(2*sigma**2))
    offset, variance, gain = cam
    return offset+gain*rng.poisson(photons)+rng.normal(0, np.sqrt(variance))


def main():
    rng = np.random.default_rng(3)
    shape = (256, 256)
    cam = camera(shape, rng)
    tmp = Path(tempfile.mkdtemp())/'camera.npz'
    np.savez(tmp, offset=cam[0], variance=cam[1], gain=np.float64(cam[2]))
    for label, amp_e in (('1 ms-like (lobe peak ~5 e-)', 5.), ('5 ms-like (lobe peak ~25 e-)', 25.),
                         ('10 ms-like (lobe peak ~45 e-)', 45.)):
        truth, est = [], {'lsq': [], 'poisson': []}
        pred = {'lsq': [], 'poisson': []}
        for trial in range(40):
            beads = [(rng.uniform(40, 216), rng.uniform(40, 216), rng.uniform(20, 160), amp_e)]
            image = render(beads, shape, cam, rng)
            for model in ('lsq', 'poisson'):
                cfg = P.Config(noise_model=model, camera_file=str(tmp) if model == 'poisson' else '', min_snr=1.5)
                rows, _ = P.localize_image(image, cfg, MODEL, 1)
                if not len(rows):
                    est[model].append((np.nan,)*3); pred[model].append((np.nan,)*3); continue
                cx, cy, angle, _ = beads[0]
                r = rows[np.argmin(np.hypot(rows[:, 2]-1-cx, rows[:, 5]-1-cy))]
                est[model].append((r[2]-1-cx, r[5]-1-cy, P.axial_difference(r[6], angle)))
                pred[model].append((r[P.C['xPrecisionPx']], r[P.C['yPrecisionPx']], r[P.C['anglePrecisionDeg']]))
        print(label)
        for model in ('lsq', 'poisson'):
            e, p = np.array(est[model]), np.array(pred[model])
            ok = np.isfinite(e).all(axis=1) & (np.abs(e[:, 0]) < 3)
            sd = e[ok].std(axis=0)
            print(f'  {model:8s} found {ok.mean():.2f}  actual SD: x {sd[0]:.3f} px, y {sd[1]:.3f} px, angle {sd[2]:.2f} deg'
                  f'  | predicted (median): x {np.median(p[ok, 0]):.3f}, y {np.median(p[ok, 1]):.3f}, angle {np.median(p[ok, 2]):.2f}'
                  f'  | bias angle {e[ok, 2].mean():+.2f} deg')


if __name__ == '__main__':
    main()
