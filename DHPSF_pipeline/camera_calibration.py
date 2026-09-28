"""Build a camera noise file (camera.npz) for sCMOS maximum-likelihood fitting.

    python DHPSF_pipeline/camera_calibration.py --dark DARK.tif --electrons-per-adu 0.8 --out cameras/prime_fullwell.npz
    python DHPSF_pipeline/camera_calibration.py --dark DARK.tif --flat F1.tif --flat F2.tif --flat F3.tif --out ...

Dark frames (shutter closed, SAME camera mode as the movies: gain mode, bit depth, readout
rate, offset, and the exposure used) give each pixel's offset (mean) and read-noise
variance. The gain comes from either
  * --electrons-per-adu (the camera's test report / spec sheet, e.g. Photometrics lists
    "system gain" in e-/ADU per mode), or
  * flat-field stacks: an evenly lit field at 2-5 brightness levels, ~100 frames each
    (photon transfer: variance above read noise = gain x signal above offset).
Frames are streamed, so long stacks do not need to fit in memory. If the dark frames are a
crop of the sensor, pass --roi X,Y (top-left on the sensor) so movies can be matched.
Use the file with:  analyze.py ... --camera cameras/prime_fullwell.npz
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pipeline as P


def stream_stats(path, max_frames=None):
    """Per-pixel mean and variance over the frames of a stack (Welford, float64)."""
    n = P.frame_count(path)
    if max_frames:
        n = min(n, max_frames)
    mean = var_sum = None
    for i in range(n):
        f = P.read_frame(path, i).astype(np.float64)
        if mean is None:
            mean, var_sum = np.zeros_like(f), np.zeros_like(f)
        delta = f-mean
        mean += delta/(i+1)
        var_sum += delta*(f-mean)
    return mean, var_sum/max(n-1, 1), n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dark', type=Path, action='append', required=True, help='dark stack(s), same camera mode as the movies')
    ap.add_argument('--flat', type=Path, action='append', default=[], help='flat-field stacks at different brightnesses')
    ap.add_argument('--electrons-per-adu', type=float, help='camera conversion factor from its test report (e-/ADU)')
    ap.add_argument('--per-pixel-gain', action='store_true', help='keep a per-pixel gain map (needs >= 3 flat levels)')
    ap.add_argument('--roi', help='X,Y sensor position of the dark frames\' top-left pixel (default 0,0)')
    ap.add_argument('--max-frames', type=int, help='use at most this many frames per stack')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    if args.electrons_per_adu is None and not args.flat:
        ap.error('the gain is needed: pass --electrons-per-adu (camera test report) or --flat stacks')

    means, variances, frames = [], [], 0
    for d in args.dark:
        m, v, n = stream_stats(d, args.max_frames)
        means.append(m*n); variances.append(v*(n-1)); frames += n
        print(f'dark {d.name}: {n} frames')
    offset = sum(means)/frames
    variance = sum(variances)/max(frames-len(args.dark), 1)
    read_sd = np.sqrt(variance)

    if args.electrons_per_adu is not None:
        gain = np.float64(1./args.electrons_per_adu)         # ADU per photoelectron
        gain_note = f'from --electrons-per-adu {args.electrons_per_adu}'
    else:
        levels = [stream_stats(f, args.max_frames)[:2] for f in args.flat]
        signal = np.array([m-offset for m, _ in levels])     # (L, H, W) ADU above offset
        excess = np.array([v-variance for _, v in levels])   # shot-noise variance in ADU^2
        # global photon-transfer slope, robust: median over pixels of per-pixel slopes
        slope = (signal*excess).sum(axis=0)/np.maximum((signal**2).sum(axis=0), 1e-9)
        good = np.isfinite(slope) & (signal.min(axis=0) > 20)
        global_gain = float(np.median(slope[good]))
        if args.per_pixel_gain and len(levels) >= 3:
            gain = np.where(good, slope, global_gain)
            gain_note = f'per-pixel photon transfer from {len(levels)} flat levels (median {global_gain:.4f} ADU/e-)'
        else:
            gain = np.float64(global_gain)
            gain_note = f'global photon transfer from {len(levels)} flat level(s)'
        print(f'gain {global_gain:.4f} ADU/e- ({1/global_gain:.3f} e-/ADU)')

    g = float(np.median(gain))
    noisy = read_sd > 3*np.median(read_sd)
    info = dict(dark_frames=frames, shape=list(offset.shape), gain_adu_per_e=g, gain_note=gain_note,
                offset_median_adu=float(np.median(offset)), offset_sd_adu=float(np.std(offset)),
                read_noise_median_adu=float(np.median(read_sd)), read_noise_median_e=float(np.median(read_sd)/g),
                read_noise_rms_e=float(np.sqrt(np.mean(variance))/g), noisy_pixel_fraction=float(noisy.mean()),
                roi=[int(v) for v in args.roi.split(',')] if args.roi else [0, 0],
                sources=dict(dark=[str(p) for p in args.dark], flat=[str(p) for p in args.flat]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, offset=offset.astype(np.float32), variance=variance.astype(np.float32),
                        gain=gain if np.ndim(gain) == 0 else gain.astype(np.float32),
                        roi=np.array(info['roi']), info=json.dumps(info))
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
