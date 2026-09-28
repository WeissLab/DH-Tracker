"""Characteristic lobe separation of a data set from spot pair distances (no calibration needed).

All spots (DoG maxima, on the smoothed seed image) are found on a few frames. Distances
between pairs of similarly bright spots are histogrammed and divided by 2*pi*r (a
pair-correlation / Ripley-type normalization), so random neighbours give a flat
background and the two lobes of each bead give a peak at the lobe separation. The same is
done per tile of the field to reveal field-dependent separation.

Use it on a new data set before choosing min/max separation. Run from the project folder (the one holding DHPSF_pipeline):
  python DHPSF_pipeline/tools/separation_profile.py MOVIE.tif [--frames 8] [--tiles 3] [--out plot.png]
"""
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the pipeline folder (this script is in tools/)
import pipeline as P


def subpixel(hp, xy):
    """Parabolic 3-point refinement of integer DoG maxima (x, y) to sub-pixel positions."""
    x, y = xy[:, 0].astype(int), xy[:, 1].astype(int)
    inside = (x > 0) & (y > 0) & (x < hp.shape[1]-1) & (y < hp.shape[0]-1)
    out = xy.astype(float).copy()
    xi, yi = x[inside], y[inside]
    c = hp[yi, xi]
    for axis, (m, p) in enumerate(((hp[yi, xi-1], hp[yi, xi+1]), (hp[yi-1, xi], hp[yi+1, xi]))):
        denominator = m-2*c+p
        shift = np.where(denominator < 0, .5*(m-p)/np.where(denominator < 0, denominator, -1), 0.)
        out[inside, axis] += np.clip(shift, -.5, .5)
    return out


def spot_pairs(path, cfg, frames, max_r=40., max_ratio=2.):
    """(distance, midpoint x, midpoint y) for spot pairs of similar brightness, one-based px."""
    out = []
    n = P.frame_count(path)
    for index in np.unique(np.linspace(0, n-1, frames).astype(int)):
        seed = P.seed_image(path, index, cfg)
        image = P.read_frame(path, index) if seed is None else seed
        xy, values, strong = P.find_peaks(image, cfg)
        keep = values >= strong
        xy, values = subpixel(P.dog(image, cfg), xy[keep]), values[keep]
        if len(xy) < 2:
            continue
        ab = cKDTree(xy).query_pairs(max_r, output_type='ndarray')
        a, b = ab.T
        similar = np.maximum(values[a], values[b]) <= max_ratio*np.minimum(values[a], values[b])
        a, b = a[similar], b[similar]
        d = np.linalg.norm(xy[a]-xy[b], axis=1)
        mid = (xy[a]+xy[b])/2+1
        out.append(np.column_stack((d, mid)))
    return np.vstack(out) if out else np.empty((0, 3))


def profile(d, bins):
    """Pair counts per annulus area, normalized so the far-distance background is 1."""
    counts, edges = np.histogram(d, bins)
    r = (edges[1:]+edges[:-1])/2
    g = counts/(2*np.pi*r*np.diff(edges))
    tail = g[r > .8*r.max()]
    return r, g/(tail.mean() if tail.size and tail.mean() > 0 else 1)


def peak(r, g, lo=6.):
    """Location, half-maximum width and height of the strongest peak above lo px."""
    smooth = gaussian_filter1d(g, 1.)
    use = r >= lo
    i = np.flatnonzero(use)[np.argmax(smooth[use])]
    half = 1+(smooth[i]-1)/2
    left, right = i, i
    while left > 0 and smooth[left] > half:
        left -= 1
    while right < len(r)-1 and smooth[right] > half:
        right += 1
    return float(r[i]), float(r[right]-r[left]), float(smooth[i])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('movie', type=Path)
    ap.add_argument('--frames', type=int, default=8)
    ap.add_argument('--tiles', type=int, default=3)
    ap.add_argument('--max-r', type=float, default=40.)
    ap.add_argument('--out', type=Path)
    args = ap.parse_args()
    cfg = P.Config()
    pairs = spot_pairs(args.movie, cfg, args.frames, args.max_r)
    bins = np.arange(2, args.max_r+.5, .5)
    r, g = profile(pairs[:, 0], bins)
    r0, width, height = peak(r, g)
    print(f'{args.movie.name}: {len(pairs)} similar-brightness spot pairs from {args.frames} frames')
    print(f'  characteristic lobe separation {r0:.1f} px (half-max width {width:.1f} px, peak {height:.1f}x background)')
    print(f'  suggested min/max_separation: {max(r0-max(2*width, 5), 4):.0f} .. {r0+max(2*width, 8):.0f} px')
    h, w = P.stack_reader(args.movie).shape[-2:]
    t = args.tiles
    grid = np.full((t, t), np.nan)
    for i in range(t):
        for j in range(t):
            m = ((pairs[:, 1] >= j*w/t) & (pairs[:, 1] < (j+1)*w/t) &
                 (pairs[:, 2] >= i*h/t) & (pairs[:, 2] < (i+1)*h/t))
            if m.sum() > 50:
                grid[i, j] = peak(*profile(pairs[m, 0], bins))[0]
    print('  per-tile separation (rows top->bottom, px):')
    for row in grid:
        print('   ', '  '.join('  -- ' if np.isnan(v) else f'{v:5.1f}' for v in row))
    if args.out:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        ax[0].plot(r, g, 'k')
        ax[0].axvline(r0, color='r', ls='--')
        ax[0].set(xlabel='spot pair distance (px)', ylabel='pairs per annulus area (background = 1)',
                  title=f'{args.movie.name}\nlobe separation {r0:.1f} px')
        im = ax[1].imshow(grid, cmap='viridis', extent=(0, w, h, 0))
        for i in range(t):
            for j in range(t):
                if np.isfinite(grid[i, j]):
                    ax[1].text((j+.5)*w/t, (i+.5)*h/t, f'{grid[i, j]:.1f}', ha='center', va='center', color='w')
        ax[1].set(title='separation peak per tile (px)', xlabel='x (px)', ylabel='y (px)')
        plt.colorbar(im, ax=ax[1], shrink=.8)
        fig.tight_layout(); fig.savefig(args.out, dpi=110); plt.close(fig)


if __name__ == '__main__':
    main()
