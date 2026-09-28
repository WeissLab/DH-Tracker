"""Motion analysis of 3D bead tracks: per-track motion type and time-resolved motion states.

Two analyses, following the models of aTrack (Simon et al., "Detecting directed motion and
confinement in single-particle trajectories using hidden variables", eLife 13, RP99347 (2026),
doi:10.7554/eLife.99347) and ExaTrack (Simon, Wiggins & Weiss, bioRxiv (2026),
doi:10.64898/2026.01.22.700663), implemented here independently with Kalman filters (exact for
these linear-Gaussian models, equivalent to aTrack's analytical recurrences):

1. Per track (aTrack): Brownian, confined or directed motion. Each axis has a hidden true
   position r and, for anomalous motion, a hidden variable h:
     Brownian   r' = r + e                       e ~ N(0, d^2)
     confined   r' = (1-l)(r + e) + l*h ,  h' = h + u,   u ~ N(0, q^2)   (h: well centre)
     directed   r' = r + w + e ,           w' = w + u,   u ~ N(0, q^2)   (w: velocity)
   observed c = r + n with the localization error n of each point (its reported precision,
   so x, y and z keep their own errors; missing frames are simply predicted over). Each model
   is fitted by maximum likelihood per track; the likelihood ratio
   rho = L_Brownian / L_alternative < alpha (default 0.05) classifies a track as confined or
   directed, as in aTrack.

2. Over time (ExaTrack-style): a population model switching between two states per frame,
   'still' (Brownian with small steps) and 'moving' (directed, with a persistent velocity),
   with first-order (Markov) transitions. Parameters are shared by all tracks and fitted by
   maximum likelihood; the filter keeps one Gaussian per state and merges (moment matching)
   after each step (an interacting-multiple-model filter; ExaTrack merges over a longer
   recent history). Every localization gets its probability of being in the moving state.

3. Refined positions: each track's model, run forward (Kalman filter) and backward (RTS
   smoother), estimates the true position at every frame from the whole track. Written as
   shifts from the analysed coordinates, with their standard deviations.

Outputs (next to the results): {movie}_track_motion.csv (per track) and
{movie}_motion_states.csv (per localization: frame, track, p_moving, velocity, refined-position
shift and standard deviation).

Usage:  python track_analysis.py RESULTS_DIR [--movie NAME] [--frame-interval MS]
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
LOG2PI = math.log(2*math.pi)


# ------------------------------------------------------------------ data
def load_tracks(csv, pixel_size_um, coords='stabilized', min_points=10):
    """Tracks as padded arrays: obs (B, T, 3) µm, var (B, T, 3) µm^2, mask (B, T), plus bookkeeping.

    coords: 'stabilized' (whole-field shake removed; default), 'corrected' or 'raw'. Each track
    starts at t = 0 (its first frame); frames it misses are masked, not dropped.
    """
    d = np.genfromtxt(csv, delimiter=',', names=True)
    xcol, ycol, zcol = {'stabilized': ('xStabilized', 'yStabilized', 'zStabilized'),
                        'corrected': ('xCorrected', 'yCorrected', 'zMicrons'),
                        'raw': ('xMean', 'yMean', 'zMicrons')}[coords]
    ids, tracks = [], []
    for t in np.unique(d['track_number']):
        r = d[d['track_number'] == t]
        r = r[np.argsort(r['frame_number'])]
        ok = np.isfinite(r[xcol]) & np.isfinite(r[ycol]) & np.isfinite(r[zcol])
        r = r[ok]
        if len(r) < min_points:
            continue
        ids.append(int(t))
        tracks.append(r)
    if not tracks:
        raise ValueError(f'no tracks with at least {min_points} valid localizations in {csv}')
    B = len(tracks)
    T = int(max(r['frame_number'][-1]-r['frame_number'][0] for r in tracks))+1
    obs = np.zeros((B, T, 3)); var = np.ones((B, T, 3)); mask = np.zeros((B, T), bool)
    frame0 = np.zeros(B, int)
    for b, r in enumerate(tracks):
        f = (r['frame_number']-r['frame_number'][0]).astype(int)
        frame0[b] = int(r['frame_number'][0])
        obs[b, f] = np.c_[r[xcol]*pixel_size_um, r[ycol]*pixel_size_um, r[zcol]]
        sx, sy, sz = (r['xPrecisionPx']*pixel_size_um, r['yPrecisionPx']*pixel_size_um, r['zPrecisionUm'])
        prec = np.c_[sx, sy, sz]
        for a in range(3):   # missing precision: the track's median
            bad = ~np.isfinite(prec[:, a]) | (prec[:, a] <= 0)
            prec[bad, a] = np.nanmedian(prec[~bad, a]) if (~bad).any() else .1
        var[b, f] = prec**2
        mask[b, f] = True
    return dict(ids=np.asarray(ids), obs=obs, var=var, mask=mask, frame0=frame0, T=T)


def estimate_error_scale(data, min_pairs=20):
    """Per-axis factor on the reported localization errors, from the data.

    For a random walk observed with white localization noise of variance s^2, consecutive steps
    are anticorrelated: cov(step_t, step_t+1) = -s^2, whatever the step size or a slow drift.
    The median over tracks of -cov / mean(reported variance) gives the factor squared. Without
    it, errors that are reported too small look like 'confinement' (white scatter around a point).
    """
    out = []
    for a in range(3):
        est = []
        for b in range(len(data['ids'])):
            m = data['mask'][b]
            ok = m[1:] & m[:-1]
            inc = np.diff(data['obs'][b, :, a])
            both = ok[1:] & ok[:-1]
            if both.sum() < min_pairs:
                continue
            c = np.median(inc[ok])
            est.append(-np.mean((inc[:-1][both]-c)*(inc[1:][both]-c))/np.mean(data['var'][b, m, a]))
        out.append(math.sqrt(min(max(np.median(est), .25), 16.)) if est else 1.)
    return np.asarray(out)


# ------------------------------------------------------------------ single-model Kalman filter
def _kalman(model, obs, var, mask, d, q=None, v=None, l=None, H0=5., smooth=False, positions=False):
    """Log-likelihood per track (B,) of one motion model; tensors obs/var (B,T,A), mask (B,T), params (B,).

    Hidden state per axis (r, h). With smooth=True also returns the RTS-smoothed hidden
    variable h (velocity or well centre), (B, T, A); with positions=True as well the smoothed
    true position r and its variance, (B, T, A) each (forward filter + backward pass: every
    position is estimated from the whole track).
    """
    d2 = (d*d)[:, None]
    one, zero = torch.ones_like(d)[:, None], torch.zeros_like(d)[:, None]
    if model == 'brownian':
        a, b, q00, q11, conf, p11_0 = one, zero, d2, zero, zero, zero
    elif model == 'directed':
        a, b, q00, q11, conf, p11_0 = one, one, d2, (q*q)[:, None], zero, (v*v)[:, None]
    elif model == 'confined':
        a, b, q00, q11, conf, p11_0 = (1-l)[:, None], l[:, None], (1-l)[:, None]**2*d2, (q*q)[:, None], one, H0**2*one
    else:
        raise ValueError(model)
    return _kalman_lin(obs, var, mask, a, b, q00, q11, conf, p11_0, smooth, positions)


BIG_VAR = 1e12      # variance given to missing frames: the gain is ~0, so they are simply predicted over


def _kalman_lin(obs, var, mask, a, b, q00, q11, conf, p11_0, smooth=False, positions=False):
    """The linear model shared by all three motion models, with per-track parameters (B, 1):

        r' = a r + b h + e (q00),   h' = h + u (q11),   observed c = r + n (var)

    Brownian: a = 1, b = 0, q11 = 0; directed: a = b = 1 (h = velocity); confined: a = 1 - l, b = l
    (h = well centre). conf (B, 1) = 1 where h starts at the first position (a well centre), else 0;
    p11_0 (B, 1) its initial variance. Because every track has its own parameters, several models
    and starting points can run as one batch: the recursion is dominated by per-operation overhead,
    so one pass over 5B tracks costs about as much as one over B.
    """
    B, T, A = obs.shape
    y0 = obs[:, 0]
    m0, m1 = y0.clone(), y0*conf
    p00 = var[:, 0].clone()
    p01 = torch.zeros_like(y0)
    p11 = p11_0.expand(B, A).clone()
    aa, ab2, bb = a*a, 2*a*b, b*b
    # missing frames: a huge measurement variance makes the update a no-op (gain ~1e-12)
    maskf = mask.to(obs.dtype)
    veff = torch.where(mask[..., None], var, torch.full_like(var, BIG_VAR))
    ll = torch.zeros(B, A, dtype=obs.dtype)
    keep = [] if smooth else None
    if smooth:
        keep.append((m0, m1, p00, p01, p11, m0, m1, p00, p01, p11))
    for t in range(1, T):
        # predict
        pm0, pm1 = a*m0+b*m1, m1
        pp00 = aa*p00+ab2*p01+bb*p11+q00
        pp01 = a*p01+b*p11
        pp11 = p11+q11
        # update
        S = pp00+veff[:, t]
        e = obs[:, t]-pm0
        k0, k1 = pp00/S, pp01/S
        m0 = pm0+k0*e
        m1 = pm1+k1*e
        p00 = pp00*(1-k0)
        p01 = pp01*(1-k0)
        p11 = pp11-k1*pp01
        ll = ll+maskf[:, t, None]*(torch.log(S)+e*e/S)
        if smooth:
            keep.append((m0, m1, p00, p01, p11, pm0, pm1, pp00, pp01, pp11))
    ll = -.5*(ll.sum(-1)+LOG2PI*A*(maskf[:, 1:].sum(1)))
    if not smooth:
        return ll
    # Rauch-Tung-Striebel smoother (2x2 per axis)
    # tracks without a hidden variable (Brownian: its variance stays 0) use the 1x1 case,
    # since P_pred is singular in h
    nohid = ((b == 0) & (q11 == 0) & (p11_0 == 0)).expand(B, A)
    sm0, sm1, s00, s01, s11 = keep[-1][:5]
    out, pos, pvar = [sm1], [sm0], [s00]
    for t in range(T-2, -1, -1):
        f0, f1, f00, f01, f11 = keep[t][:5]
        n0, n1, n00, n01, n11 = keep[t+1][5:]
        # C = P_f F^T P_pred^-1 with F = [[a, b], [0, 1]]
        det = torch.where(nohid, torch.ones_like(n00), n00*n11-n01*n01)
        i00 = torch.where(nohid, 1/n00, n11/det)
        i01 = torch.where(nohid, torch.zeros_like(n00), -n01/det)
        i11 = torch.where(nohid, torch.zeros_like(n00), n00/det)
        g00, g01 = f00*a+f01*b, f01           # (P_f F^T) row 0
        g10, g11 = f01*a+f11*b, f11           # (P_f F^T) row 1
        c10, c11 = g10*i00+g11*i01, g10*i01+g11*i11
        c00, c01 = g00*i00+g01*i01, g00*i01+g01*i11
        dm0, dm1 = sm0-n0, sm1-n1
        sm0, sm1 = f0+c00*dm0+c01*dm1, f1+c10*dm0+c11*dm1
        if positions:                         # P_s = P_f + C (P_s' - P_pred) C^T, (0, 0) element
            d00, d01, d11 = s00-n00, s01-n01, s11-n11
            s00, s01, s11 = (f00+c00*(c00*d00+c01*d01)+c01*(c00*d01+c01*d11),
                             f01+c00*(c10*d00+c11*d01)+c01*(c10*d01+c11*d11),
                             f11+c10*(c10*d00+c11*d01)+c11*(c10*d01+c11*d11))
            pos.append(sm0); pvar.append(s00)
        out.append(sm1)
    h = torch.stack(out[::-1], dim=1)
    if positions:
        return ll, h, torch.stack(pos[::-1], dim=1), torch.stack(pvar[::-1], dim=1).clamp_min(0)
    return ll, h


def _optimize(loss_fn, params, steps, lr, method='lbfgs', polish=100, tol_change=1e-10):
    """Minimize loss_fn() over params: L-BFGS with a strong-Wolfe line search (default; converges in
    far fewer evaluations than Adam on these smooth likelihoods), or Adam for `steps` steps."""
    if method == 'adam':
        opt = torch.optim.Adam(params, lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            loss = loss_fn()
            loss.backward()
            opt.step()
        return
    if method == 'hybrid':      # L-BFGS finds the optimum's basin; Adam (a step size per parameter)
        _optimize(loss_fn, params, steps, lr, 'lbfgs')          # then finishes the tracks the shared
        _optimize(loss_fn, params, polish, lr*.4, 'adam')        # line search left behind
        return
    opt = torch.optim.LBFGS(params, lr=1, max_iter=steps, history_size=20, line_search_fn='strong_wolfe',
                            tolerance_grad=1e-7, tolerance_change=tol_change)
    def closure():
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        return loss
    opt.step(closure)


# starting points (model, l, q, v) of the per-track fits; d starts from the data. The likelihoods
# have several local optima, and extra starts cost little because they share one batched pass.
# (for the gradient methods; the default grid method does not use them)
SINGLE_MODEL_STARTS = ([('brownian', .1, .005, .02)]
                       + [('confined', l, q, .02) for l in (.03, .15, .5) for q in (.002, .03)]
                       + [('directed', .1, q, v) for q in (.001, .01, .06) for v in (.005, .05)])


def fit_single_models(data, steps=150, lr=.05, kappa=1., method='grid', starts=None):
    """Fit Brownian, confined and directed models to every track; returns a dict of per-track numpy arrays.

    method 'grid' (default): a coarse grid over each model's parameters for all tracks at once
    (forward passes only), then a damped Newton polish per track from three diverse grid points
    (_grid_fit). On the 10x 5 ms movie this reached the best likelihood found by any method for every
    track (within 0.1) in 3.5 s on 12 cores / 6.8 s on one, against 65 s for 2-start Adam, which
    missed the best confined optimum for ~40% of tracks.
    Gradient alternatives, all models and starts in one batch (see _kalman_lin): 'hybrid'
    (L-BFGS then Adam), 'lbfgs' (`steps` = max iterations) or 'adam' (`steps` steps at rate lr)."""
    obs = torch.as_tensor(data['obs']); var = torch.as_tensor(data['var'])*torch.as_tensor(np.broadcast_to(np.asarray(kappa, float), (3,)))**2
    mask = torch.as_tensor(data['mask'])
    B = obs.shape[0]
    # starting step size from the robust frame-to-frame scatter, minus the localization error
    diffs = np.diff(data['obs'], axis=1)[:, :, :2]
    ok = data['mask'][:, 1:] & data['mask'][:, :-1]
    step = np.array([np.median(np.abs(diffs[b][ok[b]])) if ok[b].any() else .1 for b in range(B)])*1.4826
    noise = np.sqrt(np.median(data['var'][:, :, :2], axis=(1, 2)))
    d0 = np.sqrt(np.clip(step**2/2-noise**2, 1e-4, None))
    if method == 'grid':
        llB, pB = _grid_fit('brownian', obs, var, mask, np.log(d0))
        log_e0 = torch.log(pB['e']).numpy() if FIT_ERROR else None       # the other models start at the Brownian e
        llC, pC = _grid_fit('confined', obs, var, mask, np.log(d0), log_e0)
        llD, pD = _grid_fit('directed', obs, var, mask, np.log(d0), log_e0)
        return _single_model_output(data, obs, var, mask, llB, pB, llC, pC, llD, pD)
    # every (model, starting point) variant of every track in one batch, with its own parameters
    variants = starts or SINGLE_MODEL_STARTS
    V = len(variants)
    code = lambda m: torch.tensor(np.repeat([float(v[0] == m) for v in variants], B))
    isC, isD = code('confined'), code('directed')
    rep = lambda x: np.tile(x, V)
    theta = torch.tensor(np.stack([np.log(rep(d0)),
                                   np.repeat([math.log(v[1]/(1-v[1])) for v in variants], B),
                                   np.repeat([math.log(v[2]) for v in variants], B),
                                   np.repeat([math.log(v[3]) for v in variants], B)]), requires_grad=True)
    obsV, varV, maskV = obs.repeat(V, 1, 1), var.repeat(V, 1, 1), mask.repeat(V, 1)
    H0 = 5.

    def unpack(th):
        d, l, q, v = torch.exp(th[0]), torch.sigmoid(th[1])*.999, torch.exp(th[2]), torch.exp(th[3])
        return d, l, q, v

    def batch_ll(th):
        d, l, q, v = unpack(th)
        lc = isC*l
        a = (1-lc)[:, None]
        return _kalman_lin(obsV, varV, maskV, a, (lc+isD)[:, None], a*a*(d*d)[:, None], ((isC+isD)*q*q)[:, None],
                           isC[:, None], (isC*H0**2+isD*v*v)[:, None])

    _optimize(lambda: -batch_ll(theta).sum(), [theta], steps, lr, method)
    with torch.no_grad():
        ll = batch_ll(theta).reshape(V, B)
        d, l, q, v = (x.reshape(V, B) for x in unpack(theta))

    def best(rows, keys):
        k = torch.tensor(rows)[ll[rows].argmax(0)]                      # best starting point per track
        pick = lambda x: x[k, torch.arange(B)]
        return pick(ll), {name: pick(x) for name, x in zip('dlqv', (d, l, q, v)) if name in keys}

    rows = lambda m: [i for i, s in enumerate(variants) if s[0] == m]
    llB, pB = best(rows('brownian'), 'd')
    llC, pC = best(rows('confined'), 'dlq')
    llD, pD = best(rows('directed'), 'dqv')
    return _single_model_output(data, obs, var, mask, llB, pB, llC, pC, llD, pD)


# ------------------------------------------------------------------ per-track fits by grid search
# Parameters in unconstrained form. Brownian: log d. Directed: log d, log q, log v. Confined:
# log s, logit l, logit(q/d), with s = (1-l) d / sqrt(1-(1-l)^2) the stationary SD of the bead around
# its well centre (its cage size): the likelihood has long ridges in (d, l) -- a tight cage with fast
# relaxation (l -> 1, d large) looks like extra white noise -- which become near-axis-aligned in (s, l).
# The well centre may diffuse at most as fast as the bead (q < d). Without that bound, the limit
# l -> 0, q -> infinity (l q fixed) turns the confined model into the directed one (l (h - r) acts as
# a slowly changing velocity), and a fully optimized confined model then absorbs all directed tracks.
# The last parameter of every model is log e, a per-track factor on the reported localization errors
# (on top of the per-axis scale from the data): extra frame-to-frame noise of one track is then
# explained as localization error in every model, instead of favouring whichever model can mimic
# white noise (a tight cage). Coarse grids (first parameter relative to the track's own d0, e for the
# confined and directed models relative to the track's Brownian-fit e, the others absolute) are wide
# enough to reach the limits above.
GRID = {'brownian': [np.linspace(-5., 3., 9), np.log([.6, .8, 1., 1.3, 1.8, 2.6])],
        'confined': [np.linspace(-6., 3., 10), np.linspace(-14., 7., 11), np.linspace(-14., 6., 9), np.log([.8, 1., 1.25])],
        'directed': [np.linspace(-12., 2., 8), np.linspace(math.log(1e-6), math.log(.3), 8), np.linspace(math.log(1e-9), math.log(1.), 7),
                     np.log([.8, 1., 1.25])]}
FIT_ERROR = True        # fit the per-track error factor e (False: e = 1, the reported errors as scaled)


def _grid_axes(model):
    return GRID[model] if FIT_ERROR else GRID[model][:-1]


def _lin_params(model, th, H0=5.):
    """Unconstrained parameters th (P, N) -> the linear-model parameters of _kalman_lin, each (N, 1),
    the natural parameters, and the per-row factor on the measurement variances, e^2 (N, 1, 1)."""
    e2 = torch.exp(2*th[-1])[:, None, None] if FIT_ERROR else torch.ones(th.shape[1], 1, 1, dtype=th.dtype)
    if model == 'confined':
        s, l = torch.exp(th[0]), torch.sigmoid(th[1])*.999
        d = s*torch.sqrt(1-(1-l)**2)/(1-l)
        q = d*torch.sigmoid(th[2])                  # the well moves no faster than the bead diffuses (q < d)
    else:
        d = torch.exp(th[0])
    one, zero = torch.ones_like(d), torch.zeros_like(d)
    if model == 'brownian':
        lin, nat = (one, zero, d*d, zero, zero, zero), dict(d=d)
    elif model == 'confined':
        lin, nat = ((1-l), l, (1-l)**2*d*d, q*q, one, H0**2*one), dict(d=d, l=l, q=q)
    else:
        q, v = torch.exp(th[1]), torch.exp(th[2])
        lin, nat = (one, one, d*d, q*q, zero, v*v), dict(d=d, q=q, v=v)
    nat['e'] = torch.sqrt(e2[:, 0, 0])
    return [x[:, None] for x in lin], nat, e2


class _threads:
    """Temporarily set torch's CPU threads. The filter's tensors are small: for a few thousand rows,
    one thread is fastest (multithreading overhead exceeds the work, ~3x slower at 12 threads);
    only large batches (the coarse grids) gain from all cores."""
    def __init__(self, n):
        self.n = max(1, int(n))

    def __enter__(self):
        self.old = torch.get_num_threads()
        torch.set_num_threads(self.n)

    def __exit__(self, *exc):
        torch.set_num_threads(self.old)


BIG_BATCH_THREADS = torch.get_num_threads()     # torch's default: the physical cores


def _evaluator(model, obs, var, mask, max_rows=250_000, dtype=None):
    """ll for many parameter sets per track, forward only: theta (P, K, B) -> (K, B), in chunks.
    dtype=torch.float32 halves the cost of the coarse grids (used only to pick basins)."""
    B = obs.shape[0]
    dt = dtype or obs.dtype
    o, v, m = obs.to(dt), var.to(dt), mask

    def evaluate(theta):
        P, K = theta.shape[:2]
        out = torch.empty(K, B, dtype=obs.dtype)
        per = max(1, max_rows//B)
        with torch.no_grad(), _threads(BIG_BATCH_THREADS if K*B > 20_000 else 1):
            for k0 in range(0, K, per):
                th = theta[:, k0:k0+per].reshape(P, -1).to(dt)
                kk = th.shape[1]//B
                lin, _, e2 = _lin_params(model, th)
                out[k0:k0+kk] = _kalman_lin(o.repeat(kk, 1, 1), v.repeat(kk, 1, 1)*e2, m.repeat(kk, 1), *lin).reshape(kk, B).to(obs.dtype)
        return out
    return evaluate


def _grid_fit(model, obs, var, mask, log_d0, log_e0=None, iters=15, n_starts=3, grid_dtype=torch.float32):
    """Maximum-likelihood fit of one model to every track: a coarse grid, evaluated for all tracks and
    grid points together (forward only, no gradients), finds the best basins; a damped Newton polish
    (_newton) from each track's `n_starts` best grid points converges, and the best result is kept.
    log_e0: each track's starting log error factor (the e axis is then relative to it)."""
    B = obs.shape[0]
    axes = _grid_axes(model)
    P = len(axes)
    base = torch.zeros(P, B, dtype=obs.dtype)
    base[0] = torch.as_tensor(log_d0)                              # first parameter relative to the track's d0
    if FIT_ERROR and log_e0 is not None:
        base[-1] = torch.as_tensor(log_e0)
    combos = torch.as_tensor(np.stack(np.meshgrid(*axes, indexing='ij'), 0).reshape(P, -1))   # (P, K)
    theta = base[:, None, :]+combos[:, :, None]                    # (P, K, B)
    llg = _evaluator(model, obs, var, mask, dtype=grid_dtype)(theta)
    # diverse starts: the best grid point within each of `n_starts` slices of the model's own
    # parameter (l for confined, q for directed; e for Brownian), so different basins (e.g.
    # Brownian-like vs a wandering velocity) all get polished
    if P == 1:
        top = llg.argmax(0, keepdim=True)
    else:
        key = combos[1]
        edges = torch.quantile(torch.unique(key), torch.linspace(0, 1, n_starts+1, dtype=key.dtype))
        top = []
        for j in range(n_starts):
            inside = (key >= edges[j]) & ((key < edges[j+1]) if j < n_starts-1 else (key <= edges[j+1]))
            masked = torch.where(inside[:, None], llg, torch.full_like(llg, -torch.inf))
            top.append(masked.argmax(0))
        top = torch.stack(top)                                     # (n, B)
    n = top.shape[0]
    idx = torch.arange(B)
    th0 = torch.stack([theta[:, top[j], idx] for j in range(n)], 1).reshape(P, n*B)   # start j of track b at j*B + b
    obsN, varN, maskN = obs.repeat(n, 1, 1), var.repeat(n, 1, 1), mask.repeat(n, 1)
    evaluate = _evaluator(model, obsN, varN, maskN)
    ll0 = evaluate(th0[:, None, :])[0]                             # the starts' ll in full precision
    with _threads(1):
        th, ll = _newton(model, obsN, varN, maskN, th0, ll0, iters)
    ll = ll.reshape(n, B)
    k = ll.argmax(0)
    best = th.reshape(P, n, B)[:, k, idx]
    _, nat, _ = _lin_params(model, best)
    return ll[k, idx], nat


SHORT_STEPS = (.4, .12, .03)       # tried (at once) only where the full Newton step did not improve


def _newton(model, obs, var, mask, theta, ll, iters, evaluate=None, h=1e-4, max_step=3., tol=1e-6):
    """Damped Newton ascent of each track's log-likelihood, all tracks at once.

    Gradients come from one autograd pass; the Hessian from finite differences of gradients, with
    the P shifted copies in the same batch. The Hessian is made negative definite (eigenvalues
    clamped) and the step capped at `max_step` in the unconstrained parameters. The full step is
    tried first; tracks where it does not improve try shorter steps (all at once) and otherwise stay,
    so the likelihood never decreases and every track has its own step size. Tracks that have
    converged (gain < tol) leave the batch, so the remaining iterations only cost what is left.
    """
    P, B = theta.shape
    eye = torch.eye(P, dtype=theta.dtype)
    theta, ll = theta.clone(), ll.clone()
    active = torch.arange(B)

    def ll_of(th, rows):                                       # th (P, n*len(rows)) for the tracks `rows`
        n = th.shape[1]//len(rows)
        lin, _, e2 = _lin_params(model, th)
        return _kalman_lin(obs[rows].repeat(n, 1, 1), var[rows].repeat(n, 1, 1)*e2, mask[rows].repeat(n, 1), *lin)

    for _ in range(iters):
        if not len(active):
            break
        A = len(active)
        th_a = theta[:, active]
        shifted = torch.cat([th_a]+[th_a+h*eye[j][:, None] for j in range(P)], 1).detach().requires_grad_(True)
        G, = torch.autograd.grad(ll_of(shifted, active).sum(), shifted)
        G = G.reshape(P, 1+P, A)
        g = G[:, 0]                                                                  # (P, A)
        H = ((G[:, 1:]-g[:, None])/h).permute(2, 0, 1)                               # (A, P, P): H[b, i, j] = d g_i / d th_j
        H = .5*(H+H.transpose(1, 2))
        e, U = torch.linalg.eigh(-H)                                                 # want -H positive definite
        e = torch.clamp(e, min=1e-6*e.abs().amax(1, keepdim=True).clamp_min(1e-12)+1e-9)
        step = (U @ ((U.transpose(1, 2) @ g.T[:, :, None]).squeeze(-1)/e)[:, :, None]).squeeze(-1)   # (A, P)
        norm = step.norm(dim=1, keepdim=True)
        step = (step*torch.clamp(max_step/norm.clamp_min(1e-12), max=1.)).T                        # (P, A)
        with torch.no_grad():
            new = th_a+step
            ll_new = ll_of(new, active)
            worse = ll_new <= ll[active]
            if worse.any():                                                          # shorter steps for those
                w = torch.nonzero(worse).squeeze(1)
                alphas = torch.as_tensor(SHORT_STEPS, dtype=theta.dtype)
                cand = (th_a[:, w][:, None, :]+alphas[None, :, None]*step[:, w][:, None, :]).reshape(P, -1)
                llc = ll_of(cand, active[w]).reshape(len(SHORT_STEPS), len(w))
                k = llc.argmax(0)
                j = torch.arange(len(w))
                new[:, w] = cand.reshape(P, len(SHORT_STEPS), len(w))[:, k, j]
                ll_new[w] = llc[k, j]
            better = ll_new > ll[active]
            gain = torch.where(better, ll_new-ll[active], torch.zeros_like(ll_new))
            upd = active[better]
            theta[:, upd] = new[:, better]
            ll[upd] = ll_new[better]
            active = active[gain > tol]
    return theta, ll


def _single_model_output(data, obs, var, mask, llB, pB, llC, pC, llD, pD):
    B = obs.shape[0]
    out = {}
    # the alternatives contain the Brownian model (l, q, v -> 0): never below it
    llC, llD = torch.maximum(llC, llB), torch.maximum(llD, llB)
    ones = torch.ones(B, dtype=obs.dtype)
    eB, eC, eD = (p.get('e', ones) for p in (pB, pC, pD))          # per-track error factors (1 if not fitted)
    with torch.no_grad():
        _, vel = _kalman('directed', obs, var*(eD**2)[:, None, None], mask, smooth=True, **{k: pD[k] for k in 'dqv'})
    speed = torch.sqrt((vel**2).sum(-1)).numpy()          # µm per frame
    m = data['mask']
    # each track's direction of motion: the principal axis of its smoothed velocity (sign arbitrary;
    # motion there and back along one line shares it)
    V = vel.numpy()*m[..., None]
    u = np.stack([np.linalg.eigh(V[b].T @ V[b])[1][:, -1] for b in range(B)])
    out['u_directed'] = u
    out.update(ll_brownian=llB.numpy(), ll_confined=llC.numpy(), ll_directed=llD.numpy(),
               d_brownian=pB['d'].numpy(), d_confined=pC['d'].numpy(), l_confined=pC['l'].numpy(), q_confined=pC['q'].numpy(),
               d_directed=pD['d'].numpy(), q_directed=pD['q'].numpy(), v_directed=pD['v'].numpy(),
               e_brownian=eB.numpy(), e_confined=eC.numpy(), e_directed=eD.numpy(),
               mean_speed=np.array([speed[b][m[b]].mean() for b in range(B)]),
               max_speed=np.array([speed[b][m[b]].max() for b in range(B)]))
    return out


def classify(fit, alpha=.05):
    """aTrack's likelihood-ratio test: rho = L_Brownian / L_alternative < alpha rejects Brownian motion."""
    log_rho_c = fit['ll_brownian']-fit['ll_confined']
    log_rho_d = fit['ll_brownian']-fit['ll_directed']
    la = math.log(alpha)
    cls = np.where((log_rho_d < la) & (fit['ll_directed'] >= fit['ll_confined']), 'directed',
                   np.where(log_rho_c < la, 'confined', np.where(log_rho_d < la, 'directed', 'brownian')))
    return cls, np.exp(log_rho_c), np.exp(log_rho_d)


def refine_positions(data, fit, cls, kappa=1.):
    """Smoothed true positions (B, T, 3) and their standard deviations, from each track's own model.

    Each track uses the motion model it was classified as, with its fitted parameters; the
    Kalman filter runs forward and an RTS pass backward, so every position draws on the whole
    track, weighted by each localization's error. Only as good as the model: a real jump that
    the model does not expect is spread over a few frames.
    """
    obs = torch.as_tensor(data['obs']); mask = torch.as_tensor(data['mask'])
    var = torch.as_tensor(data['var'])*torch.as_tensor(np.broadcast_to(np.asarray(kappa, float), (3,)))**2
    g = lambda k: torch.as_tensor(np.asarray(fit[k], float))
    args = {'brownian': dict(d=g('d_brownian')),
            'confined': dict(d=g('d_confined'), l=g('l_confined'), q=g('q_confined')),
            'directed': dict(d=g('d_directed'), q=g('q_directed'), v=g('v_directed'))}
    B = obs.shape[0]
    e = {m: (g(f'e_{m}') if f'e_{m}' in fit else torch.ones(B)) for m in args}     # per-track error factors
    r = np.array(data['obs'], float); sd = np.sqrt(np.array(var))
    with torch.no_grad():
        for model, p in args.items():
            sel = np.flatnonzero(cls == model)
            if not len(sel):
                continue
            i = torch.as_tensor(sel)
            _, _, rs, rv = _kalman(model, obs[i], var[i]*(e[model][i]**2)[:, None, None], mask[i],
                                   **{k: v[i] for k, v in p.items()}, smooth=True, positions=True)
            r[sel], sd[sel] = rs.numpy(), np.sqrt(rv.numpy())
    return r, sd


# ------------------------------------------------------------------ two-state switching model
def _imm(obs, var, mask, d_still, d_move, q, v, p_sm, p_ms, keep=False):
    """Interacting-multiple-model filter for 'still' (0) and 'moving' (1) states sharing (r, w) per axis.

    still:  r' = r + e (d_still),  w' ~ N(0, v^2) (a fresh velocity if the bead starts to move)
    moving: r' = r + w + e (d_move), w' = w + u (q)
    Returns the approximate log-likelihood per track and, with keep=True, per-step filtered state
    probabilities, predicted state probabilities, state likelihoods and the moving-state velocity.
    """
    B, T, A = obs.shape
    # q, v, p_sm, p_ms: scalars (shared) or one value per track row (B,), so that several parameter
    # sets can run as copies in one batch (fit_switching's Newton steps)
    row = lambda x: torch.as_tensor(x, dtype=obs.dtype).reshape(-1).expand(B)
    q, v, p_sm, p_ms = row(q), row(v), row(p_sm), row(p_ms)
    Ptr = torch.stack([torch.stack([1-p_sm, p_sm], -1), torch.stack([p_ms, 1-p_ms], -1)], 1)   # (B, from, to)
    pi_m = p_sm/(p_sm+p_ms)
    mu = torch.stack([1-pi_m, pi_m], -1)
    y0 = obs[:, 0]
    m0 = y0[:, None].expand(B, 2, A).clone(); m1 = torch.zeros(B, 2, A)
    p00 = var[:, 0][:, None].expand(B, 2, A).clone(); p01 = torch.zeros(B, 2, A); p11 = (v*v)[:, None, None]*torch.ones(B, 2, A)
    # both states in one tensor (state axis 1): still has b = 0 and a fresh velocity each step,
    # moving has b = 1 and keeps its velocity; a = 1 in both
    bst = torch.tensor([0., 1.], dtype=obs.dtype)[None, :, None]
    keepv = bst
    col = lambda x: torch.as_tensor(x, dtype=obs.dtype).reshape(-1, 1).expand(B, 1)
    Q00 = torch.stack([col(d_still**2), col(d_move**2)], 1)                     # (B, 2, 1)
    Q11 = torch.stack([(v*v)[:, None], (q*q)[:, None]], 1)                      # (B, 2, 1)
    # missing frames: huge measurement variance (update is a no-op) and no likelihood term
    maskf = mask.to(obs.dtype)
    veff = torch.where(mask[..., None], var, torch.full_like(var, BIG_VAR))
    ll = torch.zeros(B)
    rec = dict(filt=[mu], pred=[mu], lik=[torch.ones(B, 2)], vel=[m1[:, 1]]) if keep else None
    for t in range(1, T):
        c = torch.bmm(mu[:, None, :], Ptr)[:, 0]                   # predicted state probabilities (B, 2)
        wT = (mu[:, :, None]*Ptr/c[:, None, :].clamp_min(1e-300)).transpose(1, 2)   # mixing weights [b, to, from]
        # moment-matched mixing: n = sum_i w_ij m_i, cov = sum_i w_ij (P_i + m_i m_i') - n n',
        # with positions centred on state 0 to avoid cancellation
        ref = m0[:, :1]
        c0 = m0-ref
        n0c, n1 = torch.bmm(wT, c0), torch.bmm(wT, m1)
        q00 = torch.bmm(wT, p00+c0*c0)-n0c*n0c
        q01 = torch.bmm(wT, p01+c0*m1)-n0c*n1
        q11 = torch.bmm(wT, p11+m1*m1)-n1*n1
        n0 = n0c+ref
        # predict both states at once
        pm0 = n0+bst*n1
        pm1 = n1*keepv
        pp00 = q00+2*bst*q01+bst*q11+Q00                          # b^2 = b for b in {0, 1}
        pp01 = (q01+bst*q11)*keepv
        pp11 = q11*keepv+Q11
        # update
        S = pp00+veff[:, t][:, None]
        e = obs[:, t][:, None]-pm0
        k0, k1 = pp00/S, pp01/S
        m0 = pm0+k0*e
        m1 = pm1+k1*e
        p00 = pp00*(1-k0)
        p01 = pp01*(1-k0)
        p11 = pp11-k1*pp01
        logL = -.5*maskf[:, t, None]*((torch.log(S)+e*e/S).sum(-1)+A*LOG2PI)      # (B, 2)
        shift = logL.max(-1, keepdim=True).values
        joint = c*torch.exp(logL-shift)
        tot = joint.sum(-1)
        ll = ll+torch.log(tot)+shift[:, 0]
        mu = joint/tot[:, None]
        if keep:
            rec['filt'].append(mu); rec['pred'].append(c); rec['lik'].append(torch.exp(logL-shift)); rec['vel'].append(m1[:, 1])
    return (ll, rec) if keep else ll


def _shared_newton(obs, var, mask, raw, unpack, fixed_d, iters=30, h=1e-4, max_step=2., tol=1e-5):
    """Newton ascent of the summed log-likelihood over a few parameters shared by all tracks (in place
    on `raw`). The current point and its P finite-difference neighbours run as copies of all tracks
    in one batch (one autograd pass gives each copy's gradient, hence the Hessian); step lengths are
    tried as copies too, forward only. The batch is small, so copies cost little extra time."""
    names = list(raw)
    P, B = len(names), obs.shape[0]
    r = torch.stack([raw[k].detach() for k in names])
    eye = torch.eye(P, dtype=r.dtype)

    def totals(R):                                       # R (C, P) -> summed ll per copy (C,)
        C = R.shape[0]
        p = unpack({k: R[:, i].repeat_interleave(B) for i, k in enumerate(names)})
        if fixed_d is not None:
            p['d_still'] = p['d_move'] = fixed_d.repeat(C, 1)
        return _imm(obs.repeat(C, 1, 1), var.repeat(C, 1, 1), mask.repeat(C, 1), **p).reshape(C, B).sum(1)

    with torch.no_grad():
        cur = float(totals(r[None])[0])
    for _ in range(iters):
        R = torch.cat([r[None], r[None]+h*eye]).requires_grad_(True)
        G, = torch.autograd.grad(totals(R).sum(), R)
        g, H = G[0], (G[1:]-G[0])/h
        H = .5*(H+H.T)
        e, U = torch.linalg.eigh(-H)
        e = torch.clamp(e, min=1e-6*float(e.abs().max())+1e-9)
        step = U @ ((U.T @ g)/e)
        step = step*min(1., max_step/max(float(step.norm()), 1e-12))
        alphas = torch.tensor([1., .5, .25, .1, .03], dtype=r.dtype)
        with torch.no_grad():
            tot = totals(r[None]+alphas[:, None]*step[None])
        k = int(tot.argmax())
        if float(tot[k]) <= cur:
            break
        gain = float(tot[k])-cur
        r, cur = r+alphas[k]*step, float(tot[k])
        if gain < tol:
            break
    with torch.no_grad():
        for i, k in enumerate(names):
            raw[k].copy_(r[i])


def fit_switching(data, steps=100, lr=.05, kappa=1., d_track=None, method='lbfgs', e_track=None):
    """Fit the two-state (still / moving) population model; per-localization P(moving) and velocity.

    d_track: each track's own random step size (µm/frame, e.g. from its directed-model fit). Both
    states use it, so they differ only by the persistent velocity: a bead that simply jitters more
    than others is not mistaken for a moving one. Without it, one step size is fitted for all.
    e_track: each track's factor on its localization errors (from the same fit), likewise.
    """
    obs = torch.as_tensor(data['obs']); var = torch.as_tensor(data['var'])*torch.as_tensor(np.broadcast_to(np.asarray(kappa, float), (3,)))**2
    if e_track is not None:
        var = var*torch.as_tensor(np.asarray(e_track, float))[:, None, None]**2
    mask = torch.as_tensor(data['mask'])
    raw = {'d': torch.tensor(math.log(.02), requires_grad=True),
           'q': torch.tensor(math.log(.02), requires_grad=True),
           'v': torch.tensor(math.log(.1), requires_grad=True),
           'p_sm': torch.tensor(math.log(.02/.98), requires_grad=True),
           'p_ms': torch.tensor(math.log(.05/.95), requires_grad=True)}
    fixed_d = torch.as_tensor(np.asarray(d_track, float))[:, None] if d_track is not None else None
    if fixed_d is not None:
        del raw['d']
    def unpack(r=None):
        r = raw if r is None else r
        p = {k: torch.exp(r[k]) for k in ('q', 'v')}
        p['d_still'] = p['d_move'] = fixed_d if fixed_d is not None else torch.exp(r['d'])
        p['p_sm'] = torch.sigmoid(r['p_sm'])*.5
        p['p_ms'] = torch.sigmoid(r['p_ms'])*.5
        return p
    with _threads(1):     # small tensors: one thread is fastest
        if method == 'newton':
            _shared_newton(obs, var, mask, raw, unpack, fixed_d, steps)
        else:             # few shared parameters: L-BFGS converges in a few dozen evaluations (Adam ~300 steps);
            # stopping when the summed ll (~1e4) changes by < 1e-4 moves P(moving) by < 1e-4
            _optimize(lambda: -_imm(obs, var, mask, **unpack()).sum(), list(raw.values()), steps, lr, method, tol_change=1e-4)
    with torch.no_grad():
        p = unpack()
        ll, rec = _imm(obs, var, mask, **p, keep=True)
        filt = torch.stack(rec['filt'], 1)                           # (B, T, 2)
        lik = torch.stack(rec['lik'], 1)
        Ptr = torch.stack([torch.stack([1-p['p_sm'], p['p_sm']]), torch.stack([p['p_ms'], 1-p['p_ms']])])
        # backward pass over the discrete states with the per-step state likelihoods (Kim-type smoother)
        B, T, _ = filt.shape
        beta = torch.ones(B, 2)
        post = [filt[:, -1]]
        for t in range(T-2, -1, -1):
            beta = (Ptr[None]*(lik[:, t+1]*beta)[:, None, :]).sum(-1)
            beta = beta/beta.sum(-1, keepdim=True)
            g = filt[:, t]*beta
            post.append(g/g.sum(-1, keepdim=True))
        post = torch.stack(post[::-1], 1)
        vel = torch.stack(rec['vel'], 1)
    params = {k: (float(v) if v.numel() == 1 else float(v.median())) for k, v in p.items()}
    params['step_size'] = 'per track (median shown)' if fixed_d is not None else 'shared'
    params['log_likelihood'] = float(ll.sum())
    return params, post[:, :, 1].numpy(), vel.numpy()


# ------------------------------------------------------------------ multi-state model (ExaTrack)
STATE_TYPES = ('diffusive', 'directed', 'confined')


def _survival(mean, shape, M, per_frame=24):
    """Gamma survival S(n) = P(lifetime > n frames), n = 0..M (differentiable in mean and shape)."""
    rate = shape/mean
    x = torch.linspace(0, M+1, (M+1)*per_frame+1)[1:]                         # skip x = 0 (pdf singular if shape < 1)
    logpdf = shape*torch.log(rate)+(shape-1)*torch.log(x)-rate*x-torch.lgamma(shape)
    pdf = torch.exp(logpdf)
    dx = x[1]-x[0]
    cdf = torch.cumsum(pdf, 0)*dx
    cdf = torch.cat([torch.zeros(1), cdf])
    idx = torch.arange(0, M+1)*per_frame
    return torch.clamp(1-cdf[idx], min=1e-12, max=1.)


def _hazards(means, shapes, M):
    """h[k, tau-1]: probability to leave state k after tau frames in it (tau = 1..M; tau = M also covers longer)."""
    rows = []
    for k in range(len(means)):
        S = _survival(means[k], shapes[k], M)
        rows.append(torch.clamp((S[:-1]-S[1:])/S[:-1], 1e-6, 1-1e-6))
    return torch.stack(rows)                                                   # (K, M)


def _merge(w, m, P):
    """Moment-matched Gaussian mixture: weights (B,S), means (B,S,A,3), covs (B,S,A,3,3) -> (B,A,3), (B,A,3,3)."""
    ws = w/w.sum(1, keepdim=True).clamp_min(1e-300)
    mm = torch.einsum('bs,bsai->bai', ws, m)
    d = m-mm[:, None]
    PP = torch.einsum('bs,bsaij->baij', ws, P+d[..., :, None]*d[..., None, :])
    return mm, PP


class MultiStateModel:
    """Tracks switching between motion states with gamma-distributed state lifetimes (ExaTrack-style).

    Hidden variables per axis: position r, velocity w (used by directed states) and well centre c
    (used by confined states). Within a state:
      diffusive: r' = r + e                           (an immobile state is diffusive with a tiny d)
      directed:  r' = r + w + e,           w' = w + u (q)
      confined:  r' = (1-l)(r + e) + l c,  c' = c + u (q)
    On entering a directed state a new velocity is drawn, w ~ N(0, v^2); on entering a confined state
    a new well centre, c ~ N(r, s^2): the previous state's anomalous variable is integrated out and a
    new one initialized, as in ExaTrack. The time spent in a state follows a gamma distribution (mean,
    shape; shape 1 = memoryless, first-order kinetics); each hypothesis is (state, frames in it) with
    frames capped at M, so transition probabilities can depend on the time already spent. Hypotheses
    entering a state are merged by moment matching (one Gaussian per hypothesis). A backward pass over
    the discrete hypotheses gives per-localization state probabilities.
    """

    def __init__(self, states=('diffusive', 'directed'), M=12, gamma=True, d_track=None, error='fixed', e_track=None,
                 u_track=None, lifetime_prior=None, ordered=False, other=False, still_d_max=None):
        """error: how the reported localization errors are scaled -- 'fixed' (as given), 'track' (by
        each track's factor e_track, e.g. from fit_single_models) or 'state' (a free factor per state,
        as in ExaTrack).
        u_track (B, 3): each track's direction of motion (e.g. fit_single_models' u_directed). Directed
        motion is then along that line in 3D: velocity = s u with a scalar speed s, re-drawn on entering
        the state (s ~ N(0, v^2), so either way along the line) and changing by q per frame, and x, y, z
        are filtered jointly (_run_line) -- one speed informed by all three axes, instead of three
        independent velocity components. Needs states without 'confined' (e.g. two diffusive states).
        lifetime_prior (mean_frames, sd_log): a log-normal prior on each state's mean lifetime, for
        long-lived states.
        ordered: the states are stages passed through once, in the given order (e.g. still -> moving ->
        still for an indentation that is then held): every track starts in the first, can only move on
        to the next, and stays in the last. Each track has its own switch times (from the backward
        pass); lifetimes then only set the per-frame switch probability (gamma=False, M=1 is enough).
        Diffusive stages then share one step size d (a 'still' stage cannot absorb motion the others don't).
        other (ordered, with u_track): each stage k gets a side state 'other directed motion' that it can
        enter at any time and that returns to stage k (so the stage is remembered): motion unrelated to
        the sequence, e.g. a bead pushed by a cell. Its velocity is a free 3D vector (not along u),
        re-drawn on entering (N(0, v^2) per axis) and changing by q per frame; all side states share
        one d, q, v and mean lifetime. States K..2K-1 are the side states of stages 0..K-1.
        still_d_max (ordered): an upper bound on the diffusive stages' shared step size (smooth:
        d = still_d_max sigmoid(x)), e.g. from the still beads' single-model fits."""
        assert all(s in STATE_TYPES for s in states) and len(states) >= 2
        assert error in ('fixed', 'track', 'state') and (error != 'track' or e_track is not None)
        assert u_track is None or 'confined' not in states, 'directed-along-a-line mode has no confined state'
        assert not other or (ordered and u_track is not None), 'the other-motion states need ordered=True and u_track'
        n = len(states)
        self.n_stage, self.other = n, other
        self.side = [False]*n+[True]*n if other else [False]*n
        if other:
            states = tuple(states)+('directed',)*n
        # parameter sharing: state -> index into log_d / log_mean_life, and the key of its q, v
        if ordered:
            dg, groups = [], {}
            for k, s in enumerate(states):
                g = ('side',) if self.side[k] else ('diffusive',) if s == 'diffusive' else (k,)
                dg.append(groups.setdefault(g, len(groups)))
            self.d_group = dg
            self.still_group = groups.get(('diffusive',))
        else:
            self.d_group = list(range(len(states)))
            self.still_group = None
        self.still_d_max = still_d_max if self.still_group is not None else None
        self.life_group = list(range(n))+[n]*n if other else list(range(n))
        self.qv = ['o' if self.side[k] else str(k) for k in range(len(states))]
        self.states, self.K, self.M, self.gamma = tuple(states), len(states), M, gamma
        self.d_track = torch.as_tensor(np.asarray(d_track, float)) if d_track is not None else None
        self.error = error
        self.e_track = torch.as_tensor(np.asarray(e_track, float)) if error == 'track' else None
        if u_track is not None:
            u = torch.as_tensor(np.asarray(u_track, float))
            self.u = u/u.norm(dim=1, keepdim=True).clamp_min(1e-12)
        else:
            self.u = None
        self.lifetime_prior = lifetime_prior
        K = self.K
        t = lambda v: torch.tensor(v, requires_grad=True)
        # smooth bounds (mean = 1 + e^x, shape = .3 + e^x): a hard clamp would stop the gradient at the bound
        self.raw = {'log_mean_life': t([math.log(29.)]*(max(self.life_group)+1)), 'log_shape': t([math.log(.7)]*K),
                    'init_logits': t([0.]*K)}
        if self.d_track is None:
            self.raw['log_d'] = t([math.log(.02)]*(max(self.d_group)+1))
        if K > 2:
            self.raw['dest_logits'] = t(np.zeros((K, K)).tolist())
        for k, s in enumerate(states):
            if s == 'directed' and f'log_q{self.qv[k]}' not in self.raw:
                self.raw[f'log_q{self.qv[k]}'] = t(math.log(.01)); self.raw[f'log_v{self.qv[k]}'] = t(math.log(.05))
            elif s == 'confined':
                self.raw[f'logit_l{k}'] = t(math.log(.2/.8)); self.raw[f'log_q{k}'] = t(math.log(.01))
                self.raw[f'log_s{k}'] = t(math.log(.1))
        if error == 'state':
            self.raw['log_e'] = t([0.]*K)
        if not gamma:
            del self.raw['log_shape']
        self.ordered = ordered
        if ordered:                     # every track starts in the first stage; moves only forward
            self.raw.pop('init_logits', None)
            self.raw.pop('dest_logits', None)
            if n > 2:                   # which later stage is entered (skipping is allowed): fitted
                self.raw['dest_logits'] = t((np.triu(np.full((n, n), -4.), 2)).tolist())   # skips start unlikely
            if other:                   # leaving stage k for its side state instead of the next stage
                self.raw['side_logits'] = t([-3.]*n)
                with torch.no_grad():   # the last stage is only left for its side state: rarely
                    self.raw['log_mean_life'][n-1] = math.log(200.)

    def init_from(self, fit):
        """Start each state from the single-model fits of the tracks that model describes best."""
        gC = fit['ll_confined']-fit['ll_brownian']; gD = fit['ll_directed']-fit['ll_brownian']
        med = lambda v, sel: float(np.median(v[sel])) if sel.any() else float(np.median(v))
        brown = (gC < 1) & (gD < 1)
        conf = gC > np.maximum(gD, 1)
        dirc = gD > np.maximum(gC, 1)
        with torch.no_grad():
            seen, done = 0, set()
            for k, s in enumerate(self.states):
                if self.d_group[k] in done:                         # a shared step size: set by its first state
                    continue
                done.add(self.d_group[k])
                if s == 'diffusive':
                    d = med(fit['d_brownian'], brown)*3.**seen      # several diffusive states: slow, faster, ...
                    seen += 1
                elif s == 'directed':
                    d = med(fit['d_directed'], dirc)
                    self.raw[f'log_q{self.qv[k]}'].fill_(math.log(max(med(fit['q_directed'], dirc), 1e-4)))
                    self.raw[f'log_v{self.qv[k]}'].fill_(math.log(max(med(fit['v_directed'], dirc), 1e-3)))
                else:
                    d = med(fit['d_confined'], conf)
                    l = min(max(med(fit['l_confined'], conf), .02), .95)
                    self.raw[f'logit_l{k}'].fill_(math.log(l/(1-l)))
                    self.raw[f'log_q{k}'].fill_(math.log(max(med(fit['q_confined'], conf), 1e-4)))
                    self.raw[f'log_s{k}'].fill_(math.log(max(d, 1e-3)))     # new well near the current position
                if 'log_d' in self.raw:
                    self._set_d(self.d_group[k], d)
        return self

    def n_params(self):
        return sum(v.numel() for v in self.raw.values())

    def params(self):
        r, K = self.raw, self.K
        p = {'mean_life': 1.+torch.exp(r['log_mean_life'])[self.life_group],
             'shape': .3+torch.exp(r['log_shape']) if self.gamma else torch.ones(K)}
        if self.ordered:
            p['init'] = torch.eye(K)[0]
            # forward only among the stages (k -> j > k, skips allowed); stage k <-> its side state n+k;
            # without side states the last stage is never left
            n = self.n_stage
            allowed = torch.zeros(K, K, dtype=torch.bool)
            allowed[:n, :n] = torch.triu(torch.ones(n, n), 1) > 0
            logits = torch.zeros(K, K)
            if 'dest_logits' in r:
                logits = torch.cat([torch.cat([r['dest_logits'], torch.zeros(n, K-n)], 1), torch.zeros(K-n, K)])
            if self.other:
                i = torch.arange(n)
                allowed[i, n+i] = True; allowed[n+i, i] = True
                logits = logits+torch.diag_embed(r['side_logits'], n)[:K, :K]
            logits = torch.where(allowed, logits, torch.full((K, K), -1e9))
            p['dest'] = torch.softmax(logits, 1)*allowed.any(1, keepdim=True)
            return self._state_params(p)
        p['init'] = torch.softmax(r['init_logits'], 0)
        if 'dest_logits' in r:
            logits = r['dest_logits']-1e9*torch.eye(K)
            p['dest'] = torch.softmax(logits, 1)
        else:
            p['dest'] = 1-torch.eye(K)
        return self._state_params(p)

    def _state_params(self, p):
        r = self.raw
        p['d'] = self._d_groups(r['log_d'])[self.d_group] if 'log_d' in r else None
        p['e'] = torch.exp(r['log_e']) if 'log_e' in r else None                # per-state error factors
        for k, s in enumerate(self.states):
            if s == 'directed':
                p[f'q{k}'], p[f'v{k}'] = torch.exp(r[f'log_q{self.qv[k]}']), torch.exp(r[f'log_v{self.qv[k]}'])
            elif s == 'confined':
                p[f'l{k}'] = torch.sigmoid(r[f'logit_l{k}'])*.999
                p[f'q{k}'], p[f's{k}'] = torch.exp(r[f'log_q{k}']), torch.exp(r[f'log_s{k}'])
        return p

    def _d_groups(self, raw):
        """Step size per parameter group from raw['log_d'] (the capped still group: raw is a logit)."""
        d = torch.exp(raw)
        if self.still_d_max is not None:
            g = self.still_group
            d = torch.cat([d[:g], (self.still_d_max*torch.sigmoid(raw[g])).reshape(1), d[g+1:]])
        return d

    def _set_d(self, g, d):
        if g == self.still_group and self.still_d_max is not None:
            f = min(max(d/self.still_d_max, 1e-4), .9)
            self.raw['log_d'][g] = math.log(f/(1-f))
        else:
            self.raw['log_d'][g] = math.log(max(d, 1e-4))

    def _haz(self, p):
        """Per-frame leaving probabilities (K, M); in the ordered mode the last stage is never left."""
        haz = _hazards(p['mean_life'], p['shape'], self.M)
        if self.ordered and not self.other:
            haz = torch.cat([haz[:-1], torch.zeros_like(haz[-1:])])
        return haz

    def _dynamics(self, p, B):
        """F (H,3,3) and Q (B,H,A=1,3,3) per hypothesis (state k repeated M times)."""
        K, M = self.K, self.M
        F, Q = [], []
        for k, s in enumerate(self.states):
            d2 = (self.d_track**2)[:, None] if self.d_track is not None else (p['d'][k]**2).expand(B, 1)
            f = torch.eye(3)
            q = torch.zeros(B, 3, 3)
            if s == 'diffusive':
                q = q+torch.diag_embed(torch.stack([d2[:, 0], torch.zeros(B), torch.zeros(B)], -1))
            elif s == 'directed':
                f = f.clone(); f[0, 1] = 1.
                q = q+torch.diag_embed(torch.stack([d2[:, 0], (p[f'q{k}']**2).expand(B), torch.zeros(B)], -1))
            else:
                l = p[f'l{k}']
                f = torch.stack([torch.stack([1-l, torch.zeros(()), l]), torch.tensor([0., 1., 0.]), torch.tensor([0., 0., 1.])])
                q = q+torch.diag_embed(torch.stack([(1-l)**2*d2[:, 0], torch.zeros(B), (p[f'q{k}']**2).expand(B)], -1))
            F += [f]*M
            Q += [q]*M
        return torch.stack(F), torch.stack(Q, 1)[:, :, None]

    def _enter(self, p, m, P, k):
        """Reset the anomalous variable of state k on entering it (m (B,A,3), P (B,A,3,3))."""
        s = self.states[k]
        m, P = m.clone(), P.clone()
        if s == 'directed':
            m[..., 1] = 0.
            P[..., 1, :] = 0.; P[..., :, 1] = 0.
            P[..., 1, 1] = p[f'v{k}']**2
        elif s == 'confined':
            m[..., 2] = m[..., 0]
            P[..., 2, :] = P[..., 0, :]; P[..., :, 2] = P[..., :, 0]
            P[..., 2, 2] = P[..., 0, 0]+p[f's{k}']**2
        return m, P

    def _state_consts(self, p, B):
        """Per-state linear-model coefficients, each (Bp, K, 1, 1) with Bp = B (d per track) or 1:
        r' = a r + b h + e (q00), h' = h + u (q11); and on entering: h reset from r (confined) or to
        N(0, v^2) (directed)."""
        K = self.K
        d2 = (self.d_track**2)[:, None] if self.d_track is not None else (p['d']**2)[None, :]    # (Bp, K) or (B, 1)
        d2 = d2.expand(-1, K)
        zero = torch.zeros(K)
        l = torch.stack([p[f'l{k}'] if s == 'confined' else zero[0] for k, s in enumerate(self.states)])
        q = torch.stack([p[f'q{k}'] if s != 'diffusive' else zero[0] for k, s in enumerate(self.states)])
        v = torch.stack([p[f'v{k}'] if s == 'directed' else zero[0] for k, s in enumerate(self.states)])
        s_ = torch.stack([p[f's{k}'] if s == 'confined' else zero[0] for k, s in enumerate(self.states)])
        isO = torch.tensor([float(o) for o in self.side])                      # other-motion side states
        isD = torch.tensor([float(s == 'directed') for s in self.states])-isO
        isC = torch.tensor([float(s == 'confined') for s in self.states])
        a = 1-isC*l
        b = isC*l+isD
        c = lambda x: x.reshape(-1, K)[..., None, None]
        return dict(a=c(a), b=c(b), q00=c(a*a*d2), q11=c((1-isO)*q*q), isC=c(isC), isD=c(isD), v2=c(isD*v*v), s2=c(s_*s_),
                    isO=c(isO), qo2=c(isO*q*q), vo2=c(isO*v*v))

    def run(self, obs, var, mask, keep=False):
        """Filter over all (state, age) hypotheses; returns ll per track (and, with keep, per-frame
        state probabilities and the directed-state velocity).

        Exact reduced form of _run_dense: a hypothesis in state k only ever uses its own anomalous
        variable h (velocity for directed, well centre for confined, none for diffusive) -- the other
        one is re-drawn whenever its state is entered -- so each hypothesis carries (r, h) per axis:
        means m_r, m_h and covariances P_rr, P_rh, P_hh, updated elementwise like _kalman_lin, and
        hypotheses entering a state are merged in r alone. Tensors are (B, K, M, A)."""
        if self.u is not None:
            return self._run_line(obs, var, mask, keep)
        B, T, A = obs.shape
        K, M = self.K, self.M
        H = K*M
        p = self.params()
        haz = self._haz(p)                          # (K, M)
        C = self._state_consts(p, B)
        a, b, q00, q11 = C['a'], C['b'], C['q00'], C['q11']
        aa, ab2, bb = a*a, 2*a*b, b*b
        age = torch.stack([(lambda S: S/S.sum())(_survival(p['mean_life'][k], p['shape'][k], M)[:-1]) for k in range(K)])
        mu = (p['init'][:, None]*age)[None].expand(B, K, M).clone()             # (B, K, M)
        y0, v0 = obs[:, 0], var[:, 0]
        full = lambda x: x[:, None, None, :].expand(B, K, M, A)
        mr, Prr = full(y0).clone(), full(v0).clone()

        def reset_h(mr_, Prr_):                                                # the anomalous variable on entering
            mh = C['isC']*mr_
            Prh = C['isC']*Prr_
            Phh = C['isC']*(Prr_+C['s2'])+C['isD']*C['v2']
            return mh, Prh, Phh
        mh, Prh, Phh = reset_h(mr, Prr)
        mh, Prh, Phh = mh.expand(B, K, M, A).clone(), Prh.expand(B, K, M, A).clone(), Phh.expand(B, K, M, A).clone()
        stay, leave = 1-haz, haz                                              # (K, M)
        # into target k from state j: dest[j, k] (0 on the diagonal)
        dest = p['dest']
        maskf = mask.to(obs.dtype)
        if self.e_track is not None:                                           # per-track error factors
            var = var*(self.e_track**2)[:, None, None]
        veff = torch.where(mask[..., None], var, torch.full_like(var, BIG_VAR))
        e2 = (p['e']**2)[None, :, None, None] if p['e'] is not None else 1.    # per-state error factors
        ll = torch.zeros(B)
        isD_h = torch.tensor([float(s == 'directed') for s in self.states])[None, :, None]
        rec = dict(filt=[mu.reshape(B, H)], lik=[torch.ones(B, H)], vel=[torch.zeros(B, A)]) if keep else None
        for t in range(1, T):
            sw, lw = mu*stay, mu*leave                                        # (B, K, M)
            # entering each target state (age 1): mass leaving the other states, merged in r only
            W = torch.einsum('bjm,jk->bjmk', lw, dest).reshape(B, H, K)       # weights [b, source hyp, target]
            sW = W.sum(1)                                                     # (B, K)
            ref = mr[:, :1, :1]                                               # centre positions to avoid cancellation
            cr = (mr-ref).reshape(B, H, A)
            Wn = W/sW[:, None, :].clamp_min(1e-300)
            WT = Wn.transpose(1, 2)                                           # (B, K, H)
            n_r = torch.bmm(WT, cr)                                           # (B, K, A)
            e_P = torch.bmm(WT, Prr.reshape(B, H, A)+cr*cr)-n_r*n_r
            n_r = n_r+ref[:, 0]
            e_mr, e_Prr = n_r[:, :, None], e_P[:, :, None]                    # (B, K, 1, A)
            e_mh, e_Prh, e_Phh = reset_h(e_mr, e_Prr)
            e_mu = sW[:, :, None]                                             # (B, K, 1)
            if M > 1:
                # staying: age tau -> tau+1; the last age merges ages M-1 and M
                w1, w2 = sw[:, :, M-2:M-1], sw[:, :, M-1:]
                wt = (w1+w2).clamp_min(1e-300)
                f1, f2 = (w1/wt)[..., None], (w2/wt)[..., None]
                x = lambda Z: (Z[:, :, M-2:M-1], Z[:, :, M-1:])
                (r1, r2), (h1, h2) = x(mr), x(mh)
                lr_ = f1*r1+f2*r2; lh = f1*h1+f2*h2
                dr1, dr2, dh1, dh2 = r1-lr_, r2-lr_, h1-lh, h2-lh
                (Prr1, Prr2), (Prh1, Prh2), (Phh1, Phh2) = x(Prr), x(Prh), x(Phh)
                lPrr = f1*(Prr1+dr1*dr1)+f2*(Prr2+dr2*dr2)
                lPrh = f1*(Prh1+dr1*dh1)+f2*(Prh2+dr2*dh2)
                lPhh = f1*(Phh1+dh1*dh1)+f2*(Phh2+dh2*dh2)
                cat = lambda e, mid, last: torch.cat([e, mid[:, :, :M-2], last], 2)
                c = torch.cat([e_mu, sw[:, :, :M-2], w1+w2], 2)               # predicted hypothesis probabilities
                mr, mh = cat(e_mr, mr, lr_), cat(e_mh, mh, lh)
                Prr, Prh, Phh = cat(e_Prr, Prr, lPrr), cat(e_Prh, Prh, lPrh), cat(e_Phh, Phh, lPhh)
            else:   # a single age: entering and staying merge
                w1, w2 = e_mu, sw
                wt = (w1+w2).clamp_min(1e-300)
                f1, f2 = (w1/wt)[..., None], (w2/wt)[..., None]
                lr_ = f1*e_mr+f2*mr; lh = f1*e_mh+f2*mh
                dr1, dr2, dh1, dh2 = e_mr-lr_, mr-lr_, e_mh-lh, mh-lh
                Prr = f1*(e_Prr+dr1*dr1)+f2*(Prr+dr2*dr2)
                Prh = f1*(e_Prh+dr1*dh1)+f2*(Prh+dr2*dh2)
                Phh = f1*(e_Phh+dh1*dh1)+f2*(Phh+dh2*dh2)
                mr, mh, c = lr_, lh, w1+w2
            # predict each hypothesis with its state's dynamics
            pm_r, pm_h = a*mr+b*mh, mh
            pp_rr = aa*Prr+ab2*Prh+bb*Phh+q00
            pp_rh = a*Prh+b*Phh
            pp_hh = Phh+q11
            # update (missing frames: veff huge, so the gain is ~0)
            S = pp_rr+veff[:, t][:, None, None]*e2
            e = obs[:, t][:, None, None]-pm_r
            k_r, k_h = pp_rr/S, pp_rh/S
            mr, mh = pm_r+k_r*e, pm_h+k_h*e
            Prr, Prh, Phh = pp_rr*(1-k_r), pp_rh*(1-k_r), pp_hh-k_h*pp_rh
            logL = -.5*maskf[:, t, None, None]*((torch.log(S)+e*e/S).sum(-1)+A*LOG2PI)   # (B, K, M)
            lg = logL+torch.log(c.clamp_min(1e-300))
            lse = torch.logsumexp(lg.reshape(B, H), 1)
            ll = ll+lse
            mu = torch.exp(lg-lse[:, None, None])
            if keep:
                vw = mu*isD_h
                vel = (vw[..., None]*mh).sum((1, 2))/vw.sum((1, 2))[:, None].clamp_min(1e-12)
                logLf = logL.reshape(B, H)
                live = torch.where(c.reshape(B, H) > 1e-200, logLf, torch.full_like(logLf, -1e300))
                rec['filt'].append(mu.reshape(B, H)); rec['lik'].append(torch.exp(logLf-live.max(1, keepdim=True).values).clamp_max(1e300))
                rec['vel'].append(vel)
        if not keep:
            return ll
        return (ll, *self._backward(rec, p, haz, B, T))

    def _run_line(self, obs, var, mask, keep=False):
        """run() with directed motion along each track's direction u in 3D (velocity = s u).

        Each (state, age) hypothesis carries the joint Gaussian of (x, y, z, s) -- plus (a, b, c), the
        free 3D velocity of the other-motion side states, when the model has them: the means and the
        entries of the symmetric covariance, as (B, K, M) tensors. The three coordinates of a frame are
        applied one after another as scalar measurements (exact: their errors are independent), so the
        axes are coupled through s and the position covariance. Entering a state merges the incoming
        hypotheses in (x, y, z) and re-draws the velocities (s ~ N(0, v^2) for a directed stage, each of
        a, b, c ~ N(0, v_o^2) for a side state; exactly 0 otherwise, so they have no effect there)."""
        B, T, A = obs.shape
        K, M = self.K, self.M
        H = K*M
        p = self.params()
        haz = self._haz(p)
        C = self._state_consts(p, B)
        b, d2, q2 = C['b'][..., 0], C['q00'][..., 0], C['q11'][..., 0]            # (Bp, K, 1)
        s2_enter = (C['isD']*C['v2'])[..., 0]
        g, qo2, w2_enter = C['isO'][..., 0], C['qo2'][..., 0], C['vo2'][..., 0]   # side states: r' = r + w
        u = [self.u[:, a][:, None, None] for a in range(3)]                        # (B, 1, 1) each
        bu = [b*ua for ua in u]                                                    # (B, K, 1)
        age = torch.stack([(lambda S: S/S.sum())(_survival(p['mean_life'][k], p['shape'][k], M)[:-1]) for k in range(K)])
        mu = (p['init'][:, None]*age)[None].expand(B, K, M).clone()
        if self.e_track is not None:
            var = var*(self.e_track**2)[:, None, None]
        maskf = mask.to(obs.dtype)
        veff = torch.where(mask[..., None], var, torch.full_like(var, BIG_VAR))
        e2 = (p['e']**2)[None, :, None] if p['e'] is not None else 1.
        ax = 'xyzsabc' if self.other else 'xyzs'
        W = ax[4:]                                                                 # side-state velocity (a, b, c)
        key = lambda i, j: ''.join(sorted(ax[i]+ax[j], key=ax.index))
        KEYS = [key(i, j) for i in range(len(ax)) for j in range(i, len(ax))]     # xx xy xz xs yy ... ss (+ a, b, c)
        kc = lambda c1, c2: key(ax.index(c1), ax.index(c2))
        # the enter-reset of the velocities: variances, all their covariances 0
        reset = {kk: s2_enter.expand(B, K, 1) if kk == 'ss' else w2_enter.expand(B, K, 1) if kk in ('aa', 'bb', 'cc')
                 else torch.zeros(B, K, 1) for kk in KEYS if set(kk) & set('s'+W)}
        full = lambda x: x.expand(B, K, M).clone()
        m = {ax[a]: full(obs[:, 0, a][:, None, None]) for a in range(3)}
        for c in 's'+W:
            m[c] = torch.zeros(B, K, M)
        P = {k: torch.zeros(B, K, M) for k in KEYS}
        for a in range(3):
            P[ax[a]*2] = full(var[:, 0, a][:, None, None])
        for kk, v in reset.items():
            P[kk] = full(v)
        # prediction x' = F x as sparse rows: x_a' = x_a + b u_a s + g w_a, s' = s, w' = w
        rows = {ax[a]: [(ax[a], None), ('s', bu[a])]+([(W[a], g)] if W else []) for a in range(3)}
        rows.update({c: [(c, None)] for c in 's'+W})
        noise = {ax[a]*2: d2 for a in range(3)}
        noise['ss'] = q2
        noise.update({c*2: qo2 for c in W})
        stay, leave = 1-haz, haz
        dest = p['dest']
        ll = torch.zeros(B)
        isD_h, isO_h = C['isD'][..., 0].expand(1, K, 1), C['isO'][..., 0].expand(1, K, 1)
        rec = dict(filt=[mu.reshape(B, H)], lik=[torch.ones(B, H)], vel=[torch.zeros(B, A)]) if keep else None
        for t in range(1, T):
            sw, lw = mu*stay, mu*leave
            # entering: merge the leaving mass of the other states in (x, y, z); the velocities are re-drawn
            Wt = torch.einsum('bjm,jk->bjmk', lw, dest).reshape(B, H, K)
            sW = Wt.sum(1)
            WT = (Wt/sW[:, None, :].clamp_min(1e-300)).transpose(1, 2)             # (B, K, H)
            ref = {c: m[c][:, :1, :1] for c in 'xyz'}
            cen = {c: (m[c]-ref[c]).reshape(B, H, 1) for c in 'xyz'}
            n = {c: torch.bmm(WT, cen[c])[..., 0] for c in 'xyz'}                 # (B, K)
            e_P = {}
            for i in range(3):
                for j in range(i, 3):
                    kk = key(i, j)
                    e_P[kk] = (torch.bmm(WT, P[kk].reshape(B, H, 1)+cen[ax[i]]*cen[ax[j]])[..., 0]-n[ax[i]]*n[ax[j]])[:, :, None]
            e_m = {c: (n[c]+ref[c][:, 0])[:, :, None] for c in 'xyz'}
            for c in 's'+W:
                e_m[c] = torch.zeros(B, K, 1)
            e_P.update(reset)
            e_mu = sW[:, :, None]
            if M > 1:
                w1, w2 = sw[:, :, M-2:M-1], sw[:, :, M-1:]
                wt = (w1+w2).clamp_min(1e-300)
                f1, f2 = w1/wt, w2/wt
                lm = {c: f1*m[c][:, :, M-2:M-1]+f2*m[c][:, :, M-1:] for c in ax}
                dd = {c: (m[c][:, :, M-2:M-1]-lm[c], m[c][:, :, M-1:]-lm[c]) for c in ax}
                lP = {}
                for kk in KEYS:
                    i, j = kk[0], kk[1]
                    lP[kk] = f1*(P[kk][:, :, M-2:M-1]+dd[i][0]*dd[j][0])+f2*(P[kk][:, :, M-1:]+dd[i][1]*dd[j][1])
                cat = lambda e, mid, last: torch.cat([e, mid[:, :, :M-2], last], 2)
                c_ = torch.cat([e_mu, sw[:, :, :M-2], w1+w2], 2)
                m = {c: cat(e_m[c], m[c], lm[c]) for c in ax}
                P = {kk: cat(e_P[kk], P[kk], lP[kk]) for kk in KEYS}
            else:
                w1, w2 = e_mu, sw
                wt = (w1+w2).clamp_min(1e-300)
                f1, f2 = w1/wt, w2/wt
                lm = {c: f1*e_m[c]+f2*m[c] for c in ax}
                dd = {c: (e_m[c]-lm[c], m[c]-lm[c]) for c in ax}
                P = {kk: f1*(e_P[kk]+dd[kk[0]][0]*dd[kk[1]][0])+f2*(P[kk]+dd[kk[0]][1]*dd[kk[1]][1]) for kk in KEYS}
                m, c_ = lm, w1+w2
            # predict: m' = F m, P' = F P F^T + noise (r_a: d^2, s: q^2, a, b, c: q_o^2)
            mul = lambda f, x: x if f is None else f*x
            newm = {c: sum(mul(f, m[src]) for src, f in rows[c]) for c in ax}
            newP = {}
            for kk in KEYS:
                ci, cj = kk[0], kk[1]
                tot = 0.
                for si, fi in rows[ci]:
                    for sj, fj in rows[cj]:
                        tot = tot+mul(fi, mul(fj, P[kc(si, sj)]))
                newP[kk] = tot+noise[kk] if kk in noise else tot
            m, P = newm, newP
            # update: the three coordinates one after another (independent errors)
            logL = torch.zeros(B, K, M)
            for a in range(3):
                ca = ax[a]
                S = (P[ca*2]+veff[:, t, a][:, None, None]*e2).clamp_min(1e-12)
                e = obs[:, t, a][:, None, None]-m[ca]
                col = {c: P[kc(c, ca)] for c in ax}                                 # Cov(c, measured axis)
                m = {c: m[c]+col[c]/S*e for c in ax}
                P = {kk: P[kk]-col[kk[0]]*col[kk[1]]/S for kk in KEYS}
                logL = logL-.5*maskf[:, t, None, None]*(torch.log(S)+e*e/S+LOG2PI)
            lg = logL+torch.log(c_.clamp_min(1e-300))
            lse = torch.logsumexp(lg.reshape(B, H), 1)
            ll = ll+lse
            mu = torch.exp(lg-lse[:, None, None])
            if keep:                     # velocity of the moving hypotheses: s u (stages), (a, b, c) (side states)
                vd, vo = mu*isD_h, mu*isO_h
                vel = (vd*m['s']).sum((1, 2))[:, None]*self.u
                if W:
                    vel = vel+torch.stack([(vo*m[c]).sum((1, 2)) for c in W], -1)
                vel = vel/(vd+vo).sum((1, 2))[:, None].clamp_min(1e-12)
                logLf = logL.reshape(B, H)
                live = torch.where(c_.reshape(B, H) > 1e-200, logLf, torch.full_like(logLf, -1e300))
                rec['filt'].append(mu.reshape(B, H)); rec['lik'].append(torch.exp(logLf-live.max(1, keepdim=True).values).clamp_max(1e300))
                rec['vel'].append(vel)
        if not keep:
            return ll
        return (ll, *self._backward(rec, p, haz, B, T))

    def _backward(self, rec, p, haz, B, T):
        """Backward pass over the discrete hypotheses: per-frame state probabilities (B, T, K), velocity."""
        K, M = self.K, self.M
        H = K*M
        stay, leave = (1-haz).reshape(H), haz.reshape(H)
        Tm = torch.zeros(H, H)
        for h in range(H):
            k, tau = divmod(h, M)
            Tm[h, k*M+min(tau+1, M-1)] += stay[h]
            for j in range(K):
                if j != k:
                    Tm[h, j*M] += leave[h]*p['dest'][k, j]
        filt = torch.stack(rec['filt'], 1); lik = torch.stack(rec['lik'], 1)
        beta = torch.ones(B, H)
        post = [filt[:, -1]]
        for t in range(T-2, -1, -1):
            beta = (Tm[None]*(lik[:, t+1]*beta)[:, None, :]).sum(-1)
            beta = beta/beta.sum(-1, keepdim=True).clamp_min(1e-300)
            g = filt[:, t]*beta
            post.append(g/g.sum(-1, keepdim=True).clamp_min(1e-300))
        post = torch.stack(post[::-1], 1).reshape(B, T, K, M).sum(-1)
        return post, torch.stack(rec['vel'], 1)

    def _run_dense(self, obs, var, mask, keep=False):
        """Reference implementation with a full 3x3 covariance of (r, w, c) per axis and hypothesis
        (slow; run() is its exact reduced form). Kept for the unit test."""
        if self.error == 'state':
            raise NotImplementedError('the dense reference has no per-state error factors')
        if self.e_track is not None:
            var = var*(self.e_track**2)[:, None, None]
        B, T, A = obs.shape
        K, M = self.K, self.M
        H = K*M
        p = self.params()
        haz = self._haz(p)                          # (K, M)
        F, Q = self._dynamics(p, B)
        # initial hypotheses: state fractions x stationary age distribution (proportional to survival)
        age = []
        for k in range(K):
            S = _survival(p['mean_life'][k], p['shape'][k], M)[:-1]
            age.append(S/S.sum())
        mu = (p['init'][:, None]*torch.stack(age)).reshape(1, H).expand(B, H).clone()
        m = torch.zeros(B, H, A, 3); P = torch.zeros(B, H, A, 3, 3)
        m[..., 0] = obs[:, None, 0]; m[..., 2] = obs[:, None, 0]
        P[..., 0, 0] = var[:, None, 0]
        P[..., 1, 1] = 1.; P[..., 2, 2] = 1.
        for k in range(K):
            mk, Pk = self._enter(p, m[:, k*M:(k+1)*M], P[:, k*M:(k+1)*M], k)
            m = torch.cat([m[:, :k*M], mk, m[:, (k+1)*M:]], 1); P = torch.cat([P[:, :k*M], Pk, P[:, (k+1)*M:]], 1)
        stay = (1-haz).reshape(H); leave = haz.reshape(H)
        state_of = torch.arange(H)//M
        ll = torch.zeros(B)
        rec = dict(filt=[mu], lik=[torch.ones(B, H)], vel=[torch.zeros(B, A)]) if keep else None
        for t in range(1, T):
            sw, lw = mu*stay, mu*leave                                        # (B, H)
            new_mu, new_m, new_P = [], [], []
            for k in range(K):
                sl = slice(k*M, (k+1)*M)
                # entering k (tau = 1): mass leaving any other state, times the destination probability
                src = lw*p['dest'][state_of, k][None]
                src = src*(state_of != k)[None]
                em, eP = _merge(src, m, P)
                em, eP = self._enter(p, em, eP, k)
                parts_mu, parts_m, parts_P = [src.sum(1)], [em], [eP]
                # staying: (k, tau) -> (k, tau+1); the last age merges (k, M-1) and (k, M)
                if M > 1:
                    parts_mu += [sw[:, sl][:, j] for j in range(M-2)]
                    parts_m += [m[:, sl][:, j] for j in range(M-2)]
                    parts_P += [P[:, sl][:, j] for j in range(M-2)]
                    tw = sw[:, sl][:, M-2:]
                    tm, tP = _merge(tw, m[:, sl][:, M-2:], P[:, sl][:, M-2:])
                    parts_mu.append(tw.sum(1)); parts_m.append(tm); parts_P.append(tP)
                else:   # a single age: staying and entering merge
                    tw = torch.stack([src.sum(1), sw[:, sl][:, 0]], 1)
                    tm, tP = _merge(tw, torch.stack([em, m[:, sl][:, 0]], 1), torch.stack([eP, P[:, sl][:, 0]], 1))
                    parts_mu, parts_m, parts_P = [tw.sum(1)], [tm], [tP]
                new_mu.append(torch.stack(parts_mu, 1)); new_m.append(torch.stack(parts_m, 1)); new_P.append(torch.stack(parts_P, 1))
            c = torch.cat(new_mu, 1)                                          # predicted hypothesis probabilities
            m = torch.cat(new_m, 1); P = torch.cat(new_P, 1)
            # predict with each hypothesis' dynamics
            m = torch.einsum('hij,bhaj->bhai', F, m)
            P = torch.einsum('hij,bhajk,hlk->bhail', F, P, F)+Q
            # update with the observation (H = [1, 0, 0]) where present
            o = mask[:, t][:, None, None]
            S = P[..., 0, 0]+var[:, t][:, None]
            e = obs[:, t][:, None]-m[..., 0]
            Kg = P[..., :, 0]/S[..., None]
            m = torch.where(o[..., None], m+Kg*e[..., None], m)
            P = torch.where(o[..., None, None], P-Kg[..., :, None]*P[..., 0, :][..., None, :], P)
            logL = torch.where(mask[:, t][:, None], (-.5*(LOG2PI+torch.log(S)+e*e/S)).sum(-1), torch.zeros(B, H))
            a = logL+torch.log(c.clamp_min(1e-300))       # log-sum-exp over prior x likelihood (robust to empty hypotheses)
            lse = torch.logsumexp(a, 1)
            ll = ll+lse
            mu = torch.exp(a-lse[:, None])
            if keep:
                vw = mu*torch.as_tensor([s == 'directed' for s in self.states]).repeat_interleave(M)[None]
                vel = torch.einsum('bh,bha->ba', vw, m[..., 1])/vw.sum(1, keepdim=True).clamp_min(1e-12)
                live = torch.where(c > 1e-200, logL, torch.full_like(logL, -1e300))
                rec['filt'].append(mu); rec['lik'].append(torch.exp(logL-live.max(1, keepdim=True).values).clamp_max(1e300))
                rec['vel'].append(vel)
        if not keep:
            return ll
        # transition matrix between hypotheses, for the backward pass
        Tm = torch.zeros(H, H)
        for h in range(H):
            k, tau = divmod(h, M)
            Tm[h, k*M+min(tau+1, M-1)] += stay[h]
            for j in range(K):
                if j != k:
                    Tm[h, j*M] += leave[h]*p['dest'][k, j]
        filt = torch.stack(rec['filt'], 1); lik = torch.stack(rec['lik'], 1)
        beta = torch.ones(B, H)
        post = [filt[:, -1]]
        for t in range(T-2, -1, -1):
            beta = (Tm[None]*(lik[:, t+1]*beta)[:, None, :]).sum(-1)
            beta = beta/beta.sum(-1, keepdim=True).clamp_min(1e-300)
            g = filt[:, t]*beta
            post.append(g/g.sum(-1, keepdim=True).clamp_min(1e-300))
        post = torch.stack(post[::-1], 1).reshape(B, T, K, M).sum(-1)       # (B, T, K) state probabilities
        return ll, post, torch.stack(rec['vel'], 1)

    def fit(self, data, kappa=1., steps=150, lr=.05, method='lbfgs', tol_change=1e-3, restarts=1):
        """Maximum likelihood over the shared parameters: L-BFGS (default; stops when the summed ll
        changes by < tol_change) or Adam (`steps` steps at rate lr). restarts: run L-BFGS again (fresh
        curvature memory) while a run still improves the ll by > 1 -- it can stop early after a poor
        line search, far from the optimum."""
        obs = torch.as_tensor(data['obs']); mask = torch.as_tensor(data['mask'])
        var = torch.as_tensor(data['var'])*torch.as_tensor(np.broadcast_to(np.asarray(kappa, float), (3,)))**2
        def loss():
            nll = -self.run(obs, var, mask).sum()
            if self.lifetime_prior:          # log-normal prior on each state's mean lifetime
                mean0, sd_log = self.lifetime_prior
                nll = nll+(((torch.log(self.params()['mean_life'])-math.log(mean0))/sd_log)**2).sum()/2
            if not torch.isfinite(nll):      # a trial step into an unusable region: make the line search back off
                return sum((v*0).sum() for v in self.raw.values())+1e30
            return nll
        with _threads(1):
            prev = None
            for _ in range(max(restarts, 1)):
                _optimize(loss, list(self.raw.values()), steps, lr, method, tol_change=tol_change)
                with torch.no_grad():
                    now = float(loss())
                if prev is not None and prev-now < 1.:
                    break
                prev = now
        with torch.no_grad():
            ll, post, vel = self.run(obs, var, mask, keep=True)
        self.log_likelihood = float(ll.sum())
        n = int(mask.sum())*3
        self.bic = -2*self.log_likelihood+self.n_params()*math.log(n)
        return post.numpy(), vel.numpy()

    def describe(self):
        p = self.params()
        out = {'states': list(self.states), 'log_likelihood': self.log_likelihood, 'bic': self.bic,
               'n_params': self.n_params(), 'lifetime_cap_frames': self.M, 'step_size': 'per track' if self.d_track is not None else 'per state',
               'localization_error': {'fixed': 'as reported (per-axis scale)', 'track': 'per-track factor from the single-model fits',
                                      'state': 'free factor per state'}[self.error]}
        p = {k: v.detach() if torch.is_tensor(v) else v for k, v in p.items()}
        for k, s in enumerate(self.states):
            e = {'type': s, 'mean_lifetime_frames': float(p['mean_life'][k]), 'lifetime_shape': float(p['shape'][k]),
                 'initial_fraction': float(p['init'][k])}
            if self.side[k]:
                e['type'] = f'other directed motion (3D velocity), returns to state{k-self.n_stage}'
            if p['d'] is not None:
                e['d'] = float(p['d'][k])
            if p['e'] is not None:
                e['error_factor'] = float(p['e'][k])
            if s == 'directed':
                e.update(velocity_change_q=float(p[f'q{k}']), initial_velocity_spread_v=float(p[f'v{k}']))
            if s == 'confined':
                e.update(confinement_factor_l=float(p[f'l{k}']), well_diffusion_q=float(p[f'q{k}']), initial_well_spread_s=float(p[f's{k}']))
            if self.K > 2:
                e['destination'] = {f'state{j}_{"other" if self.side[j] else self.states[j]}': float(p['dest'][k, j])
                                    for j in range(self.K) if j != k and not (self.other and p['dest'][k, j] == 0)}
            out[f'state{k}'] = e
        return out


# ------------------------------------------------------------------ driver
# stage names for the ordered-stages mode (--stages) and the motion model each uses
STAGE_KINDS = {'still': 'diffusive', 'hold': 'diffusive', 'rest': 'diffusive', 'diffusive': 'diffusive',
               'indent': 'directed', 'retract': 'directed', 'move': 'directed', 'directed': 'directed'}
DEFAULT_STAGES = ('still', 'indent', 'hold', 'retract', 'still')


def fit_stages(data, fit, cls, stages=DEFAULT_STAGES, kappa=1., other=True, still_quantile=.5):
    """Ordered stages (MultiStateModel ordered=True) for experiments with a known sequence, e.g. an
    indentation: every bead starts in the first stage and can only move forward (stages it does not
    show are skipped). Directed stages move along each bead's own direction in 3D (u_directed), with
    each bead's localization-error factor from its single-model fit; the diffusive ('still') stages
    share one step size. other: each stage also has a side state for directed motion unrelated to the
    sequence (any 3D direction), entered from and returning to that stage. Returns the model, per-frame
    state probabilities (B, T, K, or 2K with the side states K..2K-1) and the velocity (B, T, 3)."""
    kinds = tuple(STAGE_KINDS[s] for s in stages)
    e_best = np.select([cls == 'confined', cls == 'directed'], [fit['e_confined'], fit['e_directed']], fit['e_brownian'])
    # the still stages' step size may not exceed what still beads show (by default the median of the
    # Brownian beads' fits): otherwise a still stage with a large step becomes a catch-all for moving
    # beads (with the 90th percentile, the 20x cells fit ended in that optimum, ll 16894 against 47004)
    still = cls == 'brownian'
    d_max = float(np.quantile(fit['d_brownian'][still] if still.sum() >= 5 else fit['d_brownian'], still_quantile))
    make = lambda o: MultiStateModel(kinds, M=1, gamma=False, ordered=True, error='track', e_track=e_best,
                                     u_track=fit['u_directed'], other=o, still_d_max=max(d_max, 1e-3)).init_from(fit)
    model = make(False)
    post, vel = model.fit(data, kappa=kappa, restarts=10)
    if other:           # then add the side states, starting from the fitted stages
        full = make(True)
        with torch.no_grad():
            n = len(kinds)
            full.raw['log_mean_life'][:n-1] = model.raw['log_mean_life'][:n-1]
            full.raw['log_d'][:len(model.raw['log_d'])] = model.raw['log_d']
            for k, v in model.raw.items():
                if k.startswith(('log_q', 'log_v')) or k == 'dest_logits':
                    full.raw[k].copy_(v)
        model = full
        post, vel = model.fit(data, kappa=kappa, restarts=4)
    return model, post, vel


def analyse(results, movie=None, frame_interval_ms=None, coords='stabilized', alpha=.05, kappa=None, min_points=10,
            stages=None):
    """stages: None (the two-state moving/still model) or a sequence of stage names (see STAGE_KINDS),
    e.g. DEFAULT_STAGES for an indentation: the ordered-stages model instead."""
    results = Path(results)
    info = json.loads((results/'run_info.json').read_text())
    px = float(info['pixel_size_um'])
    movies = [movie] if movie else list(info['movies'])
    summaries = {}
    for name in movies:
        csv = results/f'{name}_localizations.csv'
        data = load_tracks(csv, px, coords, min_points)
        scale = estimate_error_scale(data) if kappa is None else np.broadcast_to(np.asarray(kappa, float), (3,))
        fit = fit_single_models(data, kappa=scale)
        cls, rho_c, rho_d = classify(fit, alpha)
        staged = None
        if stages:
            smodel, spost, vel = fit_stages(data, fit, cls, stages, kappa=scale)
            n = len(stages)
            moving_k = [i for i, s in enumerate(stages) if STAGE_KINDS[s] == 'directed']
            p_other = spost[:, :, n:].sum(-1)                   # other directed motion (the side states)
            stage_post = spost[:, :, :n]+spost[:, :, n:]        # the stage each frame belongs to, side trips included
            p_move = spost[:, :, moving_k].sum(-1)+p_other
            # per-frame label (0-based): the most probable stage, or n (= 'other moving') when that is more likely than not
            label = np.where(p_other > .5, n, stage_post.argmax(-1))
            staged = dict(stages=list(stages), names=list(stages)+['other moving'], model=smodel.describe(),
                          post=stage_post, p_other=p_other, label=label)
            sw = None
        else:
            sw, p_move, vel = fit_switching(data, kappa=scale, d_track=fit['d_directed'], e_track=fit.get('e_directed'))
        ref, ref_sd = refine_positions(data, fit, cls, kappa=scale)
        dt = frame_interval_ms/1000 if frame_interval_ms else None
        per_s = (lambda x, p=1: x/dt**p) if dt else (lambda x, p=1: x)
        unit_t = 's' if dt else 'frame'
        m = data['mask']
        rows = []
        for b, tid in enumerate(data['ids']):
            f0 = data['frame0'][b]
            pm = p_move[b][m[b]]
            frames = np.flatnonzero(m[b])+f0
            moving = frames[pm > .5]
            sp = np.sqrt((vel[b]**2).sum(-1))[m[b]]
            l = fit['l_confined'][b]
            rows.append(dict(
                track_number=int(tid), n_points=int(m[b].sum()), first_frame=int(frames[0]), last_frame=int(frames[-1]),
                motion_class=cls[b], rho_confined=float(rho_c[b]), rho_directed=float(rho_d[b]),
                D=float(per_s(fit['d_brownian'][b]**2/2)),
                directed_mean_speed=float(per_s(fit['mean_speed'][b])), directed_max_speed=float(per_s(fit['max_speed'][b])),
                directed_velocity_change=float(per_s(fit['q_directed'][b])),
                confinement_factor=float(l), confinement_sd=float(fit['d_confined'][b]*(1-l)/math.sqrt(max(1-(1-l)**2, 1e-12))),
                frames_moving=int((pm > .5).sum()), fraction_moving=float((pm > .5).mean()),
                moving_from=int(moving[0]) if len(moving) else -1, moving_to=int(moving[-1]) if len(moving) else -1,
                speed_while_moving=float(per_s(sp[pm > .5].mean())) if (pm > .5).any() else float('nan')))
            if staged:          # the frame each stage was entered (most probable stage per frame; -1: never)
                st = staged['post'][b][m[b]].argmax(-1)
                for i, s in enumerate(staged['stages'][1:], 1):
                    rows[-1][f'stage{i+1}_{s}_from'] = int(frames[np.argmax(st == i)]) if (st == i).any() else -1
                rows[-1]['frames_other_moving'] = int((staged['p_other'][b][m[b]] > .5).sum())
        header = list(rows[0])
        np.savetxt(results/f'{name}_track_motion.csv', np.array([[r[h] for h in header] for r in rows], dtype=object),
                   fmt='%s', delimiter=',', header=','.join(header), comments='')
        # refined positions as shifts from the analysed coordinates (x, y in px, z in µm), so they can be
        # added to any coordinate set (raw, corrected, stabilized), plus their standard deviations
        to_px = np.array([1/px, 1/px, 1.])
        shift, sd = (ref-data['obs'])*to_px, ref_sd*to_px
        states = []
        K = len(staged['stages']) if staged else 0
        for b, tid in enumerate(data['ids']):
            for t in np.flatnonzero(m[b]):
                v = vel[b, t]
                extra = []
                if staged:              # label (1-based; K+1: other moving), each stage's probability, P(other)
                    ps = staged['post'][b, t]
                    extra = [int(staged['label'][b, t])+1, *(float(x) for x in ps), float(staged['p_other'][b, t])]
                states.append((int(t+data['frame0'][b]), int(tid), float(p_move[b, t]),
                               *(float(per_s(x)) for x in v), *shift[b, t], *sd[b, t], *extra))
        header = ('frame_number,track_number,p_moving,vx,vy,vz,'
                  'refine_dx_px,refine_dy_px,refine_dz_um,refined_sd_x_px,refined_sd_y_px,refined_sd_z_um')
        fmt = ['%d', '%d', '%.4f']+['%.5g']*9
        if staged:
            header += ',stage,'+','.join(f'p_stage{i+1}' for i in range(K))+',p_other'
            fmt += ['%d']+['%.4f']*(K+1)
        np.savetxt(results/f'{name}_motion_states.csv', np.array(states), fmt=fmt, delimiter=',', header=header, comments='')
        raw_sd = np.sqrt(data['var']*np.asarray(scale)**2)*to_px
        refined = dict(model='each track\'s own aTrack model (Kalman filter + RTS smoother)',
                       median_sd_before_xyz=[float(np.median(raw_sd[..., a][m])) for a in range(3)],
                       median_sd_after_xyz=[float(np.median(sd[..., a][m])) for a in range(3)],
                       units='x, y in px; z in µm')
        counts = {k: int((cls == k).sum()) for k in ('brownian', 'confined', 'directed')}
        summaries[name] = dict(
            tracks=len(data['ids']), classes=counts, alpha=alpha, coordinates=coords, time_unit=unit_t,
            localization_error_scale_xyz=[round(float(k), 3) for k in scale],
            error_scale_source='estimated from the step anticorrelation' if kappa is None else 'given',
            units=f'D in µm²/{unit_t}; speeds in µm/{unit_t}; step lengths d in µm per frame',
            refined_positions=refined,
            method='aTrack-style per-track likelihood-ratio classification and ExaTrack-style ' +
                   ('ordered-stages model (directed stages along each bead\'s own 3D direction; still stages share one step '
                    'size; from any stage, a side state for unrelated directed motion in any 3D direction)' if staged else 'two-state switching model') +
                   '; Kalman-filter likelihoods with each localization\'s own error')
        if staged:
            lab = [staged['label'][b][m[b]] for b in range(len(data['ids']))]
            st_all = np.concatenate(lab)
            summaries[name]['stages'] = dict(names=staged['names'], model=staged['model'],
                                             labels='per frame: the most probable stage, or "other moving" (last) '
                                                    'when directed motion unrelated to the sequence is more likely than not',
                                             fraction_of_localizations=[float((st_all == i).mean()) for i in range(K+1)],
                                             beads_entering=[int(sum((l == i).any() for l in lab)) for i in range(K+1)])
            print(f"{name}: {counts} | stages {staged['stages']}: beads entering each "
                  f"{summaries[name]['stages']['beads_entering']}", flush=True)
        else:
            summaries[name]['switching_model'] = dict(sw, stationary_fraction_moving=sw['p_sm']/(sw['p_sm']+sw['p_ms']),
                                                      mean_moving_duration_frames=1/sw['p_ms'], mean_still_duration_frames=1/sw['p_sm'])
            print(f"{name}: {counts} | moving state: step {sw['d_move']:.3f} µm, velocity change {sw['q']:.3f} µm/frame, "
                  f"still -> moving {sw['p_sm']:.4f}/frame, moving -> still {sw['p_ms']:.3f}/frame", flush=True)
    (results/'track_motion_summary.json').write_text(json.dumps(summaries, indent=2))
    return summaries


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('results', type=Path)
    ap.add_argument('--movie')
    ap.add_argument('--frame-interval', type=float, help='ms between frames (reports D in µm²/s and speeds in µm/s)')
    ap.add_argument('--coords', choices=['stabilized', 'corrected', 'raw'], default='stabilized')
    ap.add_argument('--alpha', type=float, default=.05, help='likelihood-ratio threshold (aTrack default 0.05)')
    ap.add_argument('--error-scale', type=float, help='multiply the reported localization errors by this '
                                                       '(default: estimated per axis from the data)')
    ap.add_argument('--min-points', type=int, default=10)
    ap.add_argument('--stages', help='ordered stages instead of the two-state moving/still model: comma-separated names '
                                     f'from {sorted(STAGE_KINDS)}, or "indentation" for {",".join(DEFAULT_STAGES)}')
    a = ap.parse_args(argv)
    stages = None
    if a.stages:
        stages = DEFAULT_STAGES if a.stages == 'indentation' else tuple(s.strip() for s in a.stages.split(','))
        bad = [s for s in stages if s not in STAGE_KINDS]
        if bad or len(stages) < 2:
            ap.error(f'--stages: unknown or too few stage names {bad or stages}')
    analyse(a.results, a.movie, a.frame_interval, a.coords, a.alpha, a.error_scale, a.min_points, stages)


if __name__ == '__main__':
    main()
