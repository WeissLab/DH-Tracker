import unittest

import numpy as np
import torch
from scipy.stats import multivariate_normal

import track_analysis as TA


def make(tracks, sig=(.06, .06, .6)):
    B, T = len(tracks), tracks[0].shape[0]
    return dict(ids=np.arange(1, B+1), obs=np.stack(tracks), var=np.broadcast_to(np.asarray(sig)**2, (B, T, 3)).copy(),
                mask=np.ones((B, T), bool), frame0=np.ones(B, int), T=T)


class TrackAnalysisTests(unittest.TestCase):
    def test_brownian_likelihood_is_exact(self):
        # Kalman log-likelihood (conditioned on the first point) equals the Gaussian likelihood of the increments' model
        rng = np.random.default_rng(1)
        T, d, s = 12, .3, .2
        y = np.cumsum(rng.normal(0, d, (T, 3)), 0)+rng.normal(0, s, (T, 3))
        data = make([y], sig=(s, s, s))
        ll = TA._kalman('brownian', torch.as_tensor(data['obs']), torch.as_tensor(data['var']), torch.as_tensor(data['mask']),
                        d=torch.tensor([d])).item()
        # direct: with a flat prior on the start, r0 | y0 ~ N(y0, s^2) and y_t = r0 + (sum of t steps) + noise,
        # so y_1.. | y0 is Gaussian with mean y0 and cov d^2 min(t, t') + s^2 (shared r0) + s^2 I
        t = np.arange(1, T)
        cov = d*d*np.minimum.outer(t, t)+s*s+s*s*np.eye(T-1)
        direct = sum(multivariate_normal(np.full(T-1, y[0, a]), cov).logpdf(y[1:, a]) for a in range(3))
        self.assertAlmostEqual(ll, direct, places=6)

    def test_smoothed_positions_are_exact(self):
        # RTS positions and variances equal direct Gaussian conditioning on the whole track
        rng = np.random.default_rng(3)
        T, d, s = 15, .1, .3
        y = np.cumsum(rng.normal(0, d, (T, 3)), 0)+rng.normal(0, s, (T, 3))
        data = make([y], sig=(s, s, s))
        _, _, r, rv = TA._kalman('brownian', torch.as_tensor(data['obs']), torch.as_tensor(data['var']),
                                 torch.as_tensor(data['mask']), d=torch.tensor([d]), smooth=True, positions=True)
        t = np.arange(T)
        prior = s*s+d*d*np.minimum.outer(t, t)            # r_t | y_0 (flat prior on the start)
        Syy = prior[1:, 1:]+s*s*np.eye(T-1)
        G = prior[:, 1:]@np.linalg.inv(Syy)
        mean = y[0]+G@(y[1:]-y[0])
        cov = prior-G@prior[1:, :]
        np.testing.assert_allclose(r[0].numpy(), mean, atol=1e-9)
        np.testing.assert_allclose(rv[0].numpy(), np.broadcast_to(np.diag(cov)[:, None], (T, 3)), atol=1e-9)

    def test_refined_positions_reduce_error(self):
        rng = np.random.default_rng(4)
        T, truth, tracks = 91, [], []
        sig = np.array([.06, .06, .6])
        for k in range(30):
            steps = rng.normal(0, .01, (T, 3))
            if k >= 20:
                v = rng.normal(0, 1, 3); v *= .1/np.linalg.norm(v)
                steps[35:55] += v
            r = np.cumsum(steps, 0)
            truth.append(r); tracks.append(r+rng.normal(0, 1, (T, 3))*sig)
        data = make(tracks)
        fit = TA.fit_single_models(data, steps=200)
        cls, _, _ = TA.classify(fit)
        ref, sd = TA.refine_positions(data, fit, cls)
        err_raw = np.sqrt(((data['obs']-np.stack(truth))**2).mean((0, 1)))
        err_ref = np.sqrt(((ref-np.stack(truth))**2).mean((0, 1)))
        self.assertTrue(np.all(err_ref < .6*err_raw), (err_raw, err_ref))   # z: several-fold better
        # reported uncertainty is honest for beads the model describes (the still ones); for abrupt
        # starts/stops the directed model is too smooth and the z uncertainty is ~1.5-2x too small
        z = (ref-np.stack(truth))/sd
        np.testing.assert_allclose(z[:20].std((0, 1)), 1., atol=.25)

    def test_multistate_fast_filter_equals_dense(self):
        # the reduced (r, h) filter is exact: same likelihood, state probabilities, velocity and gradient
        rng = np.random.default_rng(5)
        T, tracks = 40, []
        for k in range(12):
            steps = rng.normal(0, .02, (T, 3))
            if k % 3 == 0:
                steps[15:28] += rng.normal(0, .1, 3)
            tracks.append(np.cumsum(steps, 0)+rng.normal(0, 1, (T, 3))*np.array([.06, .06, .6]))
        data = make(tracks)
        data['mask'][2, 5:8] = False                                  # missing frames too
        obs, var, mask = torch.as_tensor(data['obs']), torch.as_tensor(data['var']), torch.as_tensor(data['mask'])
        for states, M in ((('diffusive', 'directed', 'confined'), 5), (('diffusive', 'directed'), 1)):
            model = TA.MultiStateModel(states, M=M)
            with torch.no_grad():
                for v in model.raw.values():
                    v.add_(torch.as_tensor(rng.normal(0, .3, tuple(v.shape))))
                l1, p1, v1 = model.run(obs, var, mask, keep=True)
                l2, p2, v2 = model._run_dense(obs, var, mask, keep=True)
            np.testing.assert_allclose(l1.numpy(), l2.numpy(), rtol=1e-9, atol=1e-8)
            np.testing.assert_allclose(p1.numpy(), p2.numpy(), atol=1e-9)
            np.testing.assert_allclose(v1.numpy(), v2.numpy(), atol=1e-9)
            g1 = torch.autograd.grad(model.run(obs, var, mask).sum(), list(model.raw.values()))
            g2 = torch.autograd.grad(model._run_dense(obs, var, mask).sum(), list(model.raw.values()))
            for a, b in zip(g1, g2):
                np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-6, atol=1e-8)

    def test_ordered_stages_find_onsets(self):
        # still -> moving along one 3D direction (frames 20..39) -> holding; plus beads that never move
        rng = np.random.default_rng(6)
        T, tracks, onset = 70, [], []
        sig = np.array([.06, .06, .6])
        for k in range(24):
            steps = rng.normal(0, .008, (T, 3))
            if k >= 12:
                u = rng.normal(0, 1, 3); u /= np.linalg.norm(u)
                a = 20 + int(rng.integers(-3, 4))
                steps[a:a+20] += .12*u
                onset.append(a)
            tracks.append(np.cumsum(steps, 0)+rng.normal(0, 1, (T, 3))*sig)
        data = make(tracks)
        fit = TA.fit_single_models(data)
        cls, _, _ = TA.classify(fit)
        model, post, vel = TA.fit_stages(data, fit, cls, stages=('still', 'indent', 'hold'))
        stage = post.argmax(-1)                                            # (B, T), 0-based
        # never-moving beads never enter the moving stage; 'still' and 'hold' are the same diffusive model,
        # so for them the switch between the two follows the timing learned from the moving beads
        self.assertTrue(np.all(stage[:12] != 1))
        found = [int(np.argmax(stage[12+i] == 1)) for i in range(12)]
        err = np.abs(np.array(found)-(np.array(onset)))                  # the first displaced frame is onset+1
        self.assertLessEqual(float(np.median(err)), 2.)
        self.assertTrue(np.all(stage[12:, -1] == 2))                       # moving beads end holding

    def test_side_states_off_equal_plain_stages(self):
        # with the side states never entered, the extended (x, y, z, s, a, b, c) filter gives the
        # plain ordered-stages likelihood exactly
        rng = np.random.default_rng(7)
        T, tracks = 30, []
        for k in range(8):
            steps = rng.normal(0, .02, (T, 3))
            steps[10:18] += rng.normal(0, .1, 3)
            tracks.append(np.cumsum(steps, 0)+rng.normal(0, 1, (T, 3))*np.array([.06, .06, .6]))
        data = make(tracks)
        data['mask'][1, 4:6] = False
        obs, var, mask = torch.as_tensor(data['obs']), torch.as_tensor(data['var']), torch.as_tensor(data['mask'])
        u = rng.normal(0, 1, (8, 3))
        kinds = ('diffusive', 'directed', 'diffusive')
        plain = TA.MultiStateModel(kinds, M=1, gamma=False, ordered=True, u_track=u, still_d_max=.05)
        side = TA.MultiStateModel(kinds, M=1, gamma=False, ordered=True, u_track=u, still_d_max=.05, other=True)
        with torch.no_grad():
            for k, v in plain.raw.items():
                v.add_(torch.as_tensor(rng.normal(0, .3, tuple(v.shape))))
                if v.dim():
                    side.raw[k][:v.shape[0]] = v
                else:
                    side.raw[k].copy_(v)
            side.raw['side_logits'].fill_(-1e4)         # exactly 0 (a tiny leak would matter: a free 3D velocity fits far better)
            haz = side._haz
            side._haz = lambda p: haz(p)*torch.tensor([1., 1., 0., 1., 1., 1.])[:, None]   # nor is the last stage left
            l1, p1, v1 = plain.run(obs, var, mask, keep=True)
            l2, p2, v2 = side.run(obs, var, mask, keep=True)
        np.testing.assert_allclose(l1.numpy(), l2.numpy(), rtol=1e-10, atol=1e-8)
        np.testing.assert_allclose(p1.numpy(), p2[..., :3].numpy(), atol=1e-10)
        np.testing.assert_allclose(v1.numpy(), v2.numpy(), atol=1e-10)

    def test_other_motion_is_not_still(self):
        # still -> indent along u (downward, as the indenter pushes) -> hold; some beads later move again
        # in another direction (unrelated motion, e.g. pushed by a cell): that is 'other moving', not a
        # still stage with a large step
        rng = np.random.default_rng(8)
        T, tracks, onset = 80, [], {}
        sig = np.array([.06, .06, .6])
        for k in range(30):
            steps = rng.normal(0, .008, (T, 3))
            if k >= 10:
                u = rng.normal(0, 1, 3); u[2] = -abs(u[2]); u /= np.linalg.norm(u)
                a = onset[k] = 20+int(rng.integers(-3, 4))
                steps[a:a+15] += .12*u
                if k >= 22:
                    w = np.cross(u, rng.normal(0, 1, 3)); w /= np.linalg.norm(w)
                    steps[50:62] += .1*w
            tracks.append(np.cumsum(steps, 0)+rng.normal(0, 1, (T, 3))*sig)
        data = make(tracks)
        fit = TA.fit_single_models(data)
        cls, _, _ = TA.classify(fit)
        model, post, vel = TA.fit_stages(data, fit, cls, stages=('still', 'indent', 'hold'))
        p_other = post[..., 3:].sum(-1)
        self.assertGreater(float((p_other[22:, 52:61] > .5).mean()), .8)          # the unrelated moves
        self.assertLess(float((p_other[:22] > .5).mean()), .02)                    # nothing else
        stage = (post[..., :3]+post[..., 3:]).argmax(-1)
        self.assertTrue(np.all(stage[:10] != 1))                                   # still beads never indent
        found = np.array([np.argmax(stage[k] == 1) for k in range(10, 22)])        # beads that only indent
        self.assertLessEqual(float(np.median(np.abs(found-[onset[k] for k in range(10, 22)]))), 2.)
        # a bead that moves twice: both moves count as moving (the first may be called indent or other)
        p_move = post[..., [1]].sum(-1)+p_other
        self.assertGreater(float(np.mean([(p_move[k, onset[k]+2:onset[k]+13] > .5).mean() for k in range(22, 30)])), .8)

    def test_classification_and_moving_states(self):
        rng = np.random.default_rng(0)
        T, tracks = 91, []
        for k in range(24):
            steps = rng.normal(0, .01, (T, 3))
            if k >= 16:                                   # moving between frames 35 and 55
                v = rng.normal(0, 1, 3); v *= .12/np.linalg.norm(v)
                steps[35:55] += v
            tracks.append(np.cumsum(steps, 0)+rng.normal(0, 1, (T, 3))*np.array([.06, .06, .6]))
        data = make(tracks)
        fit = TA.fit_single_models(data, steps=200)
        cls, _, _ = TA.classify(fit)
        self.assertLessEqual(int((cls[:16] != 'brownian').sum()), 1)
        self.assertGreaterEqual(int((cls[16:] == 'directed').sum()), 6)
        _, p, _ = TA.fit_switching(data, steps=200, d_track=fit['d_directed'])
        self.assertLess(float((p[:16] > .5).mean()), .01)                     # still beads: not moving
        self.assertGreater(float((p[16:, 38:52] > .5).mean()), .8)            # moving beads: found while moving
        self.assertLess(float((p[16:, 65:] > .5).mean()), .05)                # and still again afterwards

    def test_error_scale_recovered(self):
        rng = np.random.default_rng(2)
        tracks = [np.cumsum(rng.normal(0, .02, (91, 3)), 0)+rng.normal(0, 1, (91, 3))*np.array([.078, .078, .78])
                  for _ in range(40)]
        k = TA.estimate_error_scale(make(tracks))          # reported errors (.06/.6) are 1.3x too small
        np.testing.assert_allclose(k, 1.3, atol=.12)


if __name__ == '__main__':
    unittest.main()
