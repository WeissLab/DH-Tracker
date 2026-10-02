"""Run with python -m unittest discover -s DHPSF_pipeline -p "test_pipeline.py" -v."""
import unittest
import numpy as np
from numpy.testing import assert_allclose
from pipeline import (Config, C, NCOL, model_jac, localize_image, track, lap, CalibrationModel,
                      axial_difference, recovery_requests, estimate_drift, reject_ring_shadows,
                      split_transient, deduplicate, reject_duplicate_lobes, fit_lateral_model, lateral_correction,
                      _lateral_design, _angle_crossing, jump_outliers, ghost_tracks, alternating_states,
                      sandwiched_tracks, find_peaks)

Z = np.linspace(-29, 29, 5801)
MODEL = CalibrationModel(Z, np.linspace(5, 175, len(Z)), 16.5+(Z/12.)**2)  # sep 16.5..22.3 px


def render(beads, shape=(160, 160), background=104., noise=2., seed=0, sigma=2.2):
    """beads: (cx, cy, angle_deg, amplitude[, sep]) with zero-based centres."""
    y, x = np.indices(shape)
    p = []
    for bead in beads:
        cx, cy, angle, amp = bead[:4]
        sep = bead[4] if len(bead) > 4 else np.interp(angle, MODEL.dense_a, MODEL.dense_sep)
        d = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])*sep/2
        for s in (-1, 1):
            p.extend([amp, cx+s*d[0], cy+s*d[1], sigma])
    p.extend([background, 0, 0])
    image = model_jac(np.array(p), x.ravel().astype(float), y.ravel().astype(float))[0].reshape(shape)
    return image+np.random.default_rng(seed).normal(0, noise, shape)


def found(rows, cx, cy, tol=.5):
    return np.any(np.hypot(rows[:, 2]-1-cx, rows[:, 5]-1-cy) < tol)


class PipelineTests(unittest.TestCase):
    def test_analytic_jacobian(self):
        rng = np.random.default_rng(10)
        x, y = rng.uniform(-10, 10, (2, 50))
        p = np.array([100, -2, 1, 2.2, 90, 5, -3, 3., 110, .1, -.2])
        _, actual = model_jac(p, x, y)
        for k in range(len(p)):
            delta = np.zeros(len(p)); delta[k] = 1e-5
            expected = (model_jac(p+delta, x, y)[0]-model_jac(p-delta, x, y)[0])/2e-5
            assert_allclose(actual[:, k], expected, atol=1e-6, rtol=1e-5)

    def test_isolated_and_overlapping_beads_subpixel(self):
        beads = [(50.3, 60.7, 30., 200.), (62.1, 71.4, 120., 150.), (120.2, 110.6, 80., 60.)]
        rows, _ = localize_image(render(beads), Config(), MODEL)
        self.assertEqual(len(rows), 3)
        for cx, cy, angle, _ in beads:
            self.assertTrue(found(rows, cx, cy, .1), (cx, cy))
            r = rows[np.argmin(np.hypot(rows[:, 2]-1-cx, rows[:, 5]-1-cy))]
            self.assertLess(abs(axial_difference(r[6], angle)), 1.)

    def test_three_spots_with_bright_shared_middle(self):
        # Two beads on one line whose inner lobes coincide: three spots, middle doubled.
        sep = float(np.interp(30., MODEL.dense_a, MODEL.dense_sep))
        d = np.array([np.cos(np.radians(30.)), np.sin(np.radians(30.))])*sep/2
        beads = [(80-d[0], 80-d[1], 30., 150., sep), (80+d[0], 80+d[1], 30., 150., sep)]
        rows, stats = localize_image(render(beads), Config(), MODEL)
        self.assertEqual(stats.get('shared_lobe_pairs', 0), 1)
        for cx, cy, *_ in beads:
            self.assertTrue(found(rows, cx, cy, .6), (cx, cy))

    def test_merged_lobes_of_two_beads_are_split(self):
        # A lobe of each bead 5 px apart fit as one wide Gaussian; only the split recovers both beads.
        sep_a, sep_b = (float(np.interp(a, MODEL.dense_a, MODEL.dense_sep)) for a in (0., 60.))
        inner = np.array([70+sep_a/2, 85.])
        d = np.array([np.cos(np.radians(60.)), np.sin(np.radians(60.))])*sep_b/2
        beads = [(70., 80., 0., 150.), (*(inner+d), 60., 150.)]
        rows, stats = localize_image(render(beads), Config(), MODEL)
        self.assertGreaterEqual(stats.get('split_lobes', 0), 1)
        self.assertEqual(len(rows), 2)
        for cx, cy, *_ in beads:
            self.assertTrue(found(rows, cx, cy), (cx, cy))
        rows, _ = localize_image(render(beads), Config(split_sigma_ratio=100.), MODEL)
        self.assertFalse(all(found(rows, cx, cy) for cx, cy, *_ in beads))

    def test_noisy_patch_gives_no_detections_and_spares_beads(self):
        beads = [(60.3, 60.7, 30., 120.), (140.2, 60.4, 100., 60.)]
        image = render(beads, shape=(200, 200))
        image[96:160, 96:160] += np.random.default_rng(3).normal(0, 16., (64, 64))   # 8x the camera noise, 2x2 tiles
        rows, _ = localize_image(image, Config(), MODEL)
        for cx, cy, *_ in beads:
            self.assertTrue(found(rows, cx, cy), (cx, cy))
        self.assertEqual(len(rows), 2)
        # the patch's noise peaks (each a Gaussian to fit) are what the local threshold removes
        in_patch = lambda cfg: int(np.sum([(96 <= x < 160) & (96 <= y < 160) for x, y in find_peaks(image, cfg)[0]]))
        self.assertLess(in_patch(Config()), in_patch(Config(local_noise_tile_px=0))/5)

    def test_signal_outputs(self):
        amp, sigma, background = 150., 2.2, 104.
        rows, _ = localize_image(render([(80.2, 79.6, 40., amp)], background=background, sigma=sigma), Config(), MODEL)
        self.assertEqual(len(rows), 1)
        r = rows[0]
        lobe = 2*np.pi*amp*sigma**2
        assert_allclose(r[[C['lobeSignal1'], C['lobeSignal2']]], lobe, rtol=.03)
        self.assertAlmostEqual(r[C['backgroundLevel']], background, delta=.5)
        within = 1-np.exp(-Config().roi_radius_px**2/(2*sigma**2))   # fraction of a Gaussian inside the disk
        assert_allclose(r[C['roiSignal']], 2*lobe*within, rtol=.05)

    def test_camera_calibration_and_likelihood_fit(self):
        import tempfile, tifffile, camera_calibration
        from pathlib import Path
        rng = np.random.default_rng(8)
        tmp = Path(tempfile.mkdtemp())
        shape = (96, 96)
        offset_map = 100+rng.normal(0, 1.5, shape)
        read_sd = 1.2*np.exp(rng.normal(0, .2, shape))
        dark = (offset_map+rng.normal(0, 1, (200,)+shape)*read_sd).astype(np.float32)
        tifffile.imwrite(tmp/'dark.tif', dark)
        camera_calibration.main(['--dark', str(tmp/'dark.tif'), '--electrons-per-adu', '1.25', '--out', str(tmp/'cam.npz')])
        with np.load(tmp/'cam.npz') as z:
            assert_allclose(z['offset'], offset_map, atol=.6)   # 200 frames: SD of the mean <= ~0.12 ADU
            assert_allclose(np.sqrt(z['variance']), read_sd, rtol=.2)
            self.assertAlmostEqual(float(z['gain']), .8)
        # a bead on this camera: the likelihood fit localizes it and reports a finite error bar
        photons = np.full(shape, 3.)
        y, x = np.indices(shape)
        for cx, cy in ((39.7, 44.1), (55.9, 50.3)):
            photons += 30*np.exp(-((x-cx)**2+(y-cy)**2)/(2*2.1**2))
        image = offset_map+.8*rng.poisson(photons)+rng.normal(0, 1, shape)*read_sd
        cfg = Config(noise_model='poisson', camera_file=str(tmp/'cam.npz'))
        rows, _ = localize_image(image, cfg, None)
        self.assertEqual(len(rows), 1)
        self.assertTrue(found(rows, 47.8, 47.2, .3))
        self.assertTrue(0 < rows[0, C['anglePrecisionDeg']] < 3)

    def test_edge_bead_is_fitted_with_clipped_window(self):
        rows, _ = localize_image(render([(8.5, 80.2, 90., 150.)]), Config(), MODEL)
        self.assertTrue(found(rows, 8.5, 80.2, .2))

    def test_ring_shadow_rejected_but_real_neighbour_kept(self):
        rows = np.zeros((3, NCOL))
        rows[:, 8] = 1
        # bright pair, faint parallel pair 7 px away (ring artefact), faint pair 30 px away
        for i, (x, y, a) in enumerate([(50, 50, 400.), (50, 57, 40.), (50, 80, 40.)]):
            rows[i, [0, 1, 2, 3, 4, 5]] = [x-8, x+8, x, y, y, y]
            rows[i, [C['amplitude1'], C['amplitude2']]] = a
        kept, dropped = reject_ring_shadows(rows, Config())
        self.assertEqual(dropped, 1)
        assert_allclose(sorted(kept[:, 5]), [50, 80])

    def test_phantom_pair_reusing_a_lobe_is_dropped(self):
        def pair(x1, y1, x2, y2, a, shared=0):
            r = np.zeros(NCOL)
            r[[0, 1, 2, 3, 4, 5]] = [x1, x2, (x1+x2)/2, y1, y2, (y1+y2)/2]
            r[[C['amplitude1'], C['amplitude2'], C['sharedLobe']]] = [a, a, shared]
            return r
        rows = np.array([pair(50, 50, 66, 50, 300.),        # real bead
                         pair(66, 50.5, 80, 55, 40.),        # phantom: reuses lobe 2 and a ring arc
                         pair(34, 50, 50.4, 50, 150., 1),    # validated shared-lobe neighbour, strong own lobe
                         pair(34, 80, 50, 50.2, 10., 1)])    # shared-lobe pair with a weak own lobe
        kept, dropped = reject_duplicate_lobes(rows, Config(), noise=2.)
        self.assertEqual(dropped, 2)
        assert_allclose(sorted(kept[:, 2]), [42.2, 58])

    def test_satellite_track_rejected_real_neighbour_kept(self):
        rows = []
        for f in range(1, 11):
            for t, (x, angle, amp) in enumerate([(100, 0., 300.), (104, 3., 40.), (124, 60., 40.)], start=1):
                r = np.zeros(NCOL)
                d = 8*np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
                r[[0, 1, 2, 3, 4, 5, 6, 8, 9]] = [x+.1*f-d[0], x+.1*f+d[0], x+.1*f, 50-d[1], 50+d[1], 50, angle, f, t]
                r[[C['amplitude1'], C['amplitude2']]] = amp
                r[C['minLobeSNR']], r[C['templateScore']] = 20, np.nan
                rows.append(r)
        kept, rejected, reasons = split_transient(np.array(rows), Config())
        self.assertTrue(set(reasons) <= {3, 5}); self.assertEqual(len(rejected), 10)   # satellite or ghost rule
        self.assertEqual(len(np.unique(kept[:, 9])), 2)   # the bright bead and the rotated neighbour

    def test_alternating_fit_states_minority_flagged(self):
        # a crowded bead whose fit flips to a second state (+9 um) in every third frame, split by
        # the z gate into interleaved tracks; a normal bead nearby (15 px) with ordinary z noise
        rng = np.random.default_rng(3)
        rows, z = [], []
        for f in range(1, 31):
            for t, x, zz in ((1 if f % 3 else 2, 100., 9. if f % 3 == 0 else 0.), (3, 115., 2.)):
                r = np.zeros(NCOL)
                r[[2, 5, 8, 9]] = [x+rng.normal(0, .2), 50+rng.normal(0, .2), f, t]
                rows.append(r); z.append(zz+rng.normal(0, .5))
        rows, z = np.array(rows), np.array(z)
        bad = alternating_states(rows, Config(), z=z)
        self.assertTrue(np.all(bad == (rows[:, 9] == 2)))

    def test_sandwiched_track_between_agreeing_neighbours(self):
        rows, z = [], []
        for f in range(1, 61):
            # one place: z 0 (frames 1-20), +9 (23-38), 0 (41-60); a second bead drifting 0 -> 9 smoothly
            if f <= 20 or f >= 41 or 23 <= f <= 38:
                t = 1 if f <= 20 else (3 if f >= 41 else 2)
                r = np.zeros(NCOL); r[[2, 5, 8, 9]] = [100, 50, f, t]; rows.append(r); z.append(9. if t == 2 else 0.)
            r = np.zeros(NCOL); r[[2, 5, 8, 9]] = [200, 50, f, 4]; rows.append(r); z.append(9*f/60)
        s = sandwiched_tracks(np.array(rows), Config(), z=np.array(z))
        self.assertEqual(s, {2})

    def test_ghost_tracks_and_z_aware_linking(self):
        rows = []
        def loc(f, t, x1, y1, x2, y2, amp, z):
            r = np.zeros(NCOL)
            r[[0, 1, 2, 3, 4, 5, 7, 8, 9]] = [x1, x2, (x1+x2)/2, y1, y2, (y1+y2)/2, z, f, t]
            r[6] = np.degrees(np.arctan2(y2-y1, x2-x1)) % 180
            r[[C['amplitude1'], C['amplitude2']]] = amp
            return r
        for f in range(1, 11):
            rows.append(loc(f, 1, 100, 100, 116, 100, 20., 0.))     # bright bead A
            rows.append(loc(f, 2, 108, 88, 124, 88, 20., 1.))       # bright bead B
            rows.append(loc(f, 3, 94, 100, 116, 88, 5., -50.))      # side-lobe/crossed ghost on A and B lobes
            rows.append(loc(f, 4, 300, 300, 316, 300, 20., 2.))     # unrelated bead
            rows.append(loc(f, 5, 124, 88, 140, 90, 20., 1.5))      # real neighbour sharing ONE lobe with B
        rows = np.array(rows)
        self.assertEqual(ghost_tracks(rows, Config()), {3})
        # z-aware linking: a 50 um jump between frames is not linked when a model is given
        z = np.linspace(-29, 29, 5801)
        model = CalibrationModel(z, np.linspace(5, 175, len(z)), np.full(len(z), 16.))
        seq = np.zeros((6, NCOL))
        seq[:, 8] = np.arange(1, 7); seq[:, 2] = 50; seq[:, 5] = 50
        seq[:, C['lobeSeparationPixels']] = 16.; seq[:, C['anglePrecisionDeg']] = .3
        seq[:3, 6], seq[3:, 6] = 90., 140.       # angle jump = ~17 um here
        self.assertEqual(len(set(track(seq, Config())[:, 9])), 1)             # xy-only linking joins them
        self.assertEqual(len(set(track(seq, Config(), model=model)[:, 9])), 2)

    def test_noise_only_image_gives_no_detections(self):
        rows, _ = localize_image(render([], seed=4), Config(), MODEL)
        self.assertEqual(len(rows), 0)

    def test_lap_global_not_greedy_and_gates(self):
        cost = np.array([[1., 2.], [1.1, 100.]])
        self.assertEqual(set(lap(cost, 50.)), {(0, 1), (1, 0)})
        self.assertEqual(lap(np.full((2, 3), np.inf), 1.), [])
        self.assertEqual(lap(np.array([[1, np.inf], [np.inf, np.inf]]), 1.), [(0, 0)])

    def test_tracking_keeps_slow_movers_and_closes_gaps(self):
        rows = np.zeros((9, NCOL))
        rows[:, 8] = [1, 2, 3, 6, 7, 1, 2, 3, 4]
        rows[:, 2] = [10, 12.5, 15, 22.5, 25, 100, 100.3, 100.1, 100.2]
        rows[:, 5] = 50
        result = track(rows, Config())
        self.assertEqual(len(set(result[:5, 9])), 1)   # 2.5 px steps and a 3-frame gap
        self.assertEqual(len(set(result[5:, 9])), 1)
        self.assertNotEqual(result[0, 9], result[5, 9])

    def test_all_trackers_follow_crossing_movers(self):
        # Two beads moving at constant velocity pass within 6 px of each other; one frame missing.
        rows = []
        for t, (x0, vx) in enumerate([(0., 4.), (40., -4.)]):
            for f in range(1, 11):
                if t == 0 and f == 6:
                    continue
                r = np.zeros(NCOL)
                r[[2, 5, 8]] = [x0+vx*(f-1), 50+3*t, f]
                rows.append(r)
        rows = np.array(rows)
        self.assertEqual(len(track(rows, Config(), method='nearest')), len(rows))  # greedy baseline may swap
        for method in ('lap', 'kalman'):
            result = track(rows, Config(), method=method)
            self.assertEqual(len(set(result[rows[:, 5] == 50, 9])), 1, method)
            self.assertEqual(len(set(result[rows[:, 5] == 53, 9])), 1, method)
        # 7 px/frame with frames 4-5 missing: the 21 px jump exceeds the LAP gap gate, but the
        # Kalman prediction carries the learned velocity through the gap.
        fast = np.zeros((6, NCOL))
        fast[:, 8] = [1, 2, 3, 6, 7, 8]
        fast[:, 2] = 7*(fast[:, 8]-1)
        self.assertEqual(len(set(track(fast, Config(), method='kalman')[:, 9])), 1)
        self.assertGreater(len(set(track(fast, Config(), method='lap')[:, 9])), 1)

    def test_jump_outliers_flip_back_and_forth(self):
        # A slowly moving bead whose pairing flips to a 40-degree-rotated, 5.6 px shifted state
        # in some frames, including an alternating stretch (like tracks 113/116 of the 10x movie).
        rows = np.zeros((30, NCOL))
        rows[:, 8] = np.arange(1, 31)
        rows[:, 9] = 1
        rows[:, 2], rows[:, 5], rows[:, 6] = 100+.1*np.arange(30), 50, 179.
        flipped = [5, 12, 18, 20, 22]
        rows[flipped, 2] += 1.; rows[flipped, 5] += 5.5; rows[flipped, 6] = 139.
        bad = jump_outliers(rows, Config())
        assert_allclose(np.flatnonzero(bad), flipped)
        # smooth fast motion (2 px/frame) is not an outlier
        rows[:, 2], rows[:, 5], rows[:, 6] = 100+2*np.arange(30), 50, 179.
        self.assertFalse(jump_outliers(rows, Config()).any())
        # slow rotation per um (10x-like): an 8 degree flip is only caught through z
        slow = CalibrationModel(np.linspace(-75, 75, 3001), np.linspace(125, 235, 3001), np.full(3001, 16.5))
        rows[:, 2], rows[:, 6] = 100., 180.
        rows[:, C['lobeSeparationPixels']] = 16.5
        rows[[4, 17, 18, 19, 20], 6] = 172.        # ~ -11 um; includes a 4-frame run
        self.assertFalse(jump_outliers(rows, Config()).any())
        assert_allclose(np.flatnonzero(jump_outliers(rows, Config(), slow)), [4, 17, 18, 19, 20])

    def test_pushed_bead_is_not_an_outlier(self):
        # like track 191 of the 20x cells movie: a bead at rest, pushed down steadily (1.5 um per frame,
        # 45 um in total, lobes rotating with it) and resting again. Every frame agrees with a
        # constant-velocity motion of its neighbours, so neither the per-track jump test nor the
        # two-depth rule may remove any of it, although most frames are far from the neighbours'
        # median and the resting depth is the majority.
        rng = np.random.default_rng(4)
        model = CalibrationModel(np.linspace(-40, 40, 3201), np.linspace(270, 90, 3201), np.full(3201, 18.))
        f = np.arange(1, 61)
        z = np.clip(28-1.5*(f-5), -17, 28)+rng.normal(0, .1, 60)
        rows = np.zeros((60, NCOL))
        rows[:, 8], rows[:, 2], rows[:, 5] = f, 1083.+rng.normal(0, .1, 60), 1174.+rng.normal(0, .1, 60)
        rows[:, 9] = np.where(f < 35, np.where(f % 2, 200, 218), 191)      # split into interleaved pieces
        rows[:, C['lobeSeparationPixels']] = 18.
        rows[:, 6] = np.interp(z, model.dense_z, model.dense_a) % 180
        self.assertFalse(jump_outliers(rows, Config(), model).any())
        self.assertFalse(alternating_states(rows, Config(), model).any())
        # just before the push, two frames read at the other z branch (above the calibrated range the
        # angle maps to -27 um): only those two are removed, the push next to them stays
        wrong = rows.copy()
        wrong[[2, 3], 6] = np.interp(-27., model.dense_z, model.dense_a) % 180
        self.assertEqual(set(f[jump_outliers(wrong, Config(), model) | alternating_states(wrong, Config(), model)]), {3, 4})
        # the same bead with three single-frame flips of 8 um while resting: only those go
        rows[[45, 50, 55], 6] = np.interp(z[[45, 50, 55]]+8, model.dense_z, model.dense_a) % 180
        assert_allclose(np.flatnonzero(jump_outliers(rows, Config(), model)), [45, 50, 55])

    def test_drift_estimate_does_not_random_walk(self):
        # no true drift, 200 beads with their own depths, z noise 0.68 um, 91 frames, 20% missing:
        # summing median steps wandered ~0.46 um by the last frame; the direct estimate stays ~0.09
        rng = np.random.default_rng(0)
        N, M = 91, 200
        rows = np.zeros((M*N, NCOL))
        rows[:, 8], rows[:, 9] = np.tile(np.arange(1, N+1), M), np.repeat(np.arange(1, M+1), N)
        rows[:, 7] = (rng.normal(0, 5, (M, 1))+rng.normal(0, .68, (M, N))).ravel()
        rows[:, 2] = np.repeat(rng.uniform(0, 1000, M), N)+rng.normal(0, .1, M*N)
        rows = rows[rng.random(len(rows)) > .2]
        drift = estimate_drift(rows, N)
        self.assertLess(np.abs(drift[:, 2]).max(), .35)
        # a real common shift is recovered
        rows[:, 2] += .05*rows[:, 8]
        d = estimate_drift(rows, N)[:, 0]
        assert_allclose(d[-1]-d[0], .05*(N-1), atol=.05)
        self.assertAlmostEqual(float(np.median(d)), 0., places=9)   # referenced to its median

    def test_jump_z_cut_follows_track_noise(self):
        # 4 um single-frame spikes (one at the second frame) on a quiet bead (z noise 0.6 um) are
        # outliers; the same deviation on a noisy 1 ms-like bead (z noise 2.3 um) is not.
        rng = np.random.default_rng(1)
        model = CalibrationModel(np.linspace(-75, 75, 3001), np.linspace(125, 235, 3001), np.full(3001, 16.5))
        def bead(sd, spikes):
            rows = np.zeros((40, NCOL))
            rows[:, 8], rows[:, 9], rows[:, 2], rows[:, 5] = np.arange(1, 41), 1, 100., 50.
            rows[:, C['lobeSeparationPixels']] = 16.5
            z = 4+rng.normal(0, sd, 40)
            z[spikes] = 8.2
            rows[:, 6] = np.interp(z, model.dense_z, model.dense_a)
            return rows
        spikes = [1, 16]
        assert_allclose(np.flatnonzero(jump_outliers(bead(.6, spikes), Config(), model)), spikes)
        self.assertFalse(np.any(jump_outliers(bead(2.3, spikes), Config(), model)[spikes]))

    def test_recovery_requests_interpolate_gap(self):
        rows = np.zeros((4, NCOL))
        rows[:, 8] = [1, 2, 4, 5]
        rows[:, 9] = 1
        for i, dx in enumerate([0, 1, 3, 4]):
            rows[i, [0, 1, 2, 3, 4, 5]] = [11+dx, 27+dx, 19+dx, 21, 21, 21]
        requests = recovery_requests(rows, Config(), 6)
        self.assertEqual(sorted(requests), [3, 6])
        assert_allclose(requests[3][0][1], [[12, 20], [28, 20]])

    def test_axial_inverse_wrap_range_and_separation_disambiguation(self):
        model = CalibrationModel(np.linspace(-10, 10, 1001), np.linspace(150, 210, 1001), np.full(1001, 16.))
        z, status = model.angle_to_z(np.array([0., 15., 100.]))
        assert_allclose(z[:2], [0, 5]); self.assertEqual(status[2], 1)
        # Non-monotonic angle curve: two roots, separated by the separation curve.
        zz = np.linspace(-2, 2, 401)
        model = CalibrationModel(zz, 20-5*np.abs(zz), 16+2*zz)
        z, status = model.angle_to_z(np.array([15., 15.]), np.array([18., 14.1]))
        assert_allclose(z, [1, -1], atol=1e-6); assert_allclose(status, [3, 3])
        z, status = model.angle_to_z(np.array([15.]), np.array([16.]))
        self.assertEqual(status[0], 2); self.assertTrue(np.isnan(z[0]))
        assert_allclose(axial_difference(1., 179.), 2.)
        # z reported only inside z_limits; angles beyond them are 'outside calibration'.
        limited = CalibrationModel(np.linspace(-10, 10, 1001), np.linspace(150, 210, 1001), np.full(1001, 16.), (-5, 5))
        z, status = limited.angle_to_z(np.array([0., 20.]))   # 20 deg = 200 -> z = +6.7, beyond +5
        assert_allclose(z[0], 0); self.assertTrue(np.isnan(z[1])); assert_allclose(status, [0, 1])

    def test_field_separation_map_shifts_expected_separation(self):
        # Saddle map: +3 px * u*v, i.e. lobes 3 px closer in the bottom-left corner (u=-1, v=+1).
        model = MODEL.with_field(np.array([0, 0, 0, 0, 3., 0]), (0., 0.), 2048)
        expected = float(np.interp(90., MODEL.dense_a, MODEL.dense_sep))
        self.assertAlmostEqual(model.separation_residual(90., expected-3, 1., 2048.), 0., places=2)
        self.assertAlmostEqual(model.separation_residual(90., expected, 1024.5, 1024.5), 0., places=2)
        self.assertAlmostEqual(MODEL.separation_residual(90., expected-3, 1., 2048.), -3., places=2)
        # A compact corner bead is only paired when the field map is used.
        image = render([(60.3, 1990.2, 90., 150., expected-3)], shape=(2048, 2048))
        self.assertEqual(len(localize_image(image, Config(sep_gate_px=2.), MODEL)[0]), 0)
        self.assertTrue(found(localize_image(image, Config(sep_gate_px=2.), model)[0], 60.3, 1990.2, .2))

    def test_z_zero_at_angle_crossing(self):
        z = np.arange(-10., 11.)
        unwrapped = 170+2*z                       # crosses 180 deg (horizontal) at z = +5
        self.assertAlmostEqual(_angle_crossing(z, unwrapped, 0.), 5.)
        self.assertAlmostEqual(_angle_crossing(z, unwrapped, 175.), 2.5)
        self.assertAlmostEqual(_angle_crossing(z, unwrapped, 160.), -5.)   # 160 deg = 340 - 180: nearest crossing

    def test_drift_is_median_shake(self):
        rng = np.random.default_rng(1)
        shake = np.array([[0, 0, 0], [1.5, -.5, .2], [0.5, 1., -.3]])
        rows = []
        for t in range(1, 21):
            x, y = rng.uniform(0, 500, 2)
            local = 3. if t == 1 else 0.  # one locally deforming bead does not bias the median
            for f in range(3):
                r = np.zeros(NCOL)
                r[[2, 5, 7, 8, 9]] = [x+shake[f, 0]+local*f, y+shake[f, 1], 2+shake[f, 2], f+1, t]
                rows.append(r)
        # the shake relative to its median over the frames (a constant reference)
        assert_allclose(estimate_drift(np.array(rows), 3), shake-np.median(shake, axis=0), atol=1e-9)

    def test_lateral_model_recovers_defocus_magnification(self):
        rng = np.random.default_rng(2)
        coef = np.array([[-.08, -.3, 0, 0, 0, 0], [.07, 0, -.3, 0, 0, 0]])
        z_grid = np.arange(-29., 30.)
        beads = []
        for _ in range(40):
            x, y = rng.uniform(1, 2048, 2)
            b = np.zeros((59, NCOL))
            b[:, 8] = np.arange(1, 60)
            b[:, 7] = z_grid
            b[:, 2], b[:, 5] = x, y
            shift = _lateral_design(z_grid, np.full(59, x), np.full(59, y))@coef.T
            b[:, 2] += shift[:, 0]+rng.normal(0, .05, 59)
            b[:, 5] += shift[:, 1]+rng.normal(0, .05, 59)
            beads.append(b)
        fitted, sd = fit_lateral_model(beads, z_grid)
        assert_allclose(fitted, coef, atol=.01)
        cx, cy = lateral_correction(beads[0], fitted)
        self.assertLess(np.ptp(cx), .5); self.assertLess(np.ptp(cy), .5)

    def test_transient_split_and_deduplicate(self):
        rows = np.zeros((6, NCOL))
        rows[:, 8] = [1, 2, 3, 1, 1, 1]
        rows[:, 9] = [5, 5, 5, 9, 11, 11]
        rows[:, 2] = [10, 10, 10, 50, 90, 91]
        rows[:, C['minLobeSNR']] = [9, 9, 9, 9, 5, 8]
        rows[:, C['templateScore']] = np.nan
        kept, rejected, reasons = split_transient(rows, Config())
        self.assertEqual(len(kept), 3); self.assertEqual(set(kept[:, 9]), {1})
        assert_allclose(reasons, [1, 1, 1])
        rows[:, C['templateScore']] = [.3, .3, .3, .9, .9, .9]   # faint and PSF-unlike: low quality
        kept, rejected, reasons = split_transient(rows, Config())
        self.assertEqual(len(kept), 3)          # SNR 9 is typical for this movie: kept (relative cut)
        kept, rejected, reasons = split_transient(rows, Config(reject_track_snr_fraction=2.))   # cut = min(15, 2 x 9)
        self.assertEqual(len(kept), 0); assert_allclose(sorted(reasons), [1, 1, 1, 2, 2, 2])
        dedup = deduplicate(rows)
        self.assertEqual(len(dedup), 5)
        self.assertIn(8, dedup[:, C['minLobeSNR']])


class LabZTests(unittest.TestCase):
    def test_lab_z_is_the_mirror_of_stack_z(self):
        # a calibration in stack coordinates and its lab-z version (z_lab = -z_stack)
        from pipeline import to_lab_z
        rng = np.random.default_rng(3)
        zs = np.arange(-40., 41.)
        dense = np.linspace(-40, 40, 8001)
        stack = dict(z=zs, dense_z=dense, dense_a=20+1.1*dense+.004*dense**2, dense_sep=17+(dense/30)**2,
                     z_limits=np.array([-30., 35.]), lateral_coef=rng.normal(0, .05, (2, 6)),
                     sample_tilt_um_per_px=np.array([.012, -.004]))
        lab = to_lab_z(stack)
        self.assertTrue(np.all(np.diff(lab['dense_z']) > 0))                     # curves stay ascending
        assert_allclose(lab['z_limits'], [-35., 30.])
        assert_allclose(lab['z'], -zs)
        assert_allclose(lab['sample_tilt_um_per_px'], [-.012, .004])
        self.assertIs(to_lab_z(lab), lab)                                           # idempotent
        m_stack = CalibrationModel(stack['dense_z'], stack['dense_a'], stack['dense_sep'], stack['z_limits'])
        m_lab = CalibrationModel(lab['dense_z'], lab['dense_a'], lab['dense_sep'], lab['z_limits'])
        angles = rng.uniform(0, 180, 200)
        seps = 17+rng.uniform(0, 2, 200)
        z_s, st_s = m_stack.angle_to_z(angles, seps)
        z_l, st_l = m_lab.angle_to_z(angles, seps)
        np.testing.assert_array_equal(st_s, st_l)
        assert_allclose(z_l, -z_s, atol=1e-9)
        # frame caches are keyed on the stack-coordinate fingerprint: the same calibration gives the same key
        stack_fp = [float(dense[0]), float(dense[-1]), float(np.sum(stack['dense_a'])), float(np.sum(stack['dense_sep'])), -30., 35.]
        assert_allclose(m_lab.fingerprint(), stack_fp)
        # the lateral correction of a localization is the same in both conventions
        rows = np.zeros((50, NCOL))
        rows[:, 2], rows[:, 5] = rng.uniform(1, 2048, 50), rng.uniform(1, 2048, 50)
        rows[:, 7] = rng.uniform(-30, 30, 50)
        flipped = rows.copy(); flipped[:, 7] *= -1
        assert_allclose(lateral_correction(flipped, lab['lateral_coef']), lateral_correction(rows, stack['lateral_coef']), atol=1e-9)


if __name__ == '__main__':
    unittest.main()
