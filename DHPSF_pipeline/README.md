# DH-Tracker: DH-PSF bead tracking

Finds fluorescent beads imaged through a double-helix point spread function (DH-PSF), measures their 3D position in every frame (z from the rotation of the two lobes, using a calibration z-scan), links them into tracks, analyses how each bead moves, and shows everything in an interactive browser app (the explorer), including deformation maps.

## What is in this folder

| | |
|---|---|
| `Start DH-Tracker.bat` (one level up) | double-click to use the program |
| `analyze.py` | runs an analysis: calibration + movie(s) → results (the explorer calls it too) |
| `pipeline.py` | the localization and tracking engine |
| `track_analysis.py` | motion analysis of the tracks (runs automatically after an analysis) |
| `explorer/` | the browser app |
| `user_settings.py` | your folders: movies (where the file picker opens) and **results** (one folder per analysis, `<name>_<date-time>/`). Chosen once at setup, remembered in your user profile (`%APPDATA%\DH-Tracker\settings.json`), changeable in the app (☰ › Folders). Without a choice, results go to `runs/` here |
| `calibrations/` | automatic cache of fitted calibrations (safe to delete; rebuilt when needed) |
| `camera_calibration.py`, `export_tables.m` | optional: sCMOS camera maps; native MATLAB tables |
| `test_*.py` | automated tests |
| `tools/` | developer scripts: validation, benchmarks, timing, converters for older result folders |
| `requirements.txt` | the exact Python packages |

## Setting up on a new computer (Windows, once)

**Easiest:** download `Install DH-Tracker.bat` from https://github.com/WeissLab/DH-Tracker (open it, **Download raw file**) and double-click it. It asks for an install folder (default `Documents\DH-Tracker`), downloads the newest version there, offers Miniconda (winget) if there is no conda, and runs steps 3–4 below. The installed copy remembers its version (`version.txt`); `Start DH-Tracker.bat` then checks GitHub (at most 5 s, skipped offline) and offers to update, replacing only the program files (results, calibration caches and settings are kept; the setup runs again only when `requirements.txt` changed). The logic is in `install.ps1`.

By hand:

1. Get the project folder (the folder holding `Start DH-Tracker.bat` and `DHPSF_pipeline/`): on https://github.com/WeissLab/DH-Tracker, **Code › Download ZIP**, then unzip it anywhere (or copy the folder from another computer). The TIFF files can be anywhere on the computer (any drive, including network drives); you pick them in the app and they are only read, never changed or copied.
2. Install Miniconda (free, from the Anaconda website; the default options are fine).
3. Double-click **`Setup DH-Tracker.bat`** in the project folder. It finds Miniconda, creates the environment `dh-tracker` with the exact package versions in `requirements.txt` (about 1 GB, a few minutes, needs internet), and runs the 68 tests. Two folder dialogs then ask where your movies are and where results should be saved (Cancel for the second: `Documents\DH-Tracker results`). It ends with "Ready" (details in `setup_log.txt`). Running it again only checks and repairs an existing setup.

   The same by hand, in an **Anaconda Prompt** from the project folder:
   ```bat
   conda create -n dh-tracker -c conda-forge --override-channels python=3.11 pip
   conda activate dh-tracker
   pip install -r DHPSF_pipeline\requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
   cd DHPSF_pipeline
   python -m unittest test_pipeline test_track_analysis explorer.test_explorer
   ```
4. Optional: MATLAB, only to also write native MATLAB tables (`*_localizations.mat`); without it the CSV and MAT outputs are written as usual.

After that, double-click `Start DH-Tracker.bat` whenever you want to use it (below). **`DH-Tracker guide.mp4`** (next to the launcher, 3 minutes, captioned) shows a whole session: a new analysis of a 20× movie, the results, and the exports.

## Quick start: one calibration + one movie

Double-click **`Start DH-Tracker.bat`** in the project folder. It finds Python, starts the explorer and opens it in your browser; if the explorer is already running, it just opens it. Keep its window open while you work. Then click **New analysis**, pick the calibration z-stack and the movie(s), check the magnification, and press Start.

Or without the browser, from an Anaconda Prompt in the project folder:

```bat
conda activate dh-tracker
python DHPSF_pipeline\analyze.py CALIBRATION.tif MOVIE.tif
```

`analyze.py` fills in everything else (see its `--help`; `--inspect FILE` shows what it infers):
- **Pixel size:** from "10x" (0.63 µm) or "20x" (0.325 µm) in the file or folder name, unless `--pixel-size` is given.
- **z step:** the stage speed in the calibration's file name (e.g. `25ums`) × the frame interval in its Micro-Manager metadata (40 ms → 1.0 µm). If the name has no stage speed, give it with `--z-step` (New analysis asks for it); the run never assumes a value, because a wrong z step scales every z.
- **z = 0:** where the lobes are horizontal or vertical, whichever the stack's middle plane is closer to.
- **z sign:** z is height, positive upwards, toward the indenter (which comes from above), so an indentation moves beads to negative z: on the 10× 5 ms movie, Δz ≈ −16 to −23 µm under the indenter. The calibration stack index runs the other way (down into the sample), so the calibration is fitted in stack coordinates and converted once (`pipeline.to_lab_z`). `run_info.json`, `summary.json` and the payload metadata record `z_convention: "up"`.
  - Result folders written before 27 Sep 2026 reported z along the stack, with the opposite sign. `python tools/convert_z_up.py FOLDER` converts one in place (every z column, the drift, the motion outputs, the calibration, the summaries and the exports; a copy of each changed file is kept in `FOLDER/_z_stack_backup/`). The explorer also flips such folders when loading them.
- **z range:** ±75 µm for 10×, the full calibrated range otherwise (`--z-range`).
- **Crop position and sensor size:** from the Micro-Manager metadata embedded in the TIFF (or a `*_metadata.txt` next to it), otherwise centred on a 2048-px sensor.
- **Objectives must match:** if the names of the calibration and a movie suggest different objectives, the run stops with an explanation.
- **Dim movies** (median lobe SNR < 6, e.g. 1 ms) automatically get the dim-data settings. Each movie is checked on its own, so bright and dim movies can share a run.
- **The calibration is checked** up front (`--check-calibration FILE`): enough beads per plane, and lobes that rotate between planes. Dim stacks and movies chosen by mistake are rejected with advice.
- **MATLAB tables** are an optional extra: if MATLAB fails, the run still completes with a warning.
- Dim calibration stacks: Use a longer exposure of the same optics; the angle–z relation doesn't depend on exposure.
- **Calibration cache:** the calibration fit is cached per calibration file in `DHPSF_pipeline/calibrations/`, so later movies with the same calibration start straight away.
- **Output:** results go to `<movie>_<date-time>/` in the results folder (chosen at setup; ☰ › Folders in the app; without a choice `DHPSF_pipeline/runs/`), with live progress in `status.json`. The TIFF files are not copied; the run records their paths (`run_info.json`), and the explorer reads the movie frames from there, so keep them where they were.

`--open` launches the explorer on the result. A first run with a new calibration takes about 5–10 minutes (250-plane stack); reruns with the same calibration take a few minutes.

## Full control

`analyze.py` sets everything up and calls `pipeline.py`. To run the engine directly, from the `DHPSF_pipeline` folder with the environment active:

```bat
python pipeline.py --calibration CAL.tif --movie NAME=MOVIE.tif --pixel-size 0.63 --z-step 1 --output runs\my_run --workers 16
python -m unittest test_pipeline test_track_analysis explorer.test_explorer
python tools\validate_results.py --results runs\my_run
python tools\benchmark_injection.py --results runs\my_run      # recall benchmark
python explorer\app.py                                          # the explorer without the launcher
```

Without `--output`, `pipeline.py` writes to a new folder in the results folder. Dependencies (exact versions in `requirements.txt`): NumPy, SciPy, tifffile, networkx, matplotlib, Pillow, PyTorch (CPU build);
MATLAB for the native table export (`--no-matlab` skips it; then run
`export_tables('PATH/TO/RUN')` in MATLAB). Inputs are
read-only. Per-frame fit caches let interrupted runs resume; when the input, the per-frame
fitting settings or the calibration change, the stale cache is refitted automatically.
`--config settings.json` overrides any `Config` field, e.g. `{"tracker": "kalman",
"min_track_length": 5}`; `--movie-config` does the same for the movies only (e.g. dim-data
settings) and `--calibration-cache DIR` shares a calibration's fit cache between runs.

Every frame is fitted independently. No coordinate is averaged over time. Time is
used only to (a) link localizations into tracks, (b) predict where a missed bead
should be and refit it there, and (c) reject tracks too short to be real.

## Running a new data set (new movie + new calibration)

You need:
1. **A calibration z-stack.** Beads imaged with the same optics (same objective, phase mask and alignment), one plane per z step.
2. **The movie(s)**, as TIFF stacks. OME/Micro-Manager files that cannot be memory-mapped are read page by page.
3. **Four numbers:**
   - pixel size in µm;
   - z step between calibration planes;
   - camera width in pixels (sensor size);
   - for cropped images, where the crop sits on the sensor. The default assumes a crop centred on the sensor. For Micro-Manager this is the `ROI` entry in `*_metadata.txt`.

The easiest way is New analysis in the explorer, or `analyze.py`, which work all of this out. With `pipeline.py` directly, use a new `--output` folder for every calibration + movie combination. Example: 10× movie with the matching 10 ms calibration (the calibration is a centred 1024² crop; 25 µm/s at 0.04 s per frame = 1 µm per plane):

```powershell
python DHPSF_pipeline/pipeline.py `
    --output DHPSF_pipeline/runs/10x_10ms `
    --calibration "D:/data/10X/3. Inter-Lobe Calibration/10x_0.04s_25ums_10ms_1_MMStack_Pos0.ome.tif" `
    --movie indent_10ms="D:/data/10X/4. Indentation/10x_20um-indent_100ums_10ms_1_cropped.tif" `
    --pixel-size 0.63 --z-step 1 --workers 16
```

`--movie` can be repeated; without `NAME=` the file name is used. Use
`--offset calibration=512,512` or `--offset NAME=X,Y` when a crop is not centred.
Before the first run, `tools/separation_profile.py MOVIE.tif --out profile.png` finds all spots on a few frames and histograms the distances between similarly bright spots (a pair-correlation / Ripley-type profile). It reports the characteristic lobe separation, overall and per field tile, without any calibration. Examples:
- 20× collagen: 16.8 px overall, 15.8–17.8 px across tiles.
- 10× 5 ms: a uniform 16.8 px.

Use it to check that the 9–27 px default range fits new optics. Then:

1. **Check the calibration:** `calibration.png`. The angle and separation curves should be smooth, with many supporting beads per plane. The printed supported z-range tells you where z is defined.
2. **Check the field map:** `field_separation.png` plus `summary.json` → `field_separation`. It shows whether the lobe separation varies across the movie's field (see below).
3. **Check the fits:** `experimental_fits.png` (middle frame), then run `tools/validate_results.py --results <output>` and `tools/benchmark_injection.py --results <output>`.
4. **Explore:** `explorer/app.py --results <output>`. The explorer reads the movies and pixel size from `<output>/run_info.json`.

Change settings with `--config settings.json` only if the new optics fall outside the defaults:
- lobe separation 9–27 px;
- lobe σ 0.8–5 px;
- bead motion under 8 px per frame (`link_distance_px`).

The 10× lobes measured about 16–23 px apart, so the defaults fit. z = 0 is where the lobes are horizontal (or vertical; whichever the stack's middle plane is closer to), unless `--z-zero middle` or an angle is given.

## Field-dependent lobe separation (per-movie self-calibration)

The 20× movies show a saddle-shaped variation of the lobe separation across the field. Lobes are about 2–3 px wider apart in the top-left and bottom-right corners and about 2–3 px closer in the bottom-left and top-right. Both movies show the same pattern (u·v coefficient ≈ 2.9 px), but the calibration stack does not. A calibration-only separation check therefore rejected compact corner beads (e.g. 11.7 px where the calibration never goes below 16.3 px).

Each movie now learns its own map:
1. Beads are localized on 12 frames with geometric gates only.
2. Only isolated beads (no other bead within 40 px) with SNR ≥ 10 are kept.
3. Their separation is compared with the calibrated separation at their angle.
4. A quadratic in field position is fitted, with outliers rejected.

Pairing, recovery and z disambiguation then expect `calibrated separation + field offset`. Each localization records the offset used (`fieldSeparationOffset`). `--no-field-correction` turns this off.

Because the calibration stack does not contain this aberration, the angle-to-z relation in regions with a large offset has not been calibrated. Treat z there, and especially its absolute value, with more caution than in the centre. A calibration acquired with the same optical alignment as the movies would remove this caveat.

## Design choices

Problems found on the real movies and with the injection benchmark (below), and how the pipeline handles them:

| Problem | What the pipeline does |
|---|---|
| A tracker whose non-link cost is a high percentile of the frame's link costs breaks stationary beads into fragments: any bead moving a little more than most is cut off. In a first version this caused about 70% of the missing bead-frames. | Non-link cost fixed at the squared gate, so every gated link beats birth plus death. A small rotation penalty (15° costs as much as 1 px) helps at crossings. Gap closing up to 6 frames. |
| Rejecting beads whose fit window touches the image edge loses them (19% recall near the edge). | Fit windows are clipped to the image; only lobe centres must lie inside it. |
| Greedy pairing of detected peaks fixes pairs before fitting and mis-pairs overlapping beads. | All peaks in a crowded region are fitted first as free Gaussians, then paired by a global maximum-weight matching. Pairs are scored against the calibrated separation-vs-angle curve, amplitude ratio and σ similarity. |
| Two beads sharing a lobe (three spots, bright middle) would yield only one bead. | Shared-lobe pairing: a leftover lobe may take an already-paired lobe as partner when the geometry fits the calibration. The shared spot must be brighter than its partner by roughly the leftover's amplitude. Flagged `sharedLobe=1`. |
| A fixed absolute detection threshold suits either bright or dim movies, not both. | Noise-adaptive threshold: strong peaks at 6 robust DoG-noise SDs. Weak peaks at 55% of that are allowed only as partners of strong peaks. |

## Pipeline

1. **Detection:**
   - Candidate lobes are found on a Gaussian running average of neighbouring frames (σ = 1 frame, ±2 frames), or of neighbouring z-planes for the calibration stack. Beads move slowly, so the average raises lobe contrast about 1.9× without blurring positions beyond the ±3 px seed tolerance. Fits are always made on the single raw frame.
   - DoG (σ 1.2 − σ 12), 5×5 maxima. Strong peaks at 6 robust noise SDs; weak partner peaks at 55% of that.
   - Where the image is clearly noisier than the frame as a whole, the threshold rises in proportion (`local_noise_tile_px` = 32, `local_noise_min_ratio` = 2): per 32-px tile, the noise is the robust SD (MAD) of vertical second differences, which cancel the smooth lobes and background. Tiles at least 2× noisier than the frame keep that ratio, all others 1, and the map is interpolated to every pixel. Without it, a noisy region gave hundreds of noise peaks that chained into fit windows of 40–90 Gaussians (on noisy planted test beads: 427–489 s per frame, now 23–30 s, with more beads found). Around bright beads and cells the ratio stays below 2, so real frames are unaffected (about 1 bead in 1000 changed).
2. **Grouping:** peaks joined by plausible pair edges (DoG ratio ≤ 8) form groups. In movies, an edge must fit the calibrated lobe separation at that angle, plus the movie's field map, within `sep_gate_px` + `prune_slack_px` = 6.5 px (`calibration_pruning`). The calibration pass, which has no curve yet, accepts any 7–27 px. On the 10× movies this kept the same detections and made localization 1.2–1.4× faster. Groups whose windows overlap are merged. Windows with more than 24 Gaussians are split recursively into overlapping sub-windows, and each keeps only pairs whose midpoint lies in its own core.
3. **Joint fit:** all peaks in a window are fitted as free symmetric Gaussians (amplitude, centre ±3 px, σ 0.8–5 px) plus an affine background, by bounded least squares with an analytic Jacobian. Windows with more than 40 parameters use a sparse Jacobian (each Gaussian only within 4σmax of its seed) and the iterative LSMR step. This gives identical results and is 2–2.5× faster on crowded or noisy planes; the dense SVD step was the bottleneck.
   - **Merged lobes:** a lobe more than 1.35× wider than the window's other lobes may be two lobes of neighbouring beads. It is replaced by two Gaussians along its long axis and only its neighbourhood is refitted (the lobes within 15 px; the rest of the window held fixed), at most 6 attempts per window; the split is kept if it pairs more lobes or lowers the total pairing cost. Refitting the whole window per attempt gave the same results but took up to a third of the run time on 20× frames.
4. **Pairing:** maximum-weight matching of usable lobes (not at a bound, amplitude ≥ 4 × camera noise SD). Cost = (log ratio / 0.5)² + (Δσ / 0.6)² + (separation residual / 1 px)². The separation residual is measured against the calibrated separation at that angle plus the movie's field offset. Hard gates: separation 9–27 px, ratio ≤ 4, |separation residual| ≤ 3.5 px, cost ≤ 25.
   - **Dim data** (the automatic dim-data settings, e.g. 1 ms):
     - Pairs are accepted on their matched-filter SNR, √(π(A₁²σ₁² + A₂²σ₂²)) / noise ≥ `min_pair_snr` = 10, which uses the light of both lobes. Each lobe then needs only 1 × noise.
     - The separation prior's SD follows the lobes' localization precision, √(0.15² + σ₁² + σ₂²) with σᵢ ≈ 0.8·noise/Aᵢ, and the gate is 6 of those SDs, between 1.2 and 3.5 px (`sep_sd_mode = 'precision'`).
     - On the 1 ms movie this raised single-frame recall from 95.1% to 97.8% with fewer false positives (31 against 39 on 8 frames). On bright data it brought no gain, so it is off there.
5. **Phantom and ring rejection:**
   - A pair reusing a lobe of a brighter accepted pair (within 3 px) is dropped. The exception is a validated shared-lobe pair whose own lobe reaches 15 × noise; a pair reusing both lobes is always dropped. Otherwise a second Gaussian on a real lobe, paired with a ring arc, gives a persistent phantom next to the true bead.
   - A faint pair (≤ 20% amplitude) with one lobe within 10 px of each lobe of a brighter pair is dropped. These are the side-lobe arcs of the DH-PSF.
6. **Tracking:** `tracker` = `lap` (default: two-stage LAP, 8 px adjacent gate, 12 px gap gate, ≤ 6-frame gaps), `kalman` (constant-velocity Kalman filter per track with LAP assignment to predictions; coasts through gaps) or `nearest` (greedy baseline).
   - **z-aware linking:** a link must also agree in depth. The z change must be within the larger of `link_z_um` = 8 µm and `link_z_sigma` = 4 combined z error bars, plus `link_z_rate_um` = 1 µm per skipped frame. Gap closing compares the median z of the last and first 5 frames of the two segments. It also re-joins adjacent-frame breaks, so one noisy z value cannot split a bead. This stops xy-only linking from stitching one bead's lobe to a neighbour's lobe at a very different depth. It also stops it from merging two beads stacked in z at the same xy (whose fits alternate between frames) into one track that jumps back and forth.
7. **Track-guided recovery:** for every track of ≥ 3 frames and every frame within 8 frames of it where it is missing, the lobe positions are predicted:
   - inside a gap: linear interpolation between the bracketing observations;
   - otherwise: the nearest observation plus the median motion of neighbouring beads within 200 px.
   The bead is then refitted at the prediction (±4 px) jointly with its neighbours. It is accepted (`recovered=1`) only if:
   - both lobes reach 3 × noise **and** 35% of the track's median lobe amplitude;
   - its angle is within 15° of the predicted angle;
   - it passes the same pairing gates;
   - it does not duplicate an existing bead.
   The amplitude and angle checks stop tracks from being extended into empty background or haze once a bead is really gone. Rounds repeat until nothing new is found, followed by re-tracking.
8. **Track rejection** (moved to `*_rejected.csv` with `rejectReason`):
   - `1` = transient: fewer than 3 frames.
   - `2` = low quality: median lobe SNR below min(15, 0.25 × the median track SNR of the movie) **and** median template score < 0.6 over the track's detected frames. Making the cut relative keeps long, real tracks in dim (1 ms) movies. The rejected tracks are diffuse haze and noise pairs; tightly clustered real beads have a low template score but high SNR and are kept.
   - `3` = satellite: a faint track that stays next to a much brighter bead with a similar angle (a persistent side-lobe phantom).
   - `4` = removed during recovery: a jump outlier. A localization is one if it lies more than 3 px or 20° away from the median of its track within ±5 frames (before and after, so a track's first frames are judged too). In z, the limit is max(2.5 µm, 4 × the track's own frame-to-frame z noise), for example about 2.5 µm for a 5 ms bead (noise 0.6 µm) and about 9 µm for a 1 ms bead. These are the "several microns and back" excursions, caused by a crossed pairing in one frame.
     Also removed: the **minority state of a bead whose fit alternates between two depths**. This happens where a lobe overlaps another bead's lobe, or in chains of lobes where two pairings fit. The reference is everything localized at the same place (centre within 3 px) in the surrounding ±5 frames, in any track. Its dominant z state is the densest cluster, not the median, which would fall between the two states. A localization more than max(5 µm, 4 × the movie's z noise) from it, and in a smaller cluster, is removed.
     The same at the segment scale: a whole track that sits at one place between a predecessor and a successor (within 12 frames, ends within 3 px) that agree with each other in z, while it differs from both by more than that cut, is removed.
     The frame is then refitted from the track prediction; a fit identical to the removed one is never re-added.
   - `5` = ghost: a track built from other beads' light. A lobe is "borrowed" if another track's lobe lies within 8 px of it (±8 frames). A track is a ghost if both lobes are borrowed in ≥ 60% of its frames **and** either:
     - it is at most half as bright as the lenders (a side-lobe fit), or
     - its z is more than 15 µm from the lenders' z and farther than theirs from the local median z of the tracks within 150 px (a crossed pair of two neighbours' lobes).
     Ghost removal runs in every recovery round, so the real beads can then be refitted in those frames.
9. **z:** the calibration slide is usually slightly tilted, so its beads sit at different depths. At 10× the tilt was 1.1° along x, putting the beads about 12 µm apart in depth across the image. Each calibration bead's depth offset is estimated from how its angle departs from the curve at different rotation rates, and a plane is fitted across the image. The beads are moved to the image centre's depth before the curve is built; this cut the per-plane angle scatter from 3.9° to 1.5°. The tilt concerns only the calibration slide and is not applied to movies (`summary.json` → `calibration_sample_tilt`). Angle → z by inverting the calibration curve (a penalized smoothing spline through the per-plane medians, weighted by their standard errors; smoothness by generalized cross-validation — interpolating the noisy medians exactly would add up to 0.3 µm of jagged z error). When an angle has two inverses, the lobe separation picks one if it is decisive (`zStatus=3`); otherwise z is NaN (`zStatus=2`).
10. **Lateral correction** (`xCorrected`, `yCorrected`): the calibration shows the lobe midpoint shifting sideways with z. That shift is mainly a defocus-dependent magnification of about 0.30 px per µm of z at the field edge, plus a uniform tilt of about 0.07 px/µm. (The terms linear in z changed sign with the z convention; the correction itself is unchanged.) A bead moving 10 µm axially near the edge would otherwise appear to move about 3 px (1 µm) radially. The model is fitted on calibration beads with per-bead offsets, robustly (Huber weights) and with the stage's random plane-to-plane wobble removed; see `summary.json` → `lateral_model`. Raw `xMean`/`yMean` are unchanged.
11. **Stabilization** (`*_drift.csv`, `xStabilized`, `yStabilized`, `zStabilized`): whole-field shake per frame, relative to its median over the movie (so stabilized positions sit at each bead's typical position, not at frame 1, which is itself one sample of the shake; `tools/rereference_drift.py` converts folders written with the old frame-1 reference), from a robust two-way fit position(bead, frame) = bead offset + frame drift (median polish, on corrected coordinates). Every frame's drift is estimated directly from all beads; summing frame-to-frame medians instead made a random walk (≈0.46 µm of pure estimator noise in z after 91 frames). Local deformation affecting a minority of beads is ignored by the median. Motion shared by most of the field would be removed too, so treat it as an optional view. XY is corrected frame by frame, because the stage really shakes. Focus drifts smoothly, so the z drift is Gaussian-smoothed over time (`drift_z_smooth_frames` = 3 frames σ). Its remaining frame-to-frame changes are measurement noise. `dz` in `*_drift.csv` is the smoothed curve, and `dzRaw` the per-frame one.
12. **Template score** (`templateScore`): zero-normalized cross-correlation of the 41×41 patch with the calibration PSF (median of isolated calibration beads) at that z. 1 means identical; neighbours inside the patch lower it. Use it to filter doubtful localizations, e.g. diffuse autofluorescence in the cells movie.

## Track motion analysis (aTrack / ExaTrack models)

After tracking, `track_analysis.py` runs automatically; skip it with `--no-motion`, or run it on older results with `python track_analysis.py RESULTS_DIR` or the explorer's Motion analysis section. It follows the motion models of:
- **aTrack**: Simon *et al.*, "Detecting directed motion and confinement in single-particle trajectories using hidden variables", *eLife* 13, RP99347 (2026), doi:10.7554/eLife.99347.
- **ExaTrack**: Simon, Wiggins & Weiss, "Analyzing single-molecule dynamics with both complex types of motion and complex transition kinetics: Benchmarking of ExaTrack", *bioRxiv* (2026), doi:10.64898/2026.01.22.700663.

The models are implemented here independently as Kalman filters, which give the exact likelihood of these linear-Gaussian models; this is equivalent to aTrack's recurrences. Every localization uses its own reported error, so x, y and z keep their very different errors. Missing frames are predicted over.

1. **Coordinates:** stabilized µm (whole-field shake removed; `--coords` changes this).
2. **Error scale:** the reported errors are first rescaled per axis from the data. For a random walk seen through white noise, consecutive steps are anticorrelated by −σ² (median over tracks). Otherwise errors reported too small look like confinement. The 10× factors are 1.21 / 1.13 / 0.97 (x / y / z) at 5 ms.
3. **Per track (aTrack):** Brownian (step d), confined (a pull *l* toward a well centre that itself diffuses, q) and directed motion (a velocity that changes slowly, q) are each fitted by maximum likelihood. The likelihood ratio ρ = L_Brownian / L_alternative < 0.05 (`--alpha`) calls a track confined or directed, as in aTrack.
   - **Localization error per track:** each model also fits a factor e on the track's reported errors (`FIT_ERROR`). Without it, a track that is simply noisier than reported looked confined (a tight, fast cage mimics white noise): on the 10× 5 ms movie, 35 of 36 "confined" tracks were such tracks. In simulations the factor costs almost no sensitivity to real confinement.
   - Each track's direction of motion (`u_directed`, the principal axis of its smoothed velocity) is kept for the direction-coupled models below.
4. **Over time**, one of:
   - **Moving / still** (default; ExaTrack-style two-state model): a population model switches per frame between **still** and **moving** with first-order transition probabilities. Both states use the track's own random step and error factor, so they differ only by a persistent velocity. One Gaussian per state is kept and merged after each step (an interacting-multiple-model filter; ExaTrack merges over a longer history). A backward pass gives each localization's P(moving).
   - **Ordered stages** (`--stages indentation`, or the explorer's "model over time" menu), for experiments with a known sequence: still → indent → hold → retract → still (or any list of `still`/`hold` and `indent`/`retract`/`move` stages). Every bead starts in the first stage and can only move forward; stages it does not show are skipped (the skip probabilities are fitted). Moving stages move along each bead's own 3D direction with a speed that changes smoothly: x, y and z are filtered jointly, so one speed is informed by all three axes. Each bead gets per-frame stage probabilities and the frame it entered each stage, so onset times can differ with distance from the indentation. On the 10× 5 ms movie, 118 of 213 beads entered the indent stage, among them all 32 within 150 µm of the indentation point, at a median frame of 22 against 36 farther out (onset against distance: r = 0.60). On the noisy 1 ms movie (median lobe SNR 3) the stage fit is not reliable.
     - **Directions:** the indenter comes from above, so in the `indent` stage beads move down (toward −z) and in `retract` up. Each bead's direction is oriented downward and it enters these stages at its own speed (from its directed fit) times a fitted factor, because beads under the indenter move tens of times faster than those far away. Without directions, which directed stage took the fast indentation was arbitrary; with one shared speed, the fastest beads fitted no stage.
     - **Bounds:** the still stages' steps are capped (smoothly) at the median step of the beads classified Brownian, and the directed stages' velocity change at the 90th percentile of the directed beads'. An unbounded step or velocity change makes its stage a catch-all: the last still stage fitted a 0.28 µm/frame step on the 20× cells movie and reported beads pushed by the indenter as still. Each stage keeps its own step: stages with identical models are interchangeable, and beads switched between them arbitrarily.
     - **Other moving:** from any stage a bead can switch to a side state of directed motion unrelated to the sequence (e.g. pushed by a cell) and back to the same stage, so the stage is remembered. Its velocity is a free 3D vector (not along the bead's indentation direction), drawn anew on each entry and changing smoothly; the side states share one set of parameters. It is labelled stage K+1 (`other moving`) where it is more probable than not, and it counts as moving.
     - Fitting: the stages alone first, then with the side states, starting from that fit; L-BFGS is restarted while it still improves the likelihood (it can stop far from the optimum after a poor line search). About 1 minute for the 20× cells movie (192 tracks), 3–4 minutes for the collagen one (333 tracks).
     - Unit tests: with the side states switched off the extended filter reproduces the plain stage model exactly; in simulations (indentation downward), beads moving again in another direction during the hold are labelled other moving (> 80% of those frames), still beads never enter the indent stage, indent onsets are found within 2 frames (median), and a bead that moves twice counts as moving during both moves (the first may then be called indent or other moving).
   - **General multi-state model** (`MultiStateModel` in Python, not yet in the command line): any set of diffusive, directed and confined states with gamma-distributed lifetimes (ExaTrack), localization error fixed, per track or per state, step size per state or per track, optional direction coupling (`u_track`) and a log-normal prior on the mean lifetimes. Its filter is an exact reduced form of the full (position, velocity, well centre) model (unit test), 9× faster per evaluation; fits take 10–60 s per movie instead of about 10 minutes.
5. **Fitting (fast on any CPU; no GPU needed):**
   - Per track: all three models are written as one linear model with per-track parameters, so every track runs through one batched Kalman filter. A coarse grid over each model's parameters (forward passes only) finds the best basins; a damped Newton polish from three diverse grid points (gradient and finite-difference Hessian for every track in one pass, each track its own step length) converges.
   - The likelihoods have long ridges and several optima. The confined model is parameterized by its cage size (stationary SD) to straighten the main ridge, and the well centre may diffuse at most as fast as the bead (q < d): without that bound, the limit l → 0, q → ∞ turns the confined model into the directed one, and a fully optimized confined model absorbs every directed track.
   - On the 10× 5 ms movie this reaches the best likelihood found by any method for every track (within 0.1) in 3.5 s (12 cores; 6.8 s on one). The previous 2-start Adam fit took 65 s and missed the best confined optimum for about 40% of tracks.
   - The switching model's four shared parameters are fitted by L-BFGS (1.7 s, was 48 s with Adam). Its filter processes both states in one tensor and mixes them with closed-form moments (identical to the previous implementation to 1e-9).
   - One CPU thread is used for small batches (faster than many: thread overhead exceeds the work), all cores only for the coarse grids.
   - A whole motion analysis takes 6–8 s per 10× movie (about 2 minutes before).
6. **Validation on simulated tracks** (the shape of the 10× data):
   - Still beads were never called directed or confined; 58 of 60 beads that moved for about 22 frames were called directed; 23 of 30 confined beads were called confined.
   - The Kalman likelihood equals the direct Gaussian likelihood (unit test).
   - Per-frame state accuracy 99.2%; 0.27% of still frames were called moving; onset found within 1 frame (median), within 3 frames for 90% of beads.
7. **On the 10× 5 ms movie:** 67 directed, 1 confined and 145 Brownian tracks (1 ms: 17, 7, 139). The moving beads form a patch around the indentation between frames 31 and 58 (median start and end of the directed tracks' moving frames).
8. **Refined positions:** with its classified model and fitted parameters, each track is run forward through the Kalman filter and backward through a Rauch–Tung–Striebel smoother. Every position is then estimated from the whole track, weighted by each localization's error, with its own SD.
   - Unit test: the smoothed positions and variances equal direct Gaussian conditioning on the whole track.
   - In simulations shaped like the 10× data, z error drops from 0.6 µm to about 0.085 µm for still beads (the reported SD matches the actual error) and to about 0.19 µm for beads that move for a while.
   - For beads that start or stop abruptly, the refined z error is 1.5–2× its reported SD, because the directed model expects smooth velocity changes.
   - Stored as shifts, so the explorer can add them to any coordinate set.

Outputs:
- `{movie}_track_motion.csv` (per track): class, ρ values, D, speeds, confinement, frames moving, first and last moving frame, speed while moving; with `--stages`, the frame each stage was entered (`stage2_indent_from`, …; −1 if never) and `frames_other_moving`.
- `{movie}_motion_states.csv` (per localization): P(moving), the velocity estimate, and the refined-position shift (`refine_dx_px, refine_dy_px, refine_dz_um`) with its SD (`refined_sd_x_px, refined_sd_y_px, refined_sd_z_um`); with `--stages`, the `stage` label (1-based; K+1 = other moving), each stage's probability `p_stage1…` (side trips included) and `p_other`. P(moving) is then the probability of the moving stages plus other moving.
- `track_motion_summary.json`: counts, error scales, and the fitted switching model or (with `--stages`) the stage parameters, the fraction of localizations in each stage and the number of beads entering it.

Units are per frame, or per second with `--frame-interval MS`; the movies do not record their frame interval.

## Outputs and coordinate conventions

`runs/<run>/<movie>_localizations.mat` hold native MATLAB tables:
- `localizations`: exactly `x1 x2 xMean y1 y2 yMean angleDegrees zMicrons frame_number track_number`
- `fitQuality`: the quality columns below plus `zEndpointRisk`
- `corrected`, `stabilized`, `drift`, `calibration`, `interpolatedCalibration`, `metadata`

`*_localizations.csv` has all of these columns in one file (used by the explorer).
`*_rejected.csv` holds transient tracks in the same format. `*_payload.mat` are
interchange matrices, not tables.

XY are **one-based pixels**, X right, Y down; the pixel size is in `run_info.json` (0.325 µm at 20×, 0.63 µm at 10×). Frames and tracks start at 1.
Lobe labels are canonical (connecting vector has nonnegative Y; angle =
`mod(atan2d(y2-y1,x2-x1),180)`, clockwise on a Y-down display). z is in stage-step units relative to the calibration plane where the lobes are horizontal (or vertical; `summary.json` → `z_zero` gives the plane). By default no refractive-index (focal-shift) correction is applied. With an air objective imaging a watery sample the focus moves farther in the sample than the stage moves, so `--true-depth` (analyze.py) or the **Report true depth** box in New analysis multiplies all exported z values by 1.33. This is the paraxial water/air index ratio; it is slightly larger at high NA, and `--axial-scale F` sets any factor. `run_info.json` and `summary.json` record the factor used. z is height, positive toward the indenter (see **z sign** above).

Quality columns:
- `residualRMS`: of the joint fit window.
- `minLobeSNR`: minimum lobe amplitude / camera noise SD.
- `jointEmitterCount`, `lobeSeparationPixels`, `amplitude1/2`, `sigma1/2`.
- `pairCost`, `separationResidual`: observed minus calibrated separation.
- `recovered`, `zStatus`: 0 unique, 1 outside calibration, 2 ambiguous, 3 resolved by separation.
- `sharedLobe`, `templateScore`, `fieldSeparationOffset`.
- **Signal outputs** (camera counts, not photons):
  - `backgroundLevel`: fitted local background per pixel at the bead.
  - `roiSignal`: sum of (image − fitted background plane) over pixels within `roi_radius_px` = 5 of either lobe centre.
  - `lobeSignal1`, `lobeSignal2`: integrated intensity of each fitted Gaussian, 2π·A·σ².
  - For a shared lobe the lobe signal is NaN, and the ROI also contains the neighbour's light. On the 10× 5 ms movie, `roiSignal` ≈ 0.98 × (`lobeSignal1` + `lobeSignal2`); the ROI also collects side-lobe rings.
- **Precision** (`xPrecisionPx`, `yPrecisionPx`, `anglePrecisionDeg`, `zPrecisionUm`): 1-SD uncertainties from the fit covariance. They are propagated to the midpoint and angle, and to z through the local slope of the calibration curve. With least squares they are sandwich estimates with a per-pixel variance a + b·signal fitted to the residuals (shot noise makes lobe pixels noisier than background; the simple residual-variance scaling was 1.5–2× too small); in simulation the actual scatter was 1.07–1.2× the reported value. With the sCMOS likelihood (`--camera`) they are Cramér-Rao-like bounds, and were calibrated to within ~10% in simulation.

## Camera calibration and sCMOS maximum likelihood (optional)

The default fit is least squares. It weights every pixel equally, although sCMOS pixels differ in offset, gain and read noise, and shot noise grows with signal. With a camera map, the fits instead maximize the Poisson likelihood with per-pixel read noise (Huang et al. 2013). In simulation this gave about 33% better angle (z) precision on dim beads, and error bars that match the real scatter.

1. Record about 1000 **dark frames** (shutter closed), in the same readout mode and ROI as the movies. For the Prime BSI movies this is "Full well" 11-bit 200 MHz. Optionally also record 3 or more stacks of uniform light at different intensities (flats).
2. Build the map:
   ```
   python camera_calibration.py --dark DARK.tif --flat F1.tif --flat F2.tif --flat F3.tif --out camera.npz
   python camera_calibration.py --dark DARK.tif --electrons-per-adu 0.xx --out camera.npz   # gain from the camera's test report
   ```
   `--roi X,Y` gives the sensor offset of the dark stack if it is cropped; `--per-pixel-gain` fits the gain per pixel (needs ≥ 3 flat levels).
3. Pass `--camera camera.npz` to `pipeline.py` or `analyze.py`. The movie's Micro-Manager ROI is used to cut the matching part of the map.

`tools/simulate_camera_noise.py` compares both estimators on synthetic beads with a given camera map.

## Validation

- **`test_pipeline.py`, `test_track_analysis.py`, `explorer/test_explorer.py`** (68 tests). `test_pipeline.py`:
  - Analytic Jacobian; overlapping beads to within 0.1 px.
  - The three-spot shared-lobe case; clipped edge fits; ring rejection; no detections in pure noise.
  - LAP global assignment; all three trackers at a crossing, and Kalman prediction through a gap.
  - Recovery prediction; z disambiguation by separation; drift; lateral model; transient split.
- **`tools/benchmark_injection.py`** adds real isolated calibration-bead PSFs (known z, scaled to movie brightness) to real movie frames. They are placed isolated, overlapping an existing bead (10–26 px), or 4–16 px from the image edge. It reports single-frame recall (no temporal recovery) by category, brightness and z, xy/z error, damage to existing beads, and unexplained extra detections.
- **`tools/validate_results.py`** re-checks every exported row and track, the calibration hold-out reproducibility, and real-data completeness: the fraction of frames in which each persistent bead is localized.

## Practical limits

- The calibration is pooled over the field of view. Field-dependent aberrations, refractive-index mismatch and non-Gaussian lobes can bias z. The template score and separation residual help spot outliers but do not measure accuracy.
- Two beads whose lobes coincide exactly on both sides (a full overlap) cannot be separated. A shared lobe's position is the centroid of the coincident lobes.
- Recovered localizations use the track as a prior for the search window only. Their coordinates come from the frame itself, but they are by construction beads the independent detector did not accept; `recovered=1` identifies them.
- Stabilization removes the median motion; if the indentation moves most of the field it will also remove real motion. Use the raw or corrected coordinates for mechanics unless you know the shake is rigid.
