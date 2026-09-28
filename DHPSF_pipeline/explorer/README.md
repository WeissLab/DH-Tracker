# DH-Tracker-2026 explorer

Local, offline web app for exploring 3D double-helix-PSF bead localizations and tracks on top of the raw movies. It covers the movie with overlays, a 3D track view, per-track plots, displacement fields, ROI selection, live re-tracking and export.

It uses only Python's standard-library `http.server`, plus numpy, tifffile, PIL and scipy. Plotly.js is bundled in `static/vendor/`, so no internet connection is needed. The server listens on 127.0.0.1 only.

## Run

**Help inside the app:** ☰ › Help (or **F1**) opens this README and the program's README as web pages, and the 3-minute video guide.

**Folders (☰ › Folders):** the movies folder (where New analysis's file picker opens) and the results folder (where new analyses are saved and which the run selector lists). **Change…** opens a folder dialog on this computer. Both are remembered in your user profile between sessions and across program updates (`user_settings.py`); setup asks for them once. Results made earlier in another folder stay there: move their folders into the results folder to see them in the list.

**Updates:** in a copy installed with the GitHub installer, the explorer asks GitHub for the newest version (at most every 6 hours; nothing is sent but the request) and shows **⬆ update available** in the top bar when there is one; ☰ › Help shows the installed version. Updating is done by the launcher: close its window and start `Start DH-Tracker-2026.bat` again.

```powershell
python DHPSF_pipeline/explorer/app.py
```

Then open http://127.0.0.1:8765. Use `127.0.0.1` rather than `localhost`.

Options:

- `--results DIR`: the result set shown first; you can switch to another one in the app. The default is `auto`, which picks the newest finished run in `DHPSF_pipeline/runs/`. If there is none yet (a new installation), the explorer opens with a "No analyses yet" screen and a New analysis button. Relative paths are resolved against the current directory, then against the repository root.
- `--port 8765`
- `--host 127.0.0.1`
- `--movie NAME=PATH`: add a movie or override a default path. Can be repeated.
- `--runs-dir DIR`: where new analyses are written. The default is `DHPSF_pipeline/runs`.
- `--analyze-script PATH`: the analysis engine. The default is `DHPSF_pipeline/analyze.py`; this option is only for testing with a stand-in script.

## Screen layout

- **Top bar**:
  - ☰ opens the Controls drawer.
  - Run selector: the result set being shown (see below). Hover over it for the folder, date and movies.
  - Movie selector.
  - **New analysis** starts an analysis. While one runs, an "analysing… N%" badge sits next to it; click the badge to see progress.
  - Frame step, play/pause and a frame slider.
  - **Markers** (centre, lobes and joining line of every bead in the current frame) and **Trails** (each bead's path over the last N frames) toggles, and a colour-by menu.
  - Three independent coordinate switches as clickable badges (also in Controls › Coordinates):
    - **correction**: the lateral shift-with-depth correction, xCorrected/yCorrected.
    - **XY stab**: subtracts the whole-field XY drift; the movie frames are shifted to match.
    - **Z stab**: subtracts the whole-field z drift (smoothed over time).
    They drive the 3D view, plots, table, displacement arrows and the exported xAnalysis/yAnalysis/zAnalysis columns.
- A divider between the movie and the side panel: drag to resize, double-click to reset.
  - Status badges showing corrected or stabilized coordinates, re-tracking and active filters.
  - ▤ shows or hides the side panel, and ? opens help.
- **Movie**: wheel zooms and drag pans. A floating toolbar holds pan/select, rectangle, lasso and polygon ROI tools, and fit-to-view. A selection pill shows the current selection and the number of ROIs, with ☆ and ✕ buttons.
  - **ROIs** are a selection tool only; they don't filter anything. A plain ROI *replaces* the selection with the tracks inside it, and earlier outlines are dropped. Hold **Shift** to *add* its tracks (labelled "+ ROIn") or **Alt** to *remove* them ("− ROIn", red). ✕ on the pill clears the selection and all ROIs.
  - **Starred tracks**: ☆ in the pill stars the selected track(s), and ★ unstars them. Starred tracks stay as chips under the pill: click one to select and show it, Shift/Ctrl-click to add or remove it, and "all ★" selects them all. Stars are kept per run and movie in this browser, and can be exported (Export › starred → CSV/MAT). They refer to the pipeline's track numbers, so they are hidden while a re-tracking is active.
- **Side panel**, with three tabs (3D, Tracks, Deformation):
  - **3D**: all tracks, with z as height (up +, so the indentation points down), coloured by frame with parula as in `LyndseyLafrenierePlotter.m`. Selected tracks are highlighted, and each track has a marker at the current frame. Click a track to select it.
  - **Tracks**: a sortable, virtualized track table plus plots for the selected tracks: z, Δx/Δy and 3D |Δr| against frame. A drift plot appears when a drift file exists.
  - **Z stabilization** uses the time-smoothed z drift (σ = 3 frames). Drift files from older runs, which have no `dzRaw` column, are smoothed on loading. The drift plot shows the smoothed dz, with the per-frame dz in red. The per-frame curve zigzags around the smoothed one, because its frame-to-frame steps are anticorrelated noise. In the Δx/Δy plot, Δx is the dark and Δy the light shade of each track's colour.
  - **Deformation**: a map of how the material moved, from a reference frame (**ref**, the undeformed state; shared with the displacement arrows' frame A) to the current movie frame. It follows playback.
    - Each bead present in both frames gives a measured displacement (x, y, z in µm). By default (⋯ › positions: **best**) the positions are lateral-corrected, whole-field drift removed in XY and Z, and refined by the motion analysis, whatever the Coordinates setting, so drift never counts as deformation; **as displayed** uses the Coordinates setting instead. These are interpolated into a continuous 3D field by Gaussian-process (kriging) interpolation: a smooth random field with a squared-exponential covariance and an unknown constant mean. Each bead is weighted by its own error at both frames, so the noisy z is smoothed rather than followed. The smoothing length ℓ and the field amplitudes are fitted by maximum likelihood at the frame with the largest displacements, then kept fixed so the map doesn't change smoothness during playback; ⋯ › smoothing ℓ sets it by hand.
    - The field is a Lagrangian map: it is drawn at where each piece of material started (the reference positions).
    - **z convention:** z is height, and up (toward the indenter, which comes from above) is +, so an indentation is **negative Δz**. This is the pipeline's convention for `zMicrons`. Result folders from before it was introduced are flipped when loaded (the server checks `z_convention` in `run_info.json` / `z_convention.json`), so all views and exports agree; `convert_z_up.py` converts such a folder's files.
    - **Warped plane** (default): the beads' median plane at the reference frame (a robust plane fit, so a tilted sample still gives a flat sheet) drawn as a 3D mesh surface, moved in x, y and z by the field at each frame. **depth** looks at a plane higher or lower in the sample, and **warp ×** exaggerates the displacement. The dots are the beads where they are now.
    - **Map (top view)**: the same plane seen from above as a colour map (as in the Methods figure), with the beads and, optionally, their lateral displacement arrows. Blank where no bead lies within about ℓ.
    - Colour by Δz (default), total or lateral |u|, Δx or Δy, or in-plane strain from the exact derivatives of the interpolated field: **areal strain** (εxx + εyy, %; negative = compressed), **shear** (maximum in-plane shear strain, %) and **surface tilt** (slope of the warped plane, °). Derivatives in depth are not offered: the bead layer is too thin to measure them.
    - The colour range is the largest value of the interpolated field near beads over the whole movie (about 30 sampled frames), so the deepest part of the dent is not clipped and frames can be compared. The axes and camera also stay fixed during playback.
    - **↻** (refresh): fits and draws the map again from scratch; if the movie's data did not load, it reloads it. A 3D view left blank because the browser dropped its graphics (WebGL) context is rebuilt automatically.
    - ⋯ menu: positions (best / as displayed), beads, bead arrows (arrow length, **auto**), smoothing length (**auto** = fitted; the box shows the value in use), and exports.
    - The info line is one short summary (beads, ℓ, a bead's displacement error, z stretch); hover it for the details. With positions **as displayed** and stabilization off it warns that drift is not removed. z is drawn stretched (the factor is shown), since the bead layer is much thinner than it is wide.
    - It needs at least 8 beads in both frames. A frame takes about 0.05–0.15 s for ~200 beads and about 1 s for ~1300 (the 2048² 10× movies). The fit and colour range run whenever the reference, positions or filters change (a message shows meanwhile; they use at most 600 beads, the drawn field all of them).
    - **Exports** (⋯ menu; frame range from Filters; saved in `results/exports`):
      - **field → CSV / MAT / VTK**: the warped plane at the current depth for every frame: grid x, y, reference height z0, ux, uy, uz (µm; z up +), areal and shear strain (%), tilt (°). The MAT arrays are frames × ny × nx. The VTK folder holds one structured grid per frame (the moved plane, with the displacement vector and strains), which ParaView opens as an animation. About 3 s for 91 frames.
      - **animation → MP4**: the current view (camera, colouring, warp) at 10 frames/s, recorded by the browser (H.264; WebM if the browser cannot make MP4).
      - **animation → GIF**: the same, 800 px wide, made by the server (Pillow), e.g. for slides.
  - **Axis ranges**: double-click an axis in the 3D view (on or next to the box edge along that axis, including the vertical z edge) or in any track plot (on the tick labels) to type its range. "auto" restores the automatic range. Ranges stay through redraws; 3D ranges reset when the units or z convention change.
- **Controls drawer**, closed by default and opened with ☰. It has collapsible sections:
  - **Display**: contrast, gamma, lobes, legend, dashed recovered localizations, only-selected, dim-unselected, tail length, fps, z colour range, auto resolution, preload.
  - **Selection**: ROI mode, invert, clear ROIs.
  - **Motion analysis** (see the pipeline README):
    - Shows how many tracks are directed, confined or Brownian (whole-track test, as in aTrack), and how often beads start and stop moving (two-state model, as in ExaTrack).
    - It runs with every new analysis. For older results, set an optional frame interval and click **run motion analysis**; this takes about 10 seconds, and the data reload when it's done.
    - **model over time**: *moving / still* (default) or *indentation stages* (still → indent → hold → retract → still; each bead can skip stages; about a minute). The still stages share one small step size, capped at what the still beads show, so a still stage cannot absorb motion. From any stage a bead can also go into **other moving** (directed motion in any 3D direction, unrelated to the indentation, e.g. pushed by a cell) and come back to the same stage. With stages, **colour: stage** (movie and 3D) colours each localization by its most probable stage (grey still, red indent, amber hold, blue retract, purple other moving) with a legend, and the info line lists how many beads entered each stage. *Moving* counts the indent and retract stages and other moving.
    - Once loaded:
      - **colour: moving** (movie) and **moving** (3D) colour each localization by its probability of moving, from grey through yellow to red.
      - The Tracks table gains **motion**, **moving** (frames) and **speed** columns.
      - The Tracks tab plots each selected track's probability of moving over the fraction of all beads moving.
      - Filters gain **motion** (moving at some point / directed / confined / Brownian).
  - **Filters**: minimum length, z range, valid z only, frame range, recovered, and **min brightness**. Brightness is a track's median weaker-lobe amplitude as a % of the tracks within 150 px (also the `bright` column in the Tracks table). Side-lobe ghosts and noise fits sit far below real beads; the pipeline removes the clear cases automatically (reject reason 5), and this slider is the manual override.
  - **Displacement field**: arrows from frame A to frame B, Δz colour map.
  - **Coordinates & extra layers**: coordinate mode and the rejected-tracks layer.
  - **Tracking**: live re-linking.
  - **Export**.
- There are no keyboard shortcuts, to keep things simple. The only keys are Enter / Backspace while drawing a polygon ROI (close / undo a vertex), and Esc to close the help (**?** button).

## Analysing new data

1. Click **New analysis** in the top bar. The window has three numbered steps. Each step's number turns green when it is complete, or orange when something needs attention, with a short status beside it. **Start analysis** stays at the bottom of the window and is enabled only when everything is ready.
2. **Step 1, Calibration**:
   - **Used before** lists recent calibration stacks (from earlier runs), each marked **fitted · starts right away** when its fit is cached and still current, or **needs fitting** otherwise. Click one to use it.
   - Or type or paste a path (Windows "Copy as path" works), or click **Browse…**. In the browser, double-click a folder to open it and double-click a file to pick it. The breadcrumb (starting at "Drives") moves back up.
   - The window then shows the frame count, size and exposure, and **checks the calibration** (10–30 s): enough beads per plane, and lobes that rotate between planes, i.e. a z-scan rather than a movie. Too dim a stack or a movie picked by mistake is reported in plain words, and Start stays disabled.
   - **Magnification**: 20× (0.325 µm/px), 10× (0.63 µm/px) or custom. It is preset from the file or folder names, with a note saying which name it came from.
     - It is never assumed: if no name says it, you must choose.
     - If the calibration and a movie seem to come from different objectives, a warning appears and you must choose; a calibration only applies to the same objective.
   - **Planes and z step** (collapsed; the defaults are usually right):
     - Choose the planes of the calibration stack to fit, [first, last], 1-based and inclusive. Planes outside the range are not fitted, which makes the run faster.
     - The range starts at the engine's suggestion: for 10×, the middle plane ±80 planes; for 20×, all planes.
     - Drag the two handles or type the numbers. The images show the first and last plane.
     - **z step** is filled in from the stage speed in the file name times the frame interval in the file's metadata (e.g. 25 µm/s × 40 ms = 1 µm), with a note saying so. If they are not available, the field is opened and marked, and Start stays disabled until you enter the z step (a wrong value would scale every z).
     - **Limit z to ±** (optional): leave it empty to use the engine's default, which is ±75 µm for 10× and the full range otherwise.
3. **Step 2, Movies**:
   - Add one or more movies with **Browse…** or by pasting paths (several pasted paths are split; in the browser, Ctrl- or Shift-click picks several files). All of them are analysed with the same calibration. Use ✕ to remove a movie.
   - Each movie is **checked for brightness** when added (a few seconds to a minute, on the central part of a few frames): **bright**, **dim · dim-data settings** (e.g. 1 ms) or **few beads**. Bright and dim movies can be mixed. The check never blocks Start, because the analysis checks each movie again itself.
   - **+ add the other N .tif files in this folder** adds the remaining movies next to the last one in one click. The calibration and continuation files of multi-file stacks are skipped, and hovering shows the names.
   - **Bead detection** (collapsed, and usually not needed; the preview loads when you open it):
     - Pick an example movie and frame; the middle frame is the default. The engine then finds beads on that frame once (`analyze.py --preview`, 5–20 s), using a permissive threshold.
     - The **strictness** slider (how much brighter than the camera noise a bead's dimmer spot must be; the pipeline's min lobe SNR) then instantly shows which beads are kept (green) and which are dropped (faint red), with a count. Wheel zooms and drag pans the preview.
     - If you move the slider, the value is passed as `--min-snr` for all movies, which overrides the automatic dim-data detection. **auto** goes back to letting the engine decide.
4. **Step 3, Start**:
   - A summary lists the calibration and planes, the movies (frames, how many are dim), the pixel size and z units, the results folder and a rough **time estimate**.
   - The estimate uses about 1.7 s per 1024 × 1024 calibration plane (skipped when the fit is cached) and 1.3 s per 1024 × 1024 movie frame, plus a minute per movie.
   - Anything still missing is listed in orange.
   - **Run name**: prefilled with the movie name plus the date and time.
   - **Report true depth in the sample**: with an air objective and a watery sample, all z values are multiplied by 1.33 (refractive-index focal shift). Left off, z is in stage units, as calibrated.
   - **Also save MATLAB tables** (needs MATLAB and takes about a minute). If MATLAB fails, the run still finishes with a warning, because the CSV results are already complete.
   - Both boxes are remembered for next time.
5. Click **Start analysis**. The window then shows the current stage, a progress bar, the elapsed time, the time left, the log and a Cancel button. Each step's share of the bar is its expected share of the time (a cached calibration takes almost none), and the time left comes from how fast the bar moved over the last few minutes, so it adapts when a crowded movie runs slower than expected. A run typically takes 5–10 minutes; large, crowded 2048² movies up to about 1.5 hours.
6. You can close the window while the analysis runs. The top-bar badge shows the progress and the time left; click it to open the window again. Reloading the page also picks the running job up again.
7. When the run finishes, click **Open results** to switch to it. If it fails, the window shows the error and the end of the log.

Behind the scenes:

- The server runs `analyze.py` as a separate process and polls the `status.json` it writes.
- Each run gets its own folder, `<name>/` in the results folder (chosen at setup; ☰ › Folders in the app; without a choice `DHPSF_pipeline/runs/`), holding the results, `status.json` and `analyze.log`. If the name is already taken, `_2` (then `_3`, …) is added.
- Only one analysis runs at a time.
- Cancel stops the whole process tree.
- If the explorer is closed, a running analysis carries on (and keeps the computer from sleeping). When the explorer is started again it finds the analysis (`job.json` in the run folder records the process), shows its progress, and Cancel works as before. An analysis that stopped without finishing (e.g. the computer restarted) is marked "interrupted".

**Switching result sets.** The run selector lists every folder in the results folder, newest first, plus the `DHPSF_pipeline/results*` folders that contain results. Runs that are still going, failed or were cancelled are listed but cannot be opened. Choosing a set reloads everything for it: the movies, pixel size, drift, rejected tracks and coordinate options. It also clears the caches and any re-tracking. No restart is needed. Each page (tab or window) keeps its own result set: every request names it (`?run=`), and the server keeps up to four other sets open besides the default, so two pages showing different runs do not interfere. The last set chosen is the default for pages opened later. The list picks up newly finished runs every 20 s and when it is opened.

## Inputs, read per movie from the selected result set

Movie paths and the pixel size come from the set's `run_info.json`, when it has one.

| file | use |
|---|---|
| `{movie}_localizations.csv` (v2) or `{movie}_payload.csv` (v1) | Required columns: `x1,x2,xMean,y1,y2,yMean,angleDegrees,zMicrons,frame_number,track_number`. Every other numeric column, such as quality metrics, `recovered`, `zStatus`, `xCorrected…` and `xStabilized…`, is passed through, shown in the tooltip and exported. |
| `{movie}_drift.csv` (optional) | `frame_number,dx,dy,dz`, with dx and dy in px and dz in µm, relative to the movie's median (frame 1 in folders written before `rereference_drift.py`). Enables the stabilized coordinates. |
| `{movie}_rejected.csv` (optional) | Transient or likely-noise tracks. Enables the grey rejected layer. |

The server reloads a CSV whenever its modification time changes.

## Coordinates

The CSV x/y values are **one-based** pixels, with +x to the right and +y down. Pixel `x` is drawn on image column `x−1`; this was checked against lobe centroids to within 0.02 px. To convert to µm, multiply px by 0.325.

The **coordinates** setting (Controls › Coordinates) drives the 3D view, the plots, the table statistics, the displacement arrows and the exported `xAnalysis/yAnalysis/zAnalysis` columns:

- **raw**: `xMean, yMean, zMicrons`.
- **lateral-corrected**: `xCorrected, yCorrected`, the midpoint with the calibrated shift-with-z removed. This is the default when those columns exist.
- **corrected + stabilized**: `xStabilized, yStabilized, zStabilized`, or corrected minus the frame's drift when those columns are missing. The frames themselves are also shifted by (−dx, −dy) on the server.
- **refined positions** (after the motion analysis): adds the refined-position shift from `{movie}_motion_states.csv`. A Kalman filter runs forward over each whole track and a smoother backward, with the track's own motion model (Brownian, confined or directed), so every position is estimated from the whole track. It takes most of the localization noise out, especially in z. It is model-based: an abrupt real jump is spread over a few frames, and for beads that start or stop suddenly the refined z error is larger than its reported SD. The shifts belong to the pipeline's tracks, so the option is off while a re-tracking is active.

The movie overlay always draws the **raw** lobe and centre positions, because that is where the PSF is. When the stabilized mode is on, those raw positions are shifted by the frame drift so they stay on the shifted frames.

## Tracking (Controls › Tracking)

Tracking re-links every localization with the pipeline's own trackers:

- `lap`: two-stage LAP with gap closing.
- `kalman`: constant-velocity Kalman filter with LAP assignment, coasting through gaps.
- `nearest`: greedy baseline.

The parameters are the link and gap distance, maximum frame gap, angle cost, the Kalman noise settings and minimum track length. The defaults are read from `pipeline.Config`.

Re-tracking runs `pipeline.track` in a separate worker process (`retrack_worker.py`), so the server stays responsive while it works. Tracks shorter than the minimum length are set aside and drawn grey. After a re-track, every view and export uses the new track ids. Exports then add `pipelineTrackNumber` and `shortTrack` columns and record the tracker name and parameters in the metadata. "Reset to pipeline" restores the original track ids.

Typical run times:

| method | cells (~14k localizations) | collagen (~22k localizations) |
|---|---|---|
| `lap` | ~0.4 s | ~1 s |
| `kalman` | ~0.7 s | ~4 s |
| `nearest` | ~3 s | ~25 s |

**Progress.** Long work shows a small panel under the top bar with a label, the elapsed time and a progress bar. The bar fills to the fraction done where that is known: loading a movie (the server reading the CSV, the download in MB, building the tracks), fitting the deformation map (smoothing length, then the colour range frame by frame) and the deformation exports (frame by frame). Work without a measurable fraction, such as switching result sets, re-tracking, track exports and the on-demand motion analysis, shows a moving bar. Work continues in a background tab.

## Export

- Four sets can be exported:
  - **all**: every localization of the movie, ignoring filters and selection (after a re-tracking, short tracks included);
  - **filtered**: the tracks that pass the current filters;
  - **selected**: the tracks you picked (click, ROI or table);
  - **starred**: the starred tracks.
- With **apply point filters** on, filtered, selected and starred tracks are cut to the rows that pass the point filters (frame range, z range, recovered); off, whole tracks are exported. Track-level filters (minimum length, brightness, motion class) decide which tracks count as filtered either way.
- Each set is written to `<results>/exports/<timestamp>_<movie>_<label>.csv` and `.mat`, and the CSV is also downloaded.
- **📂 Open export folder** (Export section, and the Deformation ⋯ menu) opens that run's `exports` folder in File Explorer.
  - The `.mat` file holds column vectors, a `columns` cell and `export_info_json`.
  - Stabilized columns are added when a drift file exists.
  - The `xAnalysis/yAnalysis/zAnalysis` columns follow the chosen coordinate mode and are all in **µm** (x, y: one-based pixels × the pixel size), whatever units the views show; the other x/y columns stay in pixels.
- ROIs can be saved as JSON.
- Every plot, the 3D view included, can be saved as PNG with the save (disk) icon in its toolbar.

## Performance design

- Movie overlays are built once per frame and per set of options as colour-bucketed `Path2D` objects in image coordinates. Panning and zooming only re-stroke those cached paths.
- Redraws are coalesced with `requestAnimationFrame`, and hover hit-testing runs at most once per animation frame.
- At low zoom only bead centres are drawn: no lobes and no labels.
- Frames are preloaded as 2×2 or 4×4 binned versions first. Full-resolution frames around the current one are fetched when the browser is idle. Playback shows whichever resolution is ready rather than waiting for full resolution.
- Hidden tabs and the closed drawer do no rendering or computation; they are marked dirty and redrawn when shown.
- The 3D figure is never rebuilt when the frame changes. Only the current-frame marker trace is restyled, once scrubbing settles, and not at all during playback. The large 3D base trace is cached across selection changes.
- 3D and plot updates after a selection are deferred so the movie and table respond first.
- The frame cursors on the plots are plain DOM lines, not Plotly relayouts.
- The track table is virtualized: only the visible rows are in the DOM.
- `window.__perf` exposes the draw, overlay-build and 3D timings.

## Tests

```powershell
python DHPSF_pipeline/explorer/test_explorer.py
```

The tests cover:

- the loader, drift and stabilized columns, and coordinate modes
- exports, including re-tracking overrides
- frame rendering and shifting
- the re-tracking worker with all three methods
- the folder browser, TIFF inspection, the detection preview and frame previews
- listing and switching runs, including over HTTP
- the analysis job manager, which is driven by a small stand-in engine script (never the real `analyze.py`): the run to completion, one job at a time, cancel, a crash, a lost final status, and input validation

## Files

- `app.py`: the server. Routes:
  - `GET /api/movies`
  - `GET /api/movie/<name>/{info,localizations,rejected,drift}`
  - `GET /api/movie/<name>/frame/<i>?vmin&vmax&gamma&bin&stab`
  - `GET /api/tracking/defaults`
  - `POST /api/movie/<name>/retrack`
  - `POST /api/export`
  - `POST /api/export_roi`
  - `GET /api/exports/<file>`
  - `GET /api/runs`, `POST /api/runs/select {id}` (sets the default); data requests take `?run=<id>` for the page's own result set
  - `GET /api/browse?path=`: a folder's sub-folders and `.tif` files. `path=drives` lists the drive roots.
  - `GET /api/inspect?path=`: `analyze.py --inspect`, cached. If that fails, basic information read with tifffile.
  - `GET /api/preview?path=&frame=`: `analyze.py --preview FILE --frame N`, cached per file and frame.
  - `GET /api/file_info?path=` and `GET /api/preview_frame?path=&frame=&vmin=&vmax=&bin=`: read-only frames of any `.tif`, with 1-based frame numbers.
  - `POST /api/analyze`: returns 409 while another analysis is running.
  - `GET /api/jobs/<id>`, `GET /api/jobs/current`, `POST /api/jobs/<id>/cancel`
- `retrack_worker.py`: the tracking worker that imports `pipeline.py`.
- `static/index.html`, `static/style.css`.
- `static/js/`:
  - `main.js`: wiring, playback, tracking and export.
  - `movie.js`: canvas, frame cache, overlays and ROI tools.
  - `plots.js`: 3D and 2D plots.
  - `data.js`: dataset, coordinates, filters and statistics.
  - `table.js`: virtualized track table.
  - `analysis.js`: the New analysis window, the file browser and job progress.
  - `util.js`, `state.js`.
- `static/vendor/plotly.min.js`: plotly.js v3.5.0.
