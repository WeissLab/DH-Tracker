# DH-Tracker-2026

**DH-Tracker: 3D tracking and ensemble analysis of fluorescent beads using the double-helix point spread function**

DH-Tracker is a browser-based graphical interface that converts movies from a double-helix (DH) microscope into 3D bead trajectories, motion analysis and deformation maps. A bead z-scan calibrates the lobe rotation with depth and the depth-dependent lateral shift across the field of view, and field-dependent variations in lobe separation are corrected from each movie's beads. Beads are localized by fitting both lobes of the DH image, including overlapping neighbours. Trajectories are linked across frames, corrected for drift using the beads themselves, and refined by Kalman smoothing. Each bead's motion is classified into states over time using the models of aTrack and ExaTrack, and the displacements of all beads are interpolated into continuous deformation and strain fields. Weiss Lab, www.WeissLab.ca.

## Install (Windows)

1. Download **[`Install DH-Tracker-2026.bat`](Install%20DH-Tracker-2026.bat)**: open it here on GitHub and click **Download raw file** (the ↓ button at the top right of the file).
2. Double-click it. If Windows says it protected your PC, click **More info › Run anyway**.
3. Choose the install folder (default `Documents\DH-Tracker-2026`). The installer then:
   - downloads the newest version from this page,
   - offers to install [Miniconda](https://docs.anaconda.com/miniconda/) if it is missing,
   - sets up the Python packages (about 1 GB, a few minutes) and runs the tests,
   - asks for your movies folder and where to save results (remembered; changeable later in the app, ☰ › Folders).

Or, in PowerShell: `& ([scriptblock]::Create((irm https://raw.githubusercontent.com/WeissLab/DH-Tracker/main/install.ps1)))`

**Use it:** double-click **`Start DH-Tracker-2026.bat`** in the install folder; the app opens in your browser. Click **New analysis**, pick the calibration z-scan and the movie(s), and press Start. **`DH-Tracker-2026 guide.mp4`** (3 minutes, captioned) shows a whole session. No GPU is needed.

**Updates:** when a newer version is on this page, the Start launcher offers to update. Your results, calibrations and settings are kept.

Without the installer: **Code › Download ZIP**, unzip, install Miniconda, and double-click `Setup DH-Tracker-2026.bat`.

## Documentation

- [DHPSF_pipeline/README.md](DHPSF_pipeline/README.md): the program: setup, analysis options, outputs, the methods behind each step, tests.
- [DHPSF_pipeline/explorer/README.md](DHPSF_pipeline/explorer/README.md): the browser app.

Both also open inside the app (☰ › Help, or F1).

## Citation and license

If you use DH-Tracker-2026 in published work, please cite it: **Cite this repository** (right-hand panel) gives APA and BibTeX, from [CITATION.cff](CITATION.cff).

Lucien E. Weiss, Department of Engineering Physics and Lassonde DeepTech Institute, Polytechnique Montréal ([ORCID 0000-0002-0971-7329](https://orcid.org/0000-0002-0971-7329); lucien.weiss@polymtl.ca).

BSD 3-Clause (see [LICENSE](LICENSE)).
