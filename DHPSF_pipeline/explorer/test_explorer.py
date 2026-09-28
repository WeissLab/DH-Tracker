"""Unit tests for the explorer's loader / export / rendering helpers.

Run:  python -m pytest DHPSF_pipeline/explorer/test_explorer.py -q
  or: python DHPSF_pipeline/explorer/test_explorer.py
Only temporary directories are written.
"""
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import app  # noqa: E402

V1_HEADER = "x1,x2,xMean,y1,y2,yMean,angleDegrees,zMicrons,frame_number,track_number"
V1_ROWS = [
    "10,20,15,30,31,30.5,5,1.5,1,1",
    "11,21,16,30,31,30.5,6,NaN,2,1",
    "100,110,105,200,201,200.5,90,-2.0,1,2",
]


def write(p: Path, header: str, rows):
    p.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_prefers_v2_file(self):
        write(self.d / "cells_payload.csv", V1_HEADER, V1_ROWS)
        self.assertEqual(app.find_localization_file(self.d, "cells").name, "cells_payload.csv")
        write(self.d / "cells_localizations.csv", V1_HEADER, V1_ROWS)
        self.assertEqual(app.find_localization_file(self.d, "cells").name, "cells_localizations.csv")
        self.assertIsNone(app.find_localization_file(self.d, "collagen"))

    def test_load_v1_and_json(self):
        write(self.d / "cells_payload.csv", V1_HEADER, V1_ROWS)
        loc = app.load_localizations(self.d / "cells_payload.csv")
        self.assertEqual(loc["n"], 3)
        self.assertEqual(loc["columns"], V1_HEADER.split(","))
        self.assertTrue(np.isnan(loc["data"]["zMicrons"][1]))
        js = json.loads(app.localizations_to_json(loc))
        self.assertIsNone(js["data"]["zMicrons"][1])           # NaN -> null
        self.assertEqual(js["data"]["frame_number"], [1, 2, 1])  # integer columns stay ints
        self.assertEqual(js["n"], 3)

    def test_load_v2_optional_columns(self):
        hdr = V1_HEADER + ",residualRMS,recovered,zStatus,flag"
        rows = [r + ",0.5,1,0,ok" for r in V1_ROWS]
        write(self.d / "m_localizations.csv", hdr, rows)
        loc = app.load_localizations(self.d / "m_localizations.csv")
        self.assertIn("recovered", loc["columns"])
        np.testing.assert_array_equal(loc["data"]["recovered"], [1, 1, 1])
        self.assertTrue(np.isnan(loc["data"]["flag"]).all())   # non-numeric -> NaN

    def test_missing_required_column(self):
        write(self.d / "bad.csv", "x1,x2,xMean", ["1,2,3"])
        with self.assertRaises(ValueError):
            app.load_localizations(self.d / "bad.csv")

    def test_drift_and_stabilized_columns(self):
        # with dzRaw the file's dz is used as is (without it, dz would be smoothed: see test below)
        write(self.d / "cells_drift.csv", "frame_number,dx,dy,dz,dzRaw", ["2,1.5,-2,0.25,0.25", "1,0,0,0,0"])
        dr = app.load_drift(app.find_drift_file(self.d, "cells"))
        np.testing.assert_array_equal(dr["frame_number"], [1, 2])      # sorted
        write(self.d / "cells_payload.csv", V1_HEADER, V1_ROWS)
        loc = app.load_localizations(self.d / "cells_payload.csv")
        added = app.add_stabilized_columns(loc, dr)
        self.assertEqual(added, ["xStabilized", "yStabilized", "zStabilized"])
        np.testing.assert_allclose(loc["data"]["xStabilized"], [15, 16 - 1.5, 105])
        np.testing.assert_allclose(loc["data"]["yStabilized"], [30.5, 30.5 + 2, 200.5])
        self.assertAlmostEqual(loc["data"]["zStabilized"][0], 1.5)
        self.assertEqual(app.add_stabilized_columns(loc, dr), [])        # idempotent

    def test_smoothed_z_drift_is_referenced_to_its_median(self):
        # older drift files have only the per-frame dz: smoothed here, and like the pipeline's drift
        # referenced to its median over the movie (not to frame 1)
        write(self.d / "old_drift.csv", "frame_number,dx,dy,dz", [f"{f},0,0,{0.1*f}" for f in range(1, 31)])
        dr = app.load_drift(self.d / "old_drift.csv")
        self.assertAlmostEqual(float(np.median(dr["dz"])), 0.0, places=9)
        np.testing.assert_allclose(dr["dzRaw"], 0.1*np.arange(1, 31))


class ExportTests(unittest.TestCase):
    def test_export_csv_and_mat(self):
        import scipy.io
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            write(d / "cells_payload.csv", V1_HEADER, V1_ROWS)
            loc = app.load_localizations(d / "cells_payload.csv")
            res = app.write_export(loc, [2, 0, 0], d / "exports", "cells", "sel/ected tracks",
                                   extra={"note": "x"}, timestamp="20000101_000000")
            self.assertEqual(res["rows"], 2)
            self.assertEqual(res["tracks"], 2)
            csv_p, mat_p = Path(res["csv"]), Path(res["mat"])
            self.assertEqual(csv_p.name, "20000101_000000_cells_sel_ected_tracks.csv")
            lines = csv_p.read_text().strip().splitlines()
            self.assertEqual(lines[0], V1_HEADER)
            self.assertEqual(len(lines), 3)
            self.assertTrue(lines[1].endswith(",1,1"))  # sorted by track then frame
            m = scipy.io.loadmat(str(mat_p))
            self.assertEqual(m["xMean"].shape, (2, 1))
            cols = [str(c[0]) for c in m["columns"].ravel()]
            self.assertEqual(cols, V1_HEADER.split(","))
            # original table not mutated / NaN written as NaN
            res2 = app.write_export(loc, [1], d / "exports", "cells", "nan", timestamp="20000101_000001")
            self.assertIn("NaN", Path(res2["csv"]).read_text())
            with self.assertRaises(ValueError):
                app.write_export(loc, [99], d / "exports", "cells", "oops")

    def test_export_adds_stabilized_columns_from_drift(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            write(d / "cells_payload.csv", V1_HEADER, V1_ROWS)
            write(d / "cells_drift.csv", "frame_number,dx,dy,dz", ["1,0,0,0", "2,1,1,0.5"])
            loc = app.load_localizations(d / "cells_payload.csv")
            res = app.write_export(loc, [0, 1], d / "ex", "cells", "stab",
                                   drift=app.load_drift(d / "cells_drift.csv"), timestamp="t")
            hdr = Path(res["csv"]).read_text().splitlines()[0].split(",")
            self.assertEqual(hdr[-3:], ["xStabilized", "yStabilized", "zStabilized"])
            self.assertNotIn("xStabilized", loc["columns"])  # cache untouched

    def test_field_export_csv_mat_vtk(self):
        import scipy.io
        nf, ny, nx = 2, 3, 4
        rng = np.random.default_rng(0)
        body = {"movie": "m", "xs": list(range(nx)), "ys": list(range(ny)), "z0": np.zeros((ny, nx)).tolist(),
                "frames": [1, 2], "info": {"reference_frame": 1}}
        for v in app.FIELD_VARS:
            body[v] = rng.normal(size=(nf, ny, nx)).tolist()
        body["uz"][1][0][0] = None                                # a hole (no beads nearby)
        with tempfile.TemporaryDirectory() as t:
            res = app.write_field_export(body, Path(t), timestamp="t")
            rows = Path(res["csv"]).read_text().splitlines()
            self.assertEqual(len(rows), 1 + nf * ny * nx)
            m = scipy.io.loadmat(res["mat"])
            self.assertEqual(m["uz"].shape, (nf, ny, nx))
            self.assertTrue(np.isnan(m["uz"][1, 0, 0]))
            np.testing.assert_allclose(m["ux"], np.asarray(body["ux"]))
            vtk = sorted(Path(res["vtk_dir"]).iterdir())
            self.assertEqual([p.name for p in vtk], ["plane_0001.vtk", "plane_0002.vtk"])
            txt = vtk[0].read_text()
            self.assertIn(f"DIMENSIONS {nx} {ny} 1", txt)
            self.assertIn("VECTORS displacement_um float", txt)
            with self.assertRaises(ValueError):
                app.write_field_export({**body, "ux": body["ux"][:1]}, Path(t), timestamp="u")

    def test_animation_gif_and_video(self):
        import base64
        from PIL import Image
        frames = []
        for c in (0, 128, 255):
            b = io.BytesIO(); Image.new("RGB", (20, 10), (c, 0, 0)).save(b, "PNG")
            frames.append("data:image/png;base64," + base64.b64encode(b.getvalue()).decode())
        with tempfile.TemporaryDirectory() as t:
            res = app.write_animation({"movie": "m", "kind": "gif", "fps": 5, "frames": frames}, Path(t), timestamp="t")
            self.assertEqual(Image.open(res["path"]).n_frames, 3)
            vid = app.write_animation({"movie": "m", "kind": "mp4", "data": "data:video/mp4;base64," + base64.b64encode(b"abc").decode()},
                                      Path(t), timestamp="v")
            self.assertEqual(Path(vid["path"]).read_bytes(), b"abc")
            with self.assertRaises(ValueError):
                app.write_animation({"movie": "m", "kind": "exe", "data": ""}, Path(t))


class UserSettingsTests(unittest.TestCase):
    def test_folders_are_remembered(self):
        import user_settings as us
        with tempfile.TemporaryDirectory() as t:
            old = os.environ.get("DHTRACKER_SETTINGS")
            os.environ["DHTRACKER_SETTINGS"] = str(Path(t) / "cfg" / "settings.json")
            try:
                self.assertEqual(us.results_dir(), us.LEGACY_RESULTS)     # nothing chosen: the program's runs/
                self.assertIsNone(us.data_dir())
                us.save(data_dir=Path(t), results_dir=Path(t) / "res")
                self.assertEqual((us.data_dir(), us.results_dir()), (Path(t), Path(t) / "res"))   # read back from disk
                us.save(data_dir=Path(t) / "gone")                          # a folder that no longer exists
                self.assertIsNone(us.data_dir())
                us.save(results_dir=None)
                self.assertEqual(us.results_dir(), us.LEGACY_RESULTS)
                (Path(t) / "cfg" / "settings.json").write_text("not json")  # a damaged file: defaults
                self.assertEqual(us.load(), {})
            finally:
                if old is None:
                    os.environ.pop("DHTRACKER_SETTINGS", None)
                else:
                    os.environ["DHTRACKER_SETTINGS"] = old


class VersionCheckTests(unittest.TestCase):
    def test_update_notice(self):
        old = app.REPO
        calls = []

        def github(answer):
            def f():
                calls.append(1)
                if isinstance(answer, Exception):
                    raise answer
                return answer
            return f
        with tempfile.TemporaryDirectory() as t:
            app.REPO = Path(t)
            app._VERSION_CACHE.clear()
            try:
                v = app.version_info(1000., github("b" * 40))            # not made by the installer: never asks
                self.assertEqual((v["installed"], v["update_available"], len(calls)), (None, False, 0))
                (Path(t) / "version.txt").write_text("a" * 40 + "\n")
                v = app.version_info(1000., github("b" * 40))
                self.assertTrue(v["update_available"])
                self.assertEqual(v["latest"], "b" * 40)
                app.version_info(2000., github("c" * 40))                # cached for 6 hours
                self.assertEqual(len(calls), 1)
                app._VERSION_CACHE.clear()
                v = app.version_info(1000., github(OSError("offline")))  # offline: no notice
                self.assertEqual((v["latest"], v["update_available"]), (None, False))
                app._VERSION_CACHE.clear()
                self.assertFalse(app.version_info(1000., github("a" * 40))["update_available"])   # up to date
                app._VERSION_CACHE.clear()
                self.assertIsNone(app.version_info(1000., github("<html>"))["latest"])            # not a version
            finally:
                app.REPO = old
                app._VERSION_CACHE.clear()


class CoordinateAndTrackingTests(unittest.TestCase):
    def test_analysis_columns_modes(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            hdr = V1_HEADER + ",xCorrected,yCorrected"
            write(d / "m_localizations.csv", hdr, [r + f",{i}.5,{i}.25" for i, r in enumerate(V1_ROWS)])
            write(d / "m_drift.csv", "frame_number,dx,dy,dz,dzRaw", ["1,0,0,0,0", "2,1,2,0.5,0.5"])
            loc = app.load_localizations(d / "m_localizations.csv")
            dr = app.load_drift(d / "m_drift.csv")
            raw = app.analysis_columns(loc, "raw", dr)
            np.testing.assert_allclose(raw["xAnalysis"], loc["data"]["xMean"])
            cor = app.analysis_columns(loc, "corrected", dr)
            np.testing.assert_allclose(cor["xAnalysis"], [0.5, 1.5, 2.5])
            st = app.analysis_columns(loc, "stabilized", dr)
            np.testing.assert_allclose(st["xAnalysis"], [0.5, 0.5, 2.5])   # frame 2 shifted by dx=1
            np.testing.assert_allclose(st["yAnalysis"], [0.25, -0.75, 2.25])
            um = app.analysis_columns(loc, "stabilized", dr, 0.325)                # x, y in µm; z unchanged
            np.testing.assert_allclose(um["xAnalysis"], 0.325 * np.array([0.5, 0.5, 2.5]))
            np.testing.assert_allclose(um["zAnalysis"], st["zAnalysis"], equal_nan=True)
            self.assertEqual(app.analysis_columns(loc, "stabilized", None)["mode"], "lateral-corrected")
            # independent XY / Z stabilization
            zonly = app.analysis_columns(loc, {"corr": True, "xy": False, "z": True}, dr)
            np.testing.assert_allclose(zonly["xAnalysis"], [0.5, 1.5, 2.5])
            np.testing.assert_allclose(zonly["zAnalysis"], loc["data"]["zMicrons"] - np.array([0, 0.5, 0]), equal_nan=True)
            self.assertEqual(zonly["mode"], "lateral-corrected · Z stabilized")
            xyonly = app.analysis_columns(loc, {"corr": False, "xy": True, "z": False}, dr)
            np.testing.assert_allclose(xyonly["zAnalysis"], loc["data"]["zMicrons"], equal_nan=True)
            self.assertEqual(xyonly["mode"], "raw · XY stabilized")
            loc1 = app.load_localizations(d / "m_localizations.csv")
            del loc1["data"]["xCorrected"]
            self.assertEqual(app.analysis_columns(loc1, "corrected", None)["mode"], "raw")

    def test_export_overrides(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            write(d / "c.csv", V1_HEADER, V1_ROWS)
            loc = app.load_localizations(d / "c.csv")
            res = app.write_export(loc, [0, 1, 2], d / "ex", "c", "rt", timestamp="t",
                                   overrides={"track_number": np.array([5., 5., 6.]),
                                              "pipelineTrackNumber": loc["data"]["track_number"]})
            lines = Path(res["csv"]).read_text().strip().splitlines()
            self.assertTrue(lines[0].endswith("track_number,pipelineTrackNumber"))
            self.assertEqual(res["tracks"], 2)
            self.assertEqual(loc["data"]["track_number"].tolist(), [1, 1, 2])  # cache untouched

    def test_resolve_results(self):
        with tempfile.TemporaryDirectory() as t:   # no runs yet (new installation) -> the runs folder itself
            runs = Path(t) / "runs"
            self.assertEqual(app.resolve_results("auto", runs_dir=runs), runs)
            self.assertTrue(runs.is_dir())
        self.assertTrue(app.resolve_results("C:/x").is_absolute())

    def test_retrack_worker(self):
        try:
            import retrack_worker as W
            W._pipeline()
        except Exception as e:  # pipeline.py being edited / deps missing
            self.skipTest(f"pipeline not importable: {e}")
        # two beads moving right by 1 px per frame for 5 frames, plus one isolated blip
        xs, ys, fs = [], [], []
        for f in range(1, 6):
            for x0, y0 in ((100, 100), (300, 300)):
                xs.append(x0 + f); ys.append(y0); fs.append(f)
        xs.append(800); ys.append(800); fs.append(3)
        n = len(xs)
        cols = {"xMean": np.array(xs, float), "yMean": np.array(ys, float), "frame_number": np.array(fs, float),
                "angleDegrees": np.zeros(n), "x1": np.array(xs) - 8., "x2": np.array(xs) + 8.,
                "y1": np.array(ys, float), "y2": np.array(ys, float)}
        for method in ("lap", "kalman", "nearest"):
            r = W.run(cols, {"method": method, "min_track_length": 3})
            ids = r["track_number"]
            self.assertEqual(len(set(ids[:-1])), 2, method)
            self.assertTrue(r["short"][-1] and not r["short"][:-1].any(), method)
            self.assertEqual(r["summary"]["n_tracks"], 2)
            self.assertEqual(r["summary"]["n_full_length"], 2)


class RenderTests(unittest.TestCase):
    def test_render_and_shift(self):
        from PIL import Image
        f = np.full((64, 64), 100, np.uint16)
        f[20, 30] = 1000
        img = np.asarray(Image.open(io.BytesIO(app.render_frame(f, 100, 1000, 1.0, fmt="png"))))
        self.assertEqual(img[20, 30], 255)
        # drift dx=+3, dy=-2 px: content at (x, y) must appear at (x-3, y+2)
        sh = np.asarray(Image.open(io.BytesIO(app.render_frame(f, 100, 1000, 1.0, fmt="png",
                                                               shift=(3, -2), cval=100))))
        self.assertEqual(np.unravel_index(sh.argmax(), sh.shape), (22, 27))
        b2 = np.asarray(Image.open(io.BytesIO(app.render_frame(f, 100, 1000, 1.0, binning=2, fmt="png"))))
        self.assertEqual(b2.shape, (32, 32))


class BrowseTests(unittest.TestCase):
    def test_browse_lists_folders_and_tifs(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            (d / "sub").mkdir()
            (d / "Another").mkdir()
            (d / ".hidden").mkdir()
            (d / "__pycache__").mkdir()
            (d / "b.TIFF").write_bytes(b"x" * 10)
            (d / "a.tif").write_bytes(b"x" * 3)
            (d / "notes.txt").write_text("x")
            r = app.browse_dir(str(d))
            self.assertEqual(Path(r["path"]), d.resolve())
            self.assertEqual(r["dirs"], ["Another", "sub"])
            self.assertEqual([f["name"] for f in r["files"]], ["a.tif", "b.TIFF"])
            self.assertEqual(r["files"][0]["size"], 3)
            self.assertEqual(Path(r["parent"]), d.resolve().parent)
            with self.assertRaises(app.HttpError) as cm:
                app.browse_dir(str(d / "missing"))
            self.assertEqual(cm.exception.code, 404)
            with self.assertRaises(app.HttpError) as cm:
                app.browse_dir("relative/path")
            self.assertEqual(cm.exception.code, 400)
        drives = app.browse_dir("drives")
        self.assertEqual(drives["path"], "drives")
        self.assertTrue(drives["dirs"])

    def test_inspect_fallback_and_naming(self):
        import tifffile
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "beads_10x_test.tif"
            tifffile.imwrite(str(p), np.zeros((5, 16, 20), np.uint16))
            info = app.fallback_inspect(p)
            self.assertEqual((info["frames"], info["height"], info["width"]), (5, 16, 20))
            self.assertEqual(info["objective_guess"], "10x")
            self.assertAlmostEqual(info["pixel_size_guess_um"], 0.63)
            # missing script -> fallback with a warning
            res = app.Inspector(script=Path(t) / "nope.py")(str(p))
            self.assertEqual(res["source"], "fallback")
            self.assertTrue(res["warning"])
            # a script that implements --inspect is used (and cached)
            fake = Path(t) / "fake.py"
            fake.write_text(FAKE_ENGINE, encoding="utf-8")
            ins = app.Inspector(script=fake)
            res = ins(str(p))
            self.assertEqual(res["source"], "analyze.py")
            self.assertEqual(res["frames"], 7)
            self.assertEqual(res["suggested_planes"], [2, 6])
            self.assertEqual(len(ins.cache), 1)
            with self.assertRaises(app.HttpError):
                ins(str(Path(t) / "missing.tif"))
            # --preview: parsed, NaN made JSON-safe, cached per frame; engine errors -> 502
            pv = ins.preview(str(p), 3)
            self.assertEqual(pv["frame"], 3)
            self.assertEqual([b["snr"] for b in pv["beads"]], [2.0, 5.5, None])
            json.dumps(pv, allow_nan=False)
            self.assertIs(ins.preview(str(p), 3), pv)
            with self.assertRaises(app.HttpError) as cm:
                ins.preview(str(p), 9)
            self.assertEqual(cm.exception.code, 502)
            with self.assertRaises(app.HttpError):
                app.Inspector(script=Path(t) / "nope.py").preview(str(p), 1)
            with self.assertRaises(app.HttpError):
                ins.preview(str(Path(t) / "notes.txt"), 1)
        now = app._dt.datetime(2026, 9, 26, 14, 5)
        self.assertEqual(app._finite({"a": [1.0, float("inf")], "b": float("nan")}), {"a": [1.0, None], "b": None})
        self.assertEqual(app.default_run_name(r"D:\x\10x_0.04s_5ms_1_MMStack_Pos0.ome.tif", now),
                         "10x_0.04s_5ms_1_20260926-1405")
        self.assertEqual(app.default_run_name("/a/movie one.tiff", now), "movie_one_20260926-1405")
        self.assertEqual(app.guess_objective(r"D:\data\20X\plain.tif"), "20x")
        self.assertIsNone(app.guess_objective(r"D:\data\plain.tif"))


class FilePreviewTests(unittest.TestCase):
    def test_info_and_frames(self):
        import tifffile
        from PIL import Image
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "stack.tif"
            stack = np.full((5, 32, 40), 100, np.uint16)
            stack[3, 10, 20] = 5000          # plane 4 (1-based)
            tifffile.imwrite(str(p), stack)
            fp = app.FilePreviews()
            info = fp.info(str(p))
            self.assertEqual((info["frames"], info["height"], info["width"]), (5, 32, 40))
            img = np.asarray(Image.open(io.BytesIO(fp.frame(str(p), {"frame": "4", "vmin": "100", "vmax": "1000"}))))
            self.assertEqual(img.shape, (32, 40))
            self.assertGreater(img[10, 20], 200)
            small = np.asarray(Image.open(io.BytesIO(fp.frame(str(p), {"frame": "1", "bin": "4"}))))
            self.assertEqual(small.shape, (8, 10))
            for bad in ({"frame": "0"}, {"frame": "6"}):
                with self.assertRaises(app.HttpError):
                    fp.frame(str(p), bad)
            (Path(t) / "x.txt").write_text("x")
            for path in (str(Path(t) / "x.txt"), "relative.tif", str(Path(t) / "missing.tif")):
                with self.assertRaises(app.HttpError):
                    fp.info(path)
            import gc
            fp.clear()
            gc.collect()      # release the memory map so Windows lets us delete the file


def make_run(d: Path, movies=("m1",), pixel=0.63, status=None, mtime=None, csv=True):
    d.mkdir(parents=True, exist_ok=True)
    info = {"version": 2, "pixel_size_um": pixel,
            "movies": {m: {"path": str(d / f"{m}.tif")} for m in movies}}
    (d / "run_info.json").write_text(json.dumps(info), encoding="utf-8")
    if csv:
        for m in movies:
            write(d / f"{m}_localizations.csv", V1_HEADER, V1_ROWS)
    if status:
        (d / "status.json").write_text(json.dumps({"state": status}), encoding="utf-8")
    if mtime:
        import os
        for p in d.iterdir():
            os.utime(p, (mtime, mtime))
    return d


class RunsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.runs = self.base / "runs"
        self.saved = (app.PIPELINE_DIR, app.RUNS_DIR, app.STATE, app.PIXEL_SIZE_UM)
        app.PIPELINE_DIR, app.RUNS_DIR = self.base, self.runs
        make_run(self.runs / "old", movies=("a",), mtime=1_000_000)
        make_run(self.runs / "new", movies=("b", "c"), pixel=0.325, mtime=2_000_000, status="done")
        make_run(self.runs / "failed", movies=("x",), status="error", csv=False, mtime=3_000_000)
        make_run(self.base / "results_legacy", movies=("cells",), pixel=0.5, mtime=1_500_000)
        (self.base / "results_empty").mkdir()
        (self.base / "other").mkdir()

    def tearDown(self):
        app.PIPELINE_DIR, app.RUNS_DIR, app.STATE, app.PIXEL_SIZE_UM = self.saved
        self.tmp.cleanup()

    def test_list_runs(self):
        runs = app.list_runs(self.base, self.runs)
        self.assertEqual([r["id"] for r in runs], ["runs/failed", "runs/new", "results_legacy", "runs/old"])
        new = runs[1]
        self.assertEqual(new["movies"], ["b", "c"])
        self.assertEqual(new["pixel_size_um"], 0.325)
        self.assertEqual(new["state"], "done")
        self.assertTrue(new["complete"])
        self.assertFalse(runs[0]["complete"])
        self.assertEqual(runs[0]["state"], "error")
        self.assertIsNone(runs[2]["state"])
        # newest complete, non-failed run wins for --results auto
        self.assertEqual(app.newest_complete_run(self.runs).name, "new")
        self.assertEqual(app.resolve_results("auto", self.runs).name, "new")

    def test_run_ids(self):
        self.assertEqual(app.run_id_for(self.runs / "new"), "runs/new")
        self.assertEqual(app.run_id_for(self.base / "results_legacy"), "results_legacy")
        self.assertEqual(app.resolve_run_id("runs/new"), self.runs / "new")
        self.assertEqual(app.resolve_run_id("results_legacy"), self.base / "results_legacy")
        for bad in ("runs/../results_legacy", "other", "runs/missing", "runs/", "results_nope", "../x"):
            with self.assertRaises(app.HttpError, msg=bad):
                app.resolve_run_id(bad)

    def test_select_switches_state(self):
        app.STATE = None
        st = app.select_results(self.runs / "old")
        self.assertEqual(list(st.movies), ["a"])
        self.assertEqual(st.run_id, "runs/old")
        self.assertAlmostEqual(app.PIXEL_SIZE_UM, 0.63)
        self.assertIsNotNone(st.loc("a"))
        st.retracks["a"] = {"x": 1}
        st.frame_cache.put("k", b"x")
        app.select_results(self.runs / "new")
        self.assertIs(app.STATE, st)                       # same object, new contents
        self.assertEqual(list(st.movies), ["b", "c"])
        self.assertAlmostEqual(app.PIXEL_SIZE_UM, 0.325)
        self.assertEqual(st.retracks, {})
        self.assertIsNone(st.frame_cache.get("k"))
        self.assertEqual(st._loc_cache, {})
        self.assertIsNone(st.loc("a"))
        self.assertEqual(st.loc("b")["n"], 3)

    def test_old_z_convention_is_flipped_on_loading(self):
        app.STATE = None
        st = app.select_results(self.runs / "old")
        csv = next((self.runs / "old").glob("*_localizations.csv"))
        raw = app.load_localizations(csv)["data"]["zMicrons"]
        self.assertTrue(app.z_is_stack(self.runs / "old"))                 # no z_convention recorded
        np.testing.assert_array_equal(st.loc("a")["data"]["zMicrons"], -raw)
        (self.runs / "old" / "z_convention.json").write_text(json.dumps({"z_convention": "up"}))
        self.assertFalse(app.z_is_stack(self.runs / "old"))
        np.testing.assert_array_equal(st.loc("a")["data"]["zMicrons"], raw)

    def test_http_runs_and_select(self):
        import threading
        import urllib.request
        app.STATE = None
        app.select_results(self.runs / "old")
        srv = app.ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{srv.server_address[1]}"

        def call(path, body=None, headers=None):
            req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json", **(headers or {})})
            try:
                with urllib.request.urlopen(req, timeout=10) as r:
                    return r.status, json.loads(r.read())
            except urllib.error.HTTPError as e:
                return e.code, json.loads(e.read())
        try:
            code, r = call("/api/runs")
            self.assertEqual(code, 200)
            self.assertEqual(r["active"], "runs/old")
            self.assertIn("results_legacy", [x["id"] for x in r["runs"]])
            code, r = call("/api/runs/select", {"id": "results_legacy"})
            self.assertEqual((code, r["run"], r["movies"]), (200, "results_legacy", ["cells"]))
            code, r = call("/api/movies")
            self.assertEqual(r["run"], "results_legacy")
            self.assertAlmostEqual(r["pixel_size_um"], 0.5)
            self.assertEqual(call("/api/runs/select", {"id": "runs/failed"})[0], 409)   # no results yet
            self.assertEqual(call("/api/runs/select", {"id": "../etc"})[0], 404)
            # a page names its own run: served side by side, the default run is untouched
            code, r = call("/api/movies?run=runs%2Fold")
            self.assertEqual((code, r["run"]), (200, "runs/old"))
            self.assertEqual(call("/api/runs")[1]["active"], "results_legacy")
            self.assertEqual(call("/api/movies?run=results_legacy")[1]["movies"][0]["name"], "cells")
            self.assertEqual(call("/api/movies?run=..%2Fetc")[0], 404)
            self.assertEqual(call("/api/movies?run=runs%2Ffailed")[0], 409)
            code, r = call("/api/browse?path=" + urllib.request.quote(str(self.base)))
            self.assertEqual(code, 200)
            self.assertIn("runs", r["dirs"])
            self.assertEqual(call("/api/browse?path=" + urllib.request.quote(str(self.base / "zzz")))[0], 404)
            # requests from other sites are refused: another host name (DNS rebinding), a cross-site
            # page (form, <img>), a foreign Origin; the app's own requests pass
            browse = "/api/browse?path=" + urllib.request.quote(str(self.base))
            self.assertEqual(call(browse, headers={"Host": "evil.example:80"})[0], 403)
            self.assertEqual(call(browse, headers={"Sec-Fetch-Site": "cross-site"})[0], 403)
            self.assertEqual(call("/api/runs/select", {"id": "runs/old"}, {"Origin": "https://evil.example"})[0], 403)
            self.assertEqual(call("/api/runs/select", {"id": "runs/old"}, {"Origin": "null"})[0], 403)
            self.assertEqual(call(browse, headers={"Sec-Fetch-Site": "same-origin", "Host": "localhost"})[0], 200)
            self.assertEqual(call("/api/runs/select", {"id": "runs/old"}, {"Origin": base, "Sec-Fetch-Site": "same-origin"})[0], 200)
        finally:
            srv.shutdown()
            srv.server_close()


# A stand-in for analyze.py: honours the CLI contract, writes status.json, finishes quickly.
# Behaviour is chosen by the run name: *slow* (runs ~30 s), *crash* (exits 3 mid-way).
FAKE_ENGINE = r'''
import argparse, json, os, sys, time
if "--inspect" in sys.argv:
    print("some log noise")
    print(json.dumps({"path": sys.argv[-1], "frames": 7, "height": 16, "width": 20, "dtype": "uint16",
                      "objective_guess": "20x", "pixel_size_guess_um": 0.325, "micromanager_roi": None,
                      "suggested_planes": [2, 6]}))
    sys.exit(0)
if "--preview" in sys.argv:
    path = sys.argv[sys.argv.index("--preview") + 1]
    frame = int(sys.argv[sys.argv.index("--frame") + 1])
    if frame > 7:
        print("frame out of range", file=sys.stderr)
        sys.exit(2)
    beads = [{"x1": 5, "y1": 5, "x2": 9, "y2": 6, "x": 7, "y": 5.5, "angle": 14, "sep": 4.1, "snr": s}
             for s in (2.0, 5.5, float("nan"))]
    print("detecting...")
    print(json.dumps({"frame": frame, "width": 20, "height": 16, "noise": 3.2, "beads": beads,
                      "suggested_min_snr": 3.5, "default_min_snr": 4.0}))
    sys.exit(0)
ap = argparse.ArgumentParser()
ap.add_argument("--calibration"); ap.add_argument("--movie", action="append"); ap.add_argument("--output")
ap.add_argument("--status-file"); ap.add_argument("--pixel-size", type=float); ap.add_argument("--z-step", type=float)
ap.add_argument("--z-range", type=float); ap.add_argument("--name"); ap.add_argument("--no-matlab", action="store_true")
ap.add_argument("--calibration-planes"); ap.add_argument("--min-snr", type=float)
a = ap.parse_args()
def status(state, stage, p, err=None):
    tmp = a.status_file + ".tmp"
    with open(tmp, "w") as fh:
        json.dump({"state": state, "stage": stage, "progress": p, "message": stage, "started": "x",
                   "updated": "y", "output": a.output, "error": err}, fh)
    for k in range(50):   # Windows: fails while the explorer is reading the file
        try:
            os.replace(tmp, a.status_file)
            return
        except PermissionError:
            time.sleep(0.01)
print("args", json.dumps(sys.argv[1:]), flush=True)
steps = 300 if "slow" in a.name else 3
for i in range(steps):
    status("running", "stage %d" % i, i / steps)
    print("step %d" % i, flush=True)
    time.sleep(0.1)
    if "crash" in a.name and i == 1:
        print("Traceback: boom", file=sys.stderr, flush=True)
        sys.exit(3)
stem = os.path.splitext(os.path.basename(a.movie[0]))[0]
with open(os.path.join(a.output, "run_info.json"), "w") as fh:
    json.dump({"pixel_size_um": a.pixel_size, "movies": {stem: {"path": a.movie[0]}}}, fh)
with open(os.path.join(a.output, stem + "_localizations.csv"), "w") as fh:
    fh.write("x1,x2,xMean,y1,y2,yMean,angleDegrees,zMicrons,frame_number,track_number\n1,2,1.5,1,1,1,0,0,1,1\n")
if "nodone" not in a.name:   # *nodone*: the final status write is lost (e.g. a locked file)
    status("done", "done", 1.0)
'''


class JobTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.fake = self.d / "fake_analyze.py"
        self.fake.write_text(FAKE_ENGINE, encoding="utf-8")
        for n in ("cal.tif", "movie_MMStack_Pos0.ome.tif"):
            (self.d / n).write_bytes(b"\0")
        self.jm = app.JobManager(runs_dir=self.d / "runs", command=[sys.executable, str(self.fake)])
        self.body = {"calibration": str(self.d / "cal.tif"), "movies": [str(self.d / "movie_MMStack_Pos0.ome.tif")],
                     "pixel_size_um": 0.63, "z_step_um": 1, "z_range_um": None, "name": None, "matlab": False}

    def tearDown(self):
        for j in self.jm.jobs.values():
            if j.alive:
                app.kill_tree(j.proc)
                j.proc.wait(10)
            self.jm._finalize(j)
        self.tmp.cleanup()

    def wait(self, job_id, timeout=30):
        import time
        t0 = time.time()
        while time.time() - t0 < timeout:
            st = self.jm.status(job_id)
            if st["state"] != "running":
                return st
            time.sleep(0.1)
        self.fail("job did not finish")

    def test_job_runs_to_done(self):
        r = self.jm.start(self.body)
        out = Path(r["output"])
        self.assertEqual(out.parent, self.d / "runs")
        self.assertTrue(out.name.startswith("movie_"), out.name)       # default name from the movie stem
        self.assertEqual(r["run_id"], "runs/" + out.name)
        st = self.wait(r["job_id"])
        self.assertEqual(st["state"], "done")
        self.assertEqual(st["progress"], 1.0)
        self.assertFalse(st["alive"])
        self.assertTrue((out / "run_info.json").is_file())
        log = (out / "analyze.log").read_text(encoding="utf-8")
        self.assertIn("--no-matlab", log)
        self.assertIn("--pixel-size", log)
        self.assertNotIn("--z-range", log)
        self.assertIn("step 2", "\n".join(st["log_tail"]))
        self.assertTrue(app.has_results(out))
        self.assertNotIn("--calibration-planes", log)
        self.assertNotIn("--min-snr", log)
        # a second run with the same name gets its own folder; two movies -> repeated --movie
        (self.d / "movie2.tif").write_bytes(b"\0")
        body = dict(self.body, name=out.name, z_range_um=5, calibration_planes=[40, 200], min_snr=3.5,
                    movies=self.body["movies"] + [str(self.d / "movie2.tif")])
        r2 = self.jm.start(body)
        self.assertEqual(Path(r2["output"]).name, out.name + "_2")
        self.assertEqual(self.wait(r2["job_id"])["state"], "done")
        log2 = (Path(r2["output"]) / "analyze.log").read_text(encoding="utf-8")
        self.assertIn("--z-range 5.0", log2)
        self.assertIn("--calibration-planes 40:200", log2)
        self.assertIn("--min-snr 3.5", log2)
        self.assertEqual(log2.splitlines()[0].count("--movie "), 2)

    def test_one_job_at_a_time_and_cancel(self):
        r = self.jm.start(dict(self.body, name="slow_run"))
        with self.assertRaises(app.JobBusy):
            self.jm.start(dict(self.body, name="another"))
        import time
        time.sleep(0.5)
        self.assertEqual(self.jm.current()["job_id"], r["job_id"])
        st = self.jm.cancel(r["job_id"])
        self.assertEqual(st["state"], "cancelled")
        self.assertFalse(st["alive"])
        saved = json.loads((Path(r["output"]) / "status.json").read_text())
        self.assertEqual(saved["state"], "cancelled")
        r2 = self.jm.start(dict(self.body, name="after_cancel"))        # free again
        self.assertEqual(self.wait(r2["job_id"])["state"], "done")

    def test_reopened_explorer_finds_and_cancels_a_running_job(self):
        # the analysis keeps running when the explorer closes; a new explorer session (a new JobManager on the
        # same runs folder) re-attaches to it from job.json, can cancel it, and marks dead ones interrupted
        import time
        r = self.jm.start(dict(self.body, name="slow_run"))
        time.sleep(0.5)
        self.assertTrue((Path(r["output"]) / "job.json").is_file())
        jm2 = app.JobManager(runs_dir=self.d / "runs", command=[sys.executable, str(self.fake)])
        cur = jm2.current()
        self.assertEqual((cur["job_id"], cur["state"], cur["alive"]), (r["job_id"], "running", True))
        with self.assertRaises(app.JobBusy):                               # still one analysis at a time
            jm2.start(dict(self.body, name="another"))
        st = jm2.cancel(r["job_id"])
        self.assertEqual((st["state"], st["alive"]), ("cancelled", False))
        self.assertFalse(self.jm.jobs[r["job_id"]].alive)                  # the process is really gone
        # a job whose process ended without a final state (e.g. the computer restarted): interrupted
        out = self.d / "runs" / "dead_run"
        out.mkdir()
        (out / "status.json").write_text(json.dumps({"state": "running", "progress": 0.4}))
        (out / "job.json").write_text(json.dumps({"job_id": "dead", "pid": 4, "pid_started": 1.0}))   # 4: not this job
        jm3 = app.JobManager(runs_dir=self.d / "runs", command=[sys.executable, str(self.fake)])
        jm3.adopt(force=True)
        saved = json.loads((out / "status.json").read_text())
        self.assertEqual(saved["state"], "error")
        self.assertIn("interrupted", saved["message"])

    def test_crash_reports_error_with_log(self):
        r = self.jm.start(dict(self.body, name="crash_run"))
        st = self.wait(r["job_id"])
        self.assertEqual(st["state"], "error")
        self.assertIn("exited (code 3)", st["message"])
        self.assertIn("boom", st["error"])
        self.assertEqual(json.loads((Path(r["output"]) / "status.json").read_text())["state"], "error")

    def test_lost_final_status_still_done(self):
        r = self.jm.start(dict(self.body, name="nodone_run"))
        st = self.wait(r["job_id"])
        self.assertEqual(st["state"], "done")          # exit 0 + run_info.json
        self.assertEqual(json.loads((Path(r["output"]) / "status.json").read_text())["state"], "done")

    def test_validation(self):
        with self.assertRaises(ValueError):
            self.jm.start(dict(self.body, calibration=str(self.d / "missing.tif")))
        with self.assertRaises(ValueError):
            self.jm.start(dict(self.body, movies=[]))
        with self.assertRaises(ValueError):
            self.jm.start(dict(self.body, pixel_size_um=-1))
        with self.assertRaises(ValueError):
            self.jm.start(dict(self.body, z_step_um="abc"))
        for planes in ([5, 2], [0, 3], "a:b", [1]):
            with self.assertRaises(ValueError, msg=str(planes)):
                self.jm.start(dict(self.body, calibration_planes=planes))
        with self.assertRaises(ValueError):
            self.jm.start(dict(self.body, min_snr=0))
        self.assertEqual(self.jm.validate(dict(self.body, calibration_planes="3:9"))["calibration_planes"], (3, 9))
        with self.assertRaises(app.HttpError):
            self.jm.status("nope")
        self.assertFalse((self.d / "runs").exists())                      # nothing created


if __name__ == "__main__":
    unittest.main(verbosity=2)
