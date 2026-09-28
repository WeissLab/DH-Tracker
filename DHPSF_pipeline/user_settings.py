"""Per-user preferences of DH-Tracker-2026: the default data folder and the results folder.

Kept in the user's profile (Windows: %APPDATA%\\DH-Tracker\\settings.json), not in the program
folder, so they are remembered between sessions and survive installing a newer version.

    python user_settings.py --choose     folder dialogs (Setup DH-Tracker-2026.bat runs this once)
    python user_settings.py              show the current settings
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

PROGRAM_DIR = Path(__file__).resolve().parent
LEGACY_RESULTS = PROGRAM_DIR / 'runs'          # where results went before this setting existed


def settings_path() -> Path:
    if os.environ.get('DHTRACKER_SETTINGS'):    # tests, or a shared lab configuration
        return Path(os.environ['DHTRACKER_SETTINGS'])
    base = os.environ.get('APPDATA') or os.environ.get('XDG_CONFIG_HOME') or str(Path.home() / '.config')
    return Path(base) / 'DH-Tracker' / 'settings.json'


def load() -> dict:
    try:
        d = json.loads(settings_path().read_text(encoding='utf-8'))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def save(**changes) -> dict:
    """Update the given settings (None removes one) and write the file; returns all settings."""
    d = load()
    for k, v in changes.items():
        if v is None:
            d.pop(k, None)
        else:
            d[k] = str(v)
    p = settings_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(d, indent=2), encoding='utf-8')
    os.replace(tmp, p)
    return d


def default_results_dir() -> Path:
    return Path.home() / 'Documents' / 'DH-Tracker results'


def results_dir() -> Path:
    """Where new analyses are written: the chosen folder, else the program's own runs/ folder."""
    p = load().get('results_dir')
    return Path(p) if p else LEGACY_RESULTS


def data_dir() -> Path | None:
    """The folder the file picker opens in first (None: not chosen, or no longer there)."""
    p = load().get('data_dir')
    return Path(p) if p and Path(p).is_dir() else None


def ask_folder(title: str, initial: Path | None) -> Path | None:
    """A folder dialog (None if cancelled or no display)."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return None
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)        # in front of the browser / console
    try:
        p = filedialog.askdirectory(title=title, initialdir=str(initial) if initial and initial.is_dir() else None,
                                    mustexist=False, parent=root)
    finally:
        root.destroy()
    return Path(p) if p else None


def choose(force: bool = False) -> dict:
    """Ask for both folders (only those not chosen yet, unless force)."""
    d = load()
    if force or 'data_dir' not in d:
        print('Choose the folder that holds your movies (TIFF files); the file picker will open there.', flush=True)
        p = ask_folder('DH-Tracker-2026: folder with your movies (TIFF files)', Path.home() / 'Documents')
        if p:
            d = save(data_dir=p)
    if force or 'results_dir' not in d:
        print(f'Choose where results are saved (suggested: {default_results_dir()}).', flush=True)
        p = ask_folder('DH-Tracker-2026: where to save results (Cancel = Documents\\DH-Tracker results)',
                       Path.home() / 'Documents')
        p = p or default_results_dir()
        p.mkdir(parents=True, exist_ok=True)
        d = save(results_dir=p)
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--choose', action='store_true', help='ask for the folders not chosen yet')
    ap.add_argument('--change', action='store_true', help='ask for both folders again')
    a = ap.parse_args(argv)
    if a.choose or a.change:
        choose(force=a.change)
    print(f'Settings file: {settings_path()}')
    print(f'  movies (file picker opens here): {data_dir() or "not chosen"}')
    print(f'  results are saved in:            {results_dir()}')


if __name__ == '__main__':
    main()
