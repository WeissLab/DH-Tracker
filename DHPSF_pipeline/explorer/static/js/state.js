// Global application state (single mutable object) -- modules communicate via util.emit/on.
// µm per pixel of the current movie (live binding; set from the server's movie info).
export let PX_UM = 0.325;
export function setPxUm(value) { if (Number.isFinite(value) && value > 0) PX_UM = value; }

// Every data request names this page's run, so pages showing different runs do not interfere
// (the server's "active" run is only the default for a newly opened page).
export function withRun(url) {
  if (!S.run || !url.startsWith('/api/') || url.startsWith('/api/runs') || /[?&]run=/.test(url)) return url;
  return `${url}${url.includes('?') ? '&' : '?'}run=${encodeURIComponent(S.run)}`;
}

export const S = {
  runs: [],             // result sets from /api/runs
  run: null,            // active run id (e.g. 'results_v2', 'runs/<name>')
  starred: new Set(),   // starred track ids (pipeline track numbers; saved per run + movie in this browser)
  movies: [],
  movie: null,          // name
  info: null,           // /info payload
  frame: 1,             // current frame, 1-based (matches frame_number)
  nFrames: 1,
  playing: false,
  fps: 10,
  loop: true,
  contrast: { vmin: 100, vmax: 300, gamma: 1 },
  json: null,           // raw localization table from the server (column arrays)
  ds: null,             // dataset (data.js) -- main tracks
  rej: null,            // rejected localizations ({movie}_rejected.csv), optional
  shortDs: null,        // localizations of tracks shorter than min length after re-tracking
  drift: null,          // drift table ({movie}_drift.csv), optional
  coords: { corr: false, xy: false, z: false },   // lateral correction, XY and Z stabilization
  stab: false,          // derived: XY stabilization active (frames are shifted)
  tracking: null,       // re-tracking result {summary, params, ids, short} or null (pipeline tracks)
  F: null,              // filter result (data.js)
  filters: { minLen: 1, zMin: null, zMax: null, validOnly: false, recovered: 'all', fStart: 1, fEnd: 1, minBright: 0, motion: 'all' },
  sel: new Set(),       // selected track ids
  hover: null,          // hovered row index
  tool: 'pan',          // pan | rect | lasso | poly
  roiMode: 'any',       // any | current
  rois: [],             // [{type:'rect'|'poly', pts:[[x,y],...] (one-based px, image plane)}]
  ov: {
    show: true, beads: true, lobes: true, colorBy: 'z', recoveredDashed: true, rejected: false, short: true,
    tails: true, tailN: 10, onlySel: false, dimUnsel: true, legend: true, scalebar: true,
    zcAuto: true, zcMin: -5, zcMax: 5,
  },
  disp: { on: false, a: 1, b: 1, followB: false, scale: 5, zmap: false, zmapAt: 'b', onlySel: false },
  v3: { colorBy: 'frame', units: 'px', yDown: false, onlySel: false, dim: true, markers: true },
  ui: { drawer: false, panel: true, tab: '3d' },
  exportApplyFilters: true,
};
