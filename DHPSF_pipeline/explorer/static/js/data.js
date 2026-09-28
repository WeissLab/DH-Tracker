// Dataset construction, filtering and per-track statistics.
import { PX_UM } from './state.js';
import { lowerBound, percentile } from './util.js';

export const OPTIONAL = ['residualRMS', 'minLobeSNR', 'jointEmitterCount', 'lobeSeparationPixels', 'zStatus', 'recovered', 'trackLength', 'flag',
  'pMoving', 'motionClass', 'framesMoving', 'speedWhileMoving', 'dTrack', 'stage', 'refDx', 'refDy', 'refDz', 'refSdX', 'refSdY', 'refSdZ',
  'xPrecisionPx', 'yPrecisionPx', 'zPrecisionUm'];
export const MOTION_FILTERS = { all: null, directed: 2, confined: 1, brownian: 0 };

export function buildDataset(json, nFramesMovie) {
  const n = json.n;
  const cols = {};
  for (const c of json.columns) {
    const src = json.data[c];
    const a = new Float64Array(n);
    for (let i = 0; i < n; i++) { const v = src[i]; a[i] = v === null ? NaN : v; }
    cols[c] = a;
  }
  const has = {};
  for (const c of OPTIONAL) has[c] = c in cols;

  const frame = new Int32Array(n), tid = new Float64Array(n);
  let maxFrame = 1;
  for (let i = 0; i < n; i++) {
    frame[i] = Math.round(cols.frame_number[i]);
    tid[i] = cols.track_number[i];
    if (frame[i] > maxFrame) maxFrame = frame[i];
  }
  const nFrames = Math.max(maxFrame, nFramesMovie || 1);

  // tracks
  const ids = Array.from(new Set(Array.from(tid).filter(Number.isFinite))).sort((a, b) => a - b);
  const kOf = new Map(ids.map((id, k) => [id, k]));
  const rowTrack = new Int32Array(n).fill(-1);
  const counts = new Int32Array(ids.length);
  for (let i = 0; i < n; i++) { const k = kOf.get(tid[i]); if (k !== undefined) { rowTrack[i] = k; counts[k]++; } }
  const trackRows = ids.map((_, k) => new Int32Array(counts[k]));
  const fill = new Int32Array(ids.length);
  for (let i = 0; i < n; i++) { const k = rowTrack[i]; if (k >= 0) trackRows[k][fill[k]++] = i; }
  for (const r of trackRows) r.sort((a, b) => frame[a] - frame[b] || a - b);

  // per-frame CSR index
  const fcount = new Int32Array(nFrames + 2);
  for (let i = 0; i < n; i++) if (frame[i] >= 1 && frame[i] <= nFrames) fcount[frame[i]]++;
  const fstart = new Int32Array(nFrames + 2);
  for (let f = 1; f <= nFrames + 1; f++) fstart[f] = fstart[f - 1] + (f - 1 >= 1 ? fcount[f - 1] : 0);
  // fstart[f] = start of frame f ; rows of frame f are frameRows[fstart[f] .. fstart[f+1])
  const frameRows = new Int32Array(fstart[nFrames + 1]);
  const fp = fstart.slice();
  for (let i = 0; i < n; i++) if (frame[i] >= 1 && frame[i] <= nFrames) frameRows[fp[frame[i]]++] = i;

  const zValid = new Uint8Array(n);
  const z = cols.zMicrons;
  for (let i = 0; i < n; i++) zValid[i] = Number.isFinite(z[i]) && (!has.zStatus || cols.zStatus[i] === 0) ? 1 : 0;
  const recovered = new Uint8Array(n);
  if (has.recovered) for (let i = 0; i < n; i++) recovered[i] = cols.recovered[i] > 0 ? 1 : 0;
  has.stabCols = 'xStabilized' in cols && 'yStabilized' in cols;
  has.corrCols = 'xCorrected' in cols && 'yCorrected' in cols;

  const ds = {
    n, cols, columns: json.columns, source: json.source, has, frame, tid, ids, kOf, rowTrack, trackRows,
    nFrames, fstart, frameRows, zValid, recovered,
    orig: json.orig || null,                 // row index into the server CSV (subset datasets)
    pipelineTid: json.pipelineTid || null,   // original track ids when re-tracked
  };
  ds.bright = trackBrightness(ds);
  has.bright = ds.bright.some(Number.isFinite);
  setCoords(ds, 'raw', null);
  return ds;
}

// Per-track relative brightness (%): the track's median weaker-lobe amplitude divided by the median
// of the tracks around it (within `radius` px). Side-lobe ghosts and noise fits are far dimmer than
// their neighbourhood; a relative measure copes with uneven illumination across the field.
export function trackBrightness(ds, radius = 150) {
  const { cols, trackRows } = ds;
  const nT = trackRows.length, out = new Float64Array(nT).fill(NaN);
  const a1 = cols.amplitude1, a2 = cols.amplitude2, snr = cols.minLobeSNR;
  if (!(a1 && a2) && !snr) return out;
  const med = (arr) => { const a = arr.filter(Number.isFinite).sort((u, v) => u - v); return a.length ? a[a.length >> 1] : NaN; };
  const amp = new Float64Array(nT), cx = new Float64Array(nT), cy = new Float64Array(nT);
  for (let k = 0; k < nT; k++) {
    const r = Array.from(trackRows[k]);
    amp[k] = med(r.map((i) => (a1 && a2 ? Math.min(a1[i], a2[i]) : snr[i])));
    cx[k] = med(r.map((i) => cols.xMean[i])); cy[k] = med(r.map((i) => cols.yMean[i]));
  }
  // uniform grid of track centres for the neighbour search
  const cell = new Map(), key = (i, j) => i * 100003 + j;
  for (let k = 0; k < nT; k++) {
    if (!Number.isFinite(amp[k])) continue;
    const q = key(Math.floor(cx[k] / radius), Math.floor(cy[k] / radius));
    if (!cell.has(q)) cell.set(q, []);
    cell.get(q).push(k);
  }
  const global = med(Array.from(amp));
  for (let k = 0; k < nT; k++) {
    if (!Number.isFinite(amp[k])) continue;
    const ci = Math.floor(cx[k] / radius), cj = Math.floor(cy[k] / radius), nb = [];
    for (let di = -1; di <= 1; di++) for (let dj = -1; dj <= 1; dj++) {
      for (const m of cell.get(key(ci + di, cj + dj)) || []) {
        if (m !== k && Math.hypot(cx[m] - cx[k], cy[m] - cy[k]) <= radius) nb.push(amp[m]);
      }
    }
    const ref = nb.length >= 3 ? med(nb) : global;
    out[k] = ref > 0 ? 100 * amp[k] / ref : NaN;
  }
  return out;
}
// row index in the server's CSV (for exports)
export const origRow = (ds, r) => (ds.orig ? ds.orig[r] : r);

// Coordinates. Three independent switches, coords = {corr, xy, z}:
//    corr -> xCorrected, yCorrected (lateral shift-with-z removed) instead of xMean, yMean
//    xy   -> minus the frame's whole-field XY drift (dx, dy); the movie frames are shifted too
//    z    -> zMicrons minus the frame's whole-field z drift (dz)
//    refined -> plus the motion analysis' refined-position shift (Kalman smoother over the whole
//               track, each track's own motion model); the SDs then come from the smoother too
//  The analysis coordinates X, Y (one-based px) and Z (µm) drive stats, 3D, plots, displacement.
//  Drawing coordinates ux.. (image plane, 0-based pixel j spans [j, j+1], u = x - 0.5) are always
//  the RAW lobe/centre positions (that is where the PSF is), minus the frame XY drift when the
//  XY-stabilized (shifted) frames are shown.
export function coordLabel(c) {
  const stab = [c.xy && 'XY', c.z && 'Z'].filter(Boolean);
  return (c.corr ? 'lateral-corrected' : 'raw') + (stab.length ? ` · ${stab.join(' + ')} stabilized` : '') + (c.refined ? ' · refined' : '');
}

export function setCoords(ds, coords, drift) {
  const { n, cols, frame, has } = ds;
  const c = { corr: !!coords.corr && has.corrCols, xy: !!coords.xy && !!drift, z: !!coords.z && !!drift,
    refined: !!coords.refined && !!has.refDx };
  ds.coords = c;
  ds.coordLabel = coordLabel(c);
  ds.stabilized = c.xy;
  const fdx = new Float64Array(n), fdy = new Float64Array(n), fdz = new Float64Array(n);
  if (c.xy || c.z) for (let i = 0; i < n; i++) { fdx[i] = drift.dxOf(frame[i]); fdy[i] = drift.dyOf(frame[i]); fdz[i] = drift.dzOf(frame[i]); }
  const baseX = c.corr ? cols.xCorrected : cols.xMean;
  const baseY = c.corr ? cols.yCorrected : cols.yMean;
  const X = new Float64Array(n), Y = new Float64Array(n), Z = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let x = baseX[i], y = baseY[i];
    if (!Number.isFinite(x)) x = cols.xMean[i];
    if (!Number.isFinite(y)) y = cols.yMean[i];
    X[i] = c.xy ? x - fdx[i] : x;
    Y[i] = c.xy ? y - fdy[i] : y;
    Z[i] = c.z ? cols.zMicrons[i] - fdz[i] : cols.zMicrons[i];
    if (c.refined && Number.isFinite(cols.refDx[i])) { X[i] += cols.refDx[i]; Y[i] += cols.refDy[i]; Z[i] += cols.refDz[i]; }
  }
  ds.X = X; ds.Y = Y; ds.Z = Z;
  // per-localization SDs of X, Y (px) and Z (µm): the smoother's when refined, else the reported precision
  const pick = (ref, rep, fallback) => {
    const o = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const v = c.refined && has[ref] ? cols[ref][i] : has[rep] ? cols[rep][i] : NaN;
      o[i] = Number.isFinite(v) && v > 0 ? v : fallback;
    }
    return o;
  };
  ds.sdX = pick('refSdX', 'xPrecisionPx', 0.1); ds.sdY = pick('refSdY', 'yPrecisionPx', 0.1); ds.sdZ = pick('refSdZ', 'zPrecisionUm', 0.5);
  if (!c.xy) { fdx.fill(0); fdy.fill(0); }   // drawing is shifted only with the XY-stabilized frames
  const sub = (a, d) => { const o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = a[i] - d[i] - 0.5; return o; };
  ds.ux = sub(cols.xMean, fdx); ds.uy = sub(cols.yMean, fdy);
  ds.ux1 = sub(cols.x1, fdx); ds.uy1 = sub(cols.y1, fdy);
  ds.ux2 = sub(cols.x2, fdx); ds.uy2 = sub(cols.y2, fdy);
  ds.zAuto = [percentile(ds.Z, 1), percentile(ds.Z, 99)];
}

// build a JSON-like column table for a subset of rows (optionally overriding track ids)
export function subsetJson(json, rows, trackIds = null) {
  const data = {};
  for (const c of json.columns) { const src = json.data[c]; data[c] = Array.from(rows, (r) => src[r]); }
  let pipelineTid = null;
  if (trackIds) { pipelineTid = Float64Array.from(rows, (r) => json.data.track_number[r]); data.track_number = Array.from(rows, (r) => trackIds[r]); }
  return { columns: json.columns, n: rows.length, source: json.source, data, orig: Int32Array.from(rows), pipelineTid };
}

// drift table -> fast per-frame lookup
// Z drift (focus) is smooth in time; frame-to-frame dz is measurement noise. Results written
// before the pipeline smoothed it (no dzRaw column) are smoothed here the same way.
export const DRIFT_Z_SIGMA = 3;
export function smoothSeries(v, sigma) {
  const n = v.length, out = new Array(n), r = Math.ceil(3 * sigma);
  const w = Array.from({ length: 2 * r + 1 }, (_, i) => Math.exp(-0.5 * ((i - r) / sigma) ** 2));
  for (let i = 0; i < n; i++) {
    let s = 0, ws = 0;
    for (let k = -r; k <= r; k++) {
      const j = Math.min(n - 1, Math.max(0, i + k)), x = v[j];
      if (Number.isFinite(x)) { s += w[k + r] * x; ws += w[k + r]; }
    }
    out[i] = ws ? s / ws : NaN;
  }
  const z0 = percentile(out.filter(Number.isFinite), 50);   // referenced to its median, as the pipeline
  return Number.isFinite(z0) ? out.map((x) => x - z0) : out;
}

export function buildDrift(json) {
  if (json.dz && !json.dzRaw) { json = { ...json, dzRaw: json.dz, dz: smoothSeries(json.dz, DRIFT_Z_SIGMA) }; }
  const m = new Map();
  json.frame_number.forEach((f, i) => m.set(f, i));
  const get = (arr) => (f) => { const i = m.get(f); const v = i === undefined ? 0 : arr[i]; return Number.isFinite(v) ? v : 0; };
  return { ...json, dxOf: get(json.dx), dyOf: get(json.dy), dzOf: get(json.dz || []) };
}

export function applyFilters(ds, f) {
  const { n, frame, zValid, recovered, has } = ds;
  const z = ds.Z;
  const rowOk = new Uint8Array(n);
  const zMin = Number.isFinite(f.zMin) ? f.zMin : -Infinity, zMax = Number.isFinite(f.zMax) ? f.zMax : Infinity;
  const zRangeOn = zMin > -Infinity || zMax < Infinity;
  for (let i = 0; i < n; i++) {
    if (frame[i] < f.fStart || frame[i] > f.fEnd) continue;
    if (f.validOnly && !zValid[i]) continue;
    if (zRangeOn && !(z[i] >= zMin && z[i] <= zMax)) continue;
    if (has.recovered) {
      if (f.recovered === 'hide' && recovered[i]) continue;
      if (f.recovered === 'only' && !recovered[i]) continue;
    }
    rowOk[i] = 1;
  }
  const nT = ds.ids.length;
  const trackOk = new Uint8Array(nT);
  const tRows = new Array(nT), tFrames = new Array(nT), stats = new Array(nT);
  let nRows = 0, nTracks = 0;
  for (let k = 0; k < nT; k++) {
    const all = ds.trackRows[k];
    let m = 0;
    for (let j = 0; j < all.length; j++) if (rowOk[all[j]]) m++;
    const rows = new Int32Array(m);
    m = 0;
    for (let j = 0; j < all.length; j++) if (rowOk[all[j]]) rows[m++] = all[j];
    tRows[k] = rows;
    tFrames[k] = Int32Array.from(rows, (r) => frame[r]);
    const dim = f.minBright > 0 && !(ds.bright[k] >= f.minBright);
    let motionOk = true;
    if (f.motion && f.motion !== 'all' && ds.has.motionClass && all.length) {
      const r0 = all[0];
      motionOk = f.motion === 'moving' ? ds.cols.framesMoving[r0] > 0 : ds.cols.motionClass[r0] === MOTION_FILTERS[f.motion];
    }
    if (m >= Math.max(1, f.minLen) && !dim && motionOk) { trackOk[k] = 1; nTracks++; nRows += m; }
    else for (let j = 0; j < m; j++) rowOk[rows[j]] = 0;
    stats[k] = trackStats(ds, ds.ids[k], rows);
    stats[k].bright = ds.bright[k];
  }
  return { rowOk, trackOk, tRows, tFrames, stats, nRows, nTracks };
}

function trackStats(ds, id, rows) {
  const { frame, recovered, has } = ds;
  const m = rows.length;
  const s = { id, len: m, first: NaN, last: NaN, meanZ: NaN, zRange: NaN, net: NaN, maxStep: NaN, pctRec: has.recovered ? 0 : NaN };
  if (!m) return s;
  s.first = frame[rows[0]]; s.last = frame[rows[m - 1]];
  let zs = 0, zn = 0, zmin = Infinity, zmax = -Infinity, rec = 0, maxStep = 0;
  for (let j = 0; j < m; j++) {
    const r = rows[j];
    const z = ds.Z[r];
    if (Number.isFinite(z)) { zs += z; zn++; if (z < zmin) zmin = z; if (z > zmax) zmax = z; }
    if (recovered[r]) rec++;
    if (j > 0) {
      const d = dist3(ds, rows[j - 1], r);
      if (d > maxStep) maxStep = d;
    }
  }
  if (zn) { s.meanZ = zs / zn; s.zRange = zmax - zmin; }
  s.net = m > 1 ? dist3(ds, rows[0], rows[m - 1]) : 0;
  s.maxStep = m > 1 ? maxStep : 0;
  if (has.recovered) s.pctRec = 100 * rec / m;
  if (has.motionClass) {
    const c = ds.cols;
    s.motion = c.motionClass[rows[0]];
    s.speed = c.speedWhileMoving[rows[0]];
    let mv = 0;
    for (let j = 0; j < m; j++) if (c.pMoving[rows[j]] > 0.5) mv++;
    s.moving = mv;
  }
  return s;
}

// 3D distance (µm) between two rows; falls back to lateral distance when z missing
export function dist3(ds, a, b) {
  const dx = (ds.X[b] - ds.X[a]) * PX_UM, dy = (ds.Y[b] - ds.Y[a]) * PX_UM;
  let dz = ds.Z[b] - ds.Z[a];
  if (!Number.isFinite(dz)) dz = 0;
  return Math.hypot(dx, dy, dz);
}

// filtered row of track k at frame f, or -1
export function rowAt(F, k, f) {
  const fr = F.tFrames[k];
  const j = lowerBound(fr, f);
  return j < fr.length && fr[j] === f ? F.tRows[k][j] : -1;
}

// rows (filtered, passing) currently visible at frame f
export function* frameRowsOk(ds, F, f) {
  if (f < 1 || f > ds.nFrames) return;
  for (let p = ds.fstart[f]; p < ds.fstart[f + 1]; p++) {
    const r = ds.frameRows[p];
    if (F.rowOk[r]) yield r;
  }
}

// all filtered rows of passing tracks (optionally only a set of track ids)
export function filteredRows(ds, F, trackIdSet = null, applyFilters = true) {
  const out = [];
  for (let k = 0; k < ds.ids.length; k++) {
    if (trackIdSet && !trackIdSet.has(ds.ids[k])) continue;
    if (applyFilters) { if (!F.trackOk[k]) continue; for (const r of F.tRows[k]) out.push(r); }
    else for (const r of ds.trackRows[k]) out.push(r);
  }
  return out;
}
