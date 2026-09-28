// Plotly views: 3D tracks (MATLAB-plotter style) and per-track 2D plots.
//
// Performance: nothing is rendered while its tab is hidden (a dirty flag is kept and the
// view is rebuilt when shown); frame changes only restyle the small current-frame marker
// trace (throttled) and move DOM frame-cursor lines -- the figures are never rebuilt on a
// frame change. The large base 3D trace is cached and re-used across selection changes.
import { S, PX_UM } from './state.js';
import { on, emit, throttle, debounce, parula, turbo, hueScale, trackColor, trackHue01, trackShade, motionMap, stageColor } from './util.js';
import { rowAt } from './data.js';
import { shiftDown } from './movie.js';

const DARK = { paper: '#0f1115', plot: '#0f1115', grid: '#262a33', font: '#c9cdd4', zero: '#3a404b' };
// Plotly's camera ("download plot as png") button is replaced by a save (disk) icon doing the same
const SAVE_ICON = { width: 24, height: 24, path: 'M17 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V7l-4-4zm-5 16a3 3 0 1 1 0-6 3 3 0 0 1 0 6zm3-10H5V5h10v4z' };
const saveButton = {
  name: 'Save as PNG', icon: SAVE_ICON,
  click: (gd) => Plotly.downloadImage(gd, { format: 'png', width: 1600, height: gd.id === 'plot3d' ? 1200 : 700, filename: `${S.movie || 'plot'}_${gd.id.replace(/^plot/, '').toLowerCase()}` }),
};
const CFG = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ['toImage', 'sendDataToCloud', 'lasso2d', 'select2d', 'toggleSpikelines', 'hoverCompareCartesian'], modeBarButtonsToAdd: [saveButton] };
const MAX_PLOT_TRACKS = 30;

let div3, divZ, divXY, divD, divDrift, divM;
// User-set axis ranges (double-click an axis). 3D: keyed by display units / z convention, so a
// change of units drops them. 2D: per plot div id and axis.
let ranges3 = { key: null, x: null, y: null, z: null };
const ranges2 = {};
let baseCache = { F: null, ds: null, key: null, L: null, color: null };
const dirty = { d3: true, tracks: true, drift: true };
export const perf = { draw3D: [], tracks: [], marker: [] };
const note = (a, v) => { a.push(v); if (a.length > 60) a.shift(); };

const vis3 = () => S.ui.panel && S.ui.tab === '3d';
const visTracks = () => S.ui.panel && S.ui.tab === 'tracks';

export function initPlots() {
  div3 = document.getElementById('plot3d');
  divZ = document.getElementById('plotZ');
  divXY = document.getElementById('plotXY');
  divD = document.getElementById('plotD');
  divDrift = document.getElementById('plotDrift');
  divM = document.getElementById('plotMotion');
  bind3dNative();
  bindAxisRanges();
  const all =() => { dirty.d3 = dirty.tracks = dirty.drift = true; refresh(); };
  on('data', () => { baseCache = { F: null }; all(); });
  on('filter', all);
  on('view3d', () => { dirty.d3 = true; refresh(); });
  // Selection: let the movie / table repaint first, then update the (expensive, ~50-150 ms)
  // WebGL scene once for a burst of selection changes.
  let selTimer = null;
  on('selection', () => {
    dirty.d3 = dirty.tracks = true;
    clearTimeout(selTimer);
    selTimer = setTimeout(refresh, 40);
  });
  on('panel', refresh);
  // Frame changes: a scatter3d restyle re-renders the whole WebGL scene (~0.1 s), so the
  // current-frame markers follow playback at ~4 updates/s (throttled) and snap to the final
  // frame once scrubbing/playback settles (debounced) -- the movie never waits for the 3D view.
  const markerSettled = debounce(updateMarkers, 180);
  const markerDuringPlay = throttle(updateMarkers, 250);
  // The top-bar "Markers" toggle also shows/hides the 3D current-position spheres.
  let markersShown = S.ov.beads;
  on('overlay', () => {
    if (S.ov.beads === markersShown || !div3.data || div3.data.length < 3) return;
    markersShown = S.ov.beads;
    try { const cam = liveCamera(); Plotly.update(div3, { visible: markersShown }, cam ? { 'scene.camera': cam } : {}, [2]); } catch (e) { /* ignore */ }
    if (markersShown) updateMarkers();
  });
  on('frame', () => {
    if (vis3()) { if (S.playing) markerDuringPlay(); markerSettled(); } else dirty.marker = true;
    if (visTracks()) updateFrameLines();
  });
  for (const d of [div3, divZ, divXY, divD, divDrift, divM]) {
    new ResizeObserver(throttle(() => {
      if (!sized(d)) return;
      if (d._pending) { const p = d._pending; d._pending = null; safePlot(d, ...p); }
      else if (d.data) { try { Plotly.Plots.resize(d); } catch (e) { /* ignore */ } updateFrameLines(); }
    }, 150)).observe(d);
  }
}

// render whatever is visible and dirty
export function refresh() {
  if (!S.ds || !S.F || typeof Plotly === 'undefined') return;
  if (vis3()) {
    if (dirty.d3) draw3D();
    else if (dirty.marker) updateMarkers();
    dirty.marker = false;
  }
  if (visTracks()) {
    if (dirty.tracks) drawTrackPlots();
    if (dirty.drift) drawDriftPlot();
    updateFrameLines();
  }
}

// 3D picking. plotly_click on gl3d scenes is unreliable (fires only for some pointer
// sequences), so we also select the hovered track on a non-drag click. Handlers are
// deferred: re-plotting a gl3d scene inside its own click event re-fires the click.
let hovered3 = null, down3 = null, lastPick = { t: 0, id: null };
function pick3(id, additive) {
  if (id === null || id === undefined) return;
  const t = performance.now();
  if (id === lastPick.id && t - lastPick.t < 400) return;
  lastPick = { t, id };
  setTimeout(() => emit('pickTrack', { id, additive, source: '3d' }), 0);
}
function bind3dNative() {
  div3.addEventListener('mousedown', (e) => { down3 = [e.clientX, e.clientY]; }, true);
  div3.addEventListener('mouseup', (e) => {
    if (!down3 || e.button !== 0) return;
    const moved = Math.hypot(e.clientX - down3[0], e.clientY - down3[1]) > 4;
    down3 = null;
    if (!moved && hovered3 !== null && !(e.target.closest && e.target.closest('.modebar'))) pick3(hovered3, e.shiftKey || e.ctrlKey);
  }, true);
}

function bindClick(d) {
  if (d._bound || !d.on) return;
  d._bound = true;
  if (d === div3) {
    d.on('plotly_click', (ev) => {
      const p = ev.points && ev.points[0];
      if (!p || p.customdata === null || p.customdata === undefined) return;
      pick3(p.customdata, shiftDown);
    });
    d.on('plotly_hover', (ev) => { const p = ev.points && ev.points[0]; hovered3 = p && p.customdata !== undefined ? p.customdata : null; });
    d.on('plotly_unhover', () => { hovered3 = null; });
  } else {
    let last = { t: 0, f: null };
    d.on('plotly_click', (ev) => {
      const p = ev.points && ev.points[0];
      if (!p) return;
      const f = Math.round(p.x), t = performance.now();
      if (f === last.f && t - last.t < 400) return;
      last = { t, f };
      setTimeout(() => emit('setFrame', f), 0);
    });
    d.on('plotly_relayout', () => updateFrameLines());
  }
}

// ------------------------------------------------------------------ axis ranges
// Double-click an axis to type its range. 3D: the click is matched to the nearest projected
// edge of the scene box (tick labels sit just outside it). 2D: Plotly's axis drag strips.
function mulVec(m, v) {   // column-major 4x4 (gl-matrix) times [x, y, z, w]
  return [0, 1, 2, 3].map((r) => m[r] * v[0] + m[4 + r] * v[1] + m[8 + r] * v[2] + m[12 + r] * v[3]);
}
function sceneAxisAt(clientX, clientY) {
  const sc = div3._fullLayout && div3._fullLayout.scene && div3._fullLayout.scene._scene;
  const g = sc && sc.glplot;
  if (!g || !g.cameraParams || !g.bounds) return null;
  const { model, view, projection } = g.cameraParams;
  const rect = g.canvas.getBoundingClientRect();
  if (!rect.width) return null;
  const [lo, hi] = g.bounds;
  const toScreen = (p) => {
    const c = mulVec(projection, mulVec(view, mulVec(model, [p[0], p[1], p[2], 1])));
    return [rect.left + (c[0] / c[3] + 1) / 2 * rect.width, rect.top + (1 - c[1] / c[3]) / 2 * rect.height];
  };
  const corner = (i) => [i & 1 ? hi[0] : lo[0], i & 2 ? hi[1] : lo[1], i & 4 ? hi[2] : lo[2]];
  let best = null;
  for (let i = 0; i < 8; i++) {
    for (let a = 0; a < 3; a++) {
      if (i & (1 << a)) continue;
      const p = toScreen(corner(i)), q = toScreen(corner(i | (1 << a)));
      const dx = q[0] - p[0], dy = q[1] - p[1], L2 = dx * dx + dy * dy || 1;
      const t = Math.max(0, Math.min(1, ((clientX - p[0]) * dx + (clientY - p[1]) * dy) / L2));
      const d = Math.hypot(clientX - p[0] - t * dx, clientY - p[1] - t * dy);
      if (!best || d < best.d) best = { d, axis: 'xyz'[a] };
    }
  }
  return best && best.d <= 45 ? best.axis : null;
}

let pop = null;
function rangePopover(clientX, clientY, title, current, apply) {
  if (!pop) {
    pop = document.createElement('div');
    pop.className = 'axisPop';
    pop.innerHTML = `<div class="axisPopTitle"></div>
      <div class="axisPopRow"><input type="number" step="any" data-k="lo" title="minimum"><span>to</span><input type="number" step="any" data-k="hi" title="maximum"></div>
      <div class="axisPopRow"><button data-k="auto" title="Automatic range">auto</button><span class="spacer"></span><button data-k="ok" class="primary">set</button></div>`;
    document.body.appendChild(pop);
    document.addEventListener('mousedown', (e) => { if (pop.style.display !== 'none' && !pop.contains(e.target)) pop.style.display = 'none'; }, true);
  }
  const [loIn, hiIn] = pop.querySelectorAll('input');
  const fmt = (v) => (Number.isFinite(v) ? +v.toPrecision(5) : '');
  pop.querySelector('.axisPopTitle').textContent = title;
  loIn.value = fmt(Math.min(...current)); hiIn.value = fmt(Math.max(...current));
  const close = () => { pop.style.display = 'none'; };
  const set = () => {
    const a = parseFloat(loIn.value), b = parseFloat(hiIn.value);
    if (!Number.isFinite(a) || !Number.isFinite(b) || a === b) { loIn.focus(); return; }
    apply([Math.min(a, b), Math.max(a, b)]); close();
  };
  pop.querySelector('[data-k="ok"]').onclick = set;
  pop.querySelector('[data-k="auto"]').onclick = () => { apply(null); close(); };
  pop.onkeydown = (e) => { if (e.key === 'Enter') set(); else if (e.key === 'Escape') close(); e.stopPropagation(); };
  pop.style.display = 'flex';
  const w = pop.offsetWidth, h = pop.offsetHeight;
  pop.style.left = `${Math.max(4, Math.min(clientX - w / 2, innerWidth - w - 4))}px`;
  pop.style.top = `${Math.max(4, Math.min(clientY + 12, innerHeight - h - 4))}px`;
  loIn.focus(); loIn.select();
}

function ranges3Key() { const v = S.v3; return `${S.movie}|${v.units}`; }

function bindAxisRanges() {
  div3.addEventListener('dblclick', (e) => {
    const axis = sceneAxisAt(e.clientX, e.clientY);
    if (!axis) return;
    e.stopPropagation(); e.preventDefault();
    const fl = div3._fullLayout.scene[`${axis}axis`];
    const t = axisTitles()[axis];
    rangePopover(e.clientX, e.clientY, `${t} range`, fl.range.slice(), (r) => {
      if (ranges3.key !== ranges3Key()) ranges3 = { key: ranges3Key(), x: null, y: null, z: null };
      ranges3[axis] = r;
      dirty.d3 = true; draw3D();
    });
  }, true);
  for (const d of [divZ, divXY, divD, divDrift, divM]) {
    if (!d) continue;
    d.addEventListener('dblclick', (e) => {
      const cls = e.target.classList;
      if (!cls || !cls.contains('drag')) return;
      if (!d._fullLayout || !d._fullLayout.xaxis) return;
      // Plotly's axis strips: ewdrag / wdrag / edrag along x, nsdrag / ndrag / sdrag along y
      // (a second y set on the right side for the overlaid yaxis2)
      let axName = null;
      if (['ewdrag', 'wdrag', 'edrag'].some((c) => cls.contains(c))) axName = 'xaxis';
      else if (['nsdrag', 'ndrag', 'sdrag'].some((c) => cls.contains(c))) {
        const xa = d._fullLayout.xaxis, r = e.target.getBoundingClientRect(), box = d.getBoundingClientRect();
        axName = (r.left + r.width / 2 - box.left) > xa._offset + xa._length / 2 && d._fullLayout.yaxis2 ? 'yaxis2' : 'yaxis';
      }
      if (!axName || !d._fullLayout[axName]) return;
      e.stopPropagation(); e.preventDefault();
      const fa = d._fullLayout[axName];
      const title = (fa.title && fa.title.text) || (axName === 'xaxis' ? 'frame' : 'value');
      rangePopover(e.clientX, e.clientY, `${title} range`, fa.range.slice(), (r) => {
        (ranges2[d.id] = ranges2[d.id] || {})[axName] = r;
        Plotly.relayout(d, r ? { [`${axName}.range`]: r, [`${axName}.autorange`]: false } : { [`${axName}.autorange`]: true })
          .then(() => updateFrameLines());
      });
    }, true);
  }
}
// user ranges onto a freshly built 2D layout
function withRanges2(d, L) {
  const r = ranges2[d.id];
  if (r) for (const [ax, v] of Object.entries(r)) if (L[ax]) { if (v) { L[ax].range = v; L[ax].autorange = false; } else { delete L[ax].range; L[ax].autorange = true; } }
  return L;
}

const sized = (d) => d.clientWidth > 60 && d.clientHeight > 60;
// Plotly throws "Something went wrong with axis scaling" when drawn into a zero-sized
// container: defer until it has a size, and recover from a broken state by purging once.
function safePlot(d, traces, layout, retry = true) {
  if (!sized(d)) { d._pending = [traces, layout]; return Promise.resolve(false); }
  try {
    return Plotly.react(d, traces, layout, CFG).then(() => { bindClick(d); if (d !== div3) updateFrameLines(); return true; })
      .catch((e) => { console.warn('plot failed', d.id, e); });
  } catch (e) {
    Plotly.purge(d);
    d._bound = false;
    if (retry) { d._pending = null; return new Promise((res) => setTimeout(() => res(safePlot(d, traces, layout, false)), 250)); }
    console.warn('plot failed', d.id, e);
    return Promise.resolve(false);
  }
}

// ------------------------------------------------------------------ 3D
function coords(r) {
  const ds = S.ds, v = S.v3;
  const k = v.units === 'um' ? PX_UM : 1;
  const z = ds.Z[r];
  return [ds.X[r] * k, ds.Y[r] * k, z];      // z is height (up +): the indenter pushes into −z
}

function buildLines(trackFilter) {
  const ds = S.ds, F = S.F;
  const x = [], y = [], z = [], col = [], cd = [];
  for (let k = 0; k < ds.ids.length; k++) {
    if (!F.trackOk[k]) continue;
    const id = ds.ids[k];
    if (!trackFilter(id)) continue;
    const rows = F.tRows[k];
    if (rows.length < 2 && !S.sel.has(id)) continue;
    for (const r of rows) {
      const [a, b, cc] = coords(r);
      x.push(a); y.push(b); z.push(Number.isFinite(cc) ? cc : null);
      const cb = S.v3.colorBy;
      col.push(cb === 'frame' ? ds.frame[r] : cb === 'z' ? (Number.isFinite(ds.Z[r]) ? ds.Z[r] : null)
        : cb === 'motion' ? (ds.cols.pMoving && Number.isFinite(ds.cols.pMoving[r]) ? ds.cols.pMoving[r] : null)
          : cb === 'stage' ? (ds.cols.stage && Number.isFinite(ds.cols.stage[r]) ? ds.cols.stage[r] : null) : trackHue01(id));
      cd.push(id);
    }
    x.push(null); y.push(null); z.push(null); col.push(null); cd.push(null);
  }
  return { x, y, z, col, cd };
}

function lineColorSpec(L) {
  const v = S.v3;
  if (v.colorBy === 'frame') return { color: L.col, colorscale: parula.plotly, cmin: 1, cmax: S.nFrames, showscale: true, colorbar: { title: { text: 'Frame' }, thickness: 10, len: 0.5, tickfont: { color: DARK.font } } };
  if (v.colorBy === 'motion') return { color: L.col, colorscale: motionMap.plotly, cmin: 0, cmax: 1, showscale: true, colorbar: { title: { text: 'P(moving)' }, thickness: 10, len: 0.5 } };
  if (v.colorBy === 'stage') {           // discrete colour steps, one per stage number 1..K
    const names = S.stageNames || ['1', '2', '3'], K = names.length;
    const scale = [];
    names.forEach((_, i) => { scale.push([i / K, stageColor(i + 1)], [(i + 1) / K, stageColor(i + 1)]); });
    return { color: L.col, colorscale: scale, cmin: 0.5, cmax: K + 0.5, showscale: true,
      colorbar: { title: { text: 'stage' }, thickness: 10, len: 0.5, tickvals: names.map((_, i) => i + 1), ticktext: names } };
  }
  if (v.colorBy === 'z') {
    const [a, b] = S.ds.zAuto;
    return { color: L.col, colorscale: turbo.plotly, cmin: a, cmax: b, showscale: true, colorbar: { title: { text: 'z (µm)' }, thickness: 10, len: 0.5 } };
  }
  return { color: L.col, colorscale: hueScale, cmin: 0, cmax: 1, showscale: false };
}

function markerTrace() {
  const ds = S.ds, F = S.F, f = S.frame;
  const x = [], y = [], z = [], color = [], size = [], cd = [];
  for (let k = 0; k < ds.ids.length; k++) {
    if (!F.trackOk[k]) continue;
    const id = ds.ids[k];
    if (S.v3.onlySel && S.sel.size && !S.sel.has(id)) continue;
    const r = rowAt(F, k, f);
    if (r < 0) continue;
    const [a, b, c] = coords(r);
    x.push(a); y.push(b); z.push(Number.isFinite(c) ? c : null);
    const sel = S.sel.has(id);
    color.push('#ffffff'); size.push(sel ? 6 : 3); cd.push(id);
  }
  // Current-position spheres (shown with the top-bar "Markers" toggle).
  return { x, y, z, customdata: cd, marker: { color, size, opacity: 0.9, symbol: 'circle', line: { width: 0 } } };
}

function axisTitles() {
  const u = S.v3.units === 'um' ? 'µm' : 'px';
  return { x: `x (${u})`, y: `y (${u})`, z: 'z (µm, up)' };
}

export function draw3D() {
  if (!S.ds || !S.F || typeof Plotly === 'undefined') return;
  if (!vis3()) { dirty.d3 = true; return; }
  const t0 = performance.now();
  dirty.d3 = false; dirty.marker = false;
  const v = S.v3;
  const baseKey = JSON.stringify(v) + '|' + S.ds.coordLabel + (v.onlySel ? '|' + [...S.sel].join(',') : '');
  let base;
  if (baseCache.F === S.F && baseCache.ds === S.ds && baseCache.key === baseKey) base = baseCache.L;
  else {
    base = buildLines((id) => !v.onlySel || !S.sel.size || S.sel.has(id));
    baseCache = { F: S.F, ds: S.ds, key: baseKey, L: base, color: lineColorSpec(base) };
  }
  const selL = buildLines((id) => S.sel.has(id));
  const dim = S.sel.size && v.dim;
  const traces = [
    {
      type: 'scatter3d', mode: 'lines', name: 'tracks', x: base.x, y: base.y, z: base.z, customdata: base.cd,
      line: { width: 3, ...baseCache.color }, opacity: dim ? 0.25 : 1, connectgaps: false,
      hovertemplate: 'track %{customdata}<br>x %{x:.1f} y %{y:.1f}<br>z %{z:.2f}<extra></extra>',
    },
    {
      type: 'scatter3d', mode: 'lines', name: 'selected', x: selL.x, y: selL.y, z: selL.z, customdata: selL.cd,
      // same colour scale as the base tracks (frame / z / track), just thicker
      line: { width: 9, ...lineColorSpec(selL), showscale: false },
      connectgaps: false, hovertemplate: '<b>track %{customdata}</b><br>x %{x:.1f} y %{y:.1f}<br>z %{z:.2f}<extra></extra>',
    },
    { type: 'scatter3d', mode: 'markers', name: 'frame', hovertemplate: 'track %{customdata} (current frame)<extra></extra>', visible: S.ov.beads, ...markerTrace() },
  ];
  const t = axisTitles();
  const ax = (title, extra = {}) => ({ title: { text: title }, backgroundcolor: DARK.plot, gridcolor: DARK.grid, zerolinecolor: DARK.zero, showbackground: true, color: DARK.font, ...extra });
  if (ranges3.key !== ranges3Key()) ranges3 = { key: ranges3Key(), x: null, y: null, z: null };
  const rng = (a, reversed = false) => (ranges3[a] ? { range: reversed ? [ranges3[a][1], ranges3[a][0]] : ranges3[a], autorange: false } : {});
  const layout = {
    paper_bgcolor: DARK.paper, plot_bgcolor: DARK.plot, font: { color: DARK.font, size: 12 },
    margin: { l: 0, r: 0, t: 28, b: 0 }, showlegend: false,
    title: { text: `tracks · colour by ${v.colorBy} · ${S.ds.coordLabel}${S.tracking ? ' · re-tracked' : ''}`, font: { size: 12 }, x: 0.02, y: 0.985 },
    uirevision: `${S.movie}|${v.units}|${v.yDown}`,
    scene: {
      xaxis: ax(t.x, rng('x')), yaxis: ax(t.y, ranges3.y ? rng('y', v.yDown) : v.yDown ? { autorange: 'reversed' } : {}), zaxis: ax(t.z, rng('z')),
      aspectmode: v.units === 'um' ? 'data' : 'cube', bgcolor: DARK.plot,
    },
    hoverlabel: { bgcolor: '#222', font: { color: '#eee' } },
  };
  // keep the on-screen view unless the units / z convention changed (a new uirevision)
  const cam = div3._fullLayout && div3._fullLayout.uirevision === layout.uirevision ? liveCamera() : null;
  if (cam) layout.scene.camera = cam;
  safePlot(div3, traces, layout).then(() => note(perf.draw3D, performance.now() - t0));
}

// The camera as currently shown. Plotly writes a rotated/zoomed camera back to the layout only
// when some interactions end, and every restyle/react re-applies the layout camera -- so the
// view snapped back on each marker update during playback. Always pass the live camera on.
function liveCamera() {
  const sc = div3 && div3._fullLayout && div3._fullLayout.scene && div3._fullLayout.scene._scene;
  if (!sc || !sc.getCamera) return null;
  try { return sc.getCamera(); } catch (e) { return null; }
}

function updateMarkers() {
  if (!vis3()) { dirty.marker = true; return; }
  if (!S.ds || !S.F || !div3.data || div3.data.length < 3 || !div3._fullLayout || !sized(div3) || div3._pending || !S.ov.beads) return;
  const t0 = performance.now();
  const m = markerTrace();
  try {
    const cam = liveCamera();
    Plotly.update(div3, { x: [m.x], y: [m.y], z: [m.z], customdata: [m.customdata], 'marker.color': [m.marker.color], 'marker.size': [m.marker.size] },
      cam ? { 'scene.camera': cam } : {}, [2]);
  } catch (e) { console.warn('marker update failed', e); }
  note(perf.marker, performance.now() - t0);
}

// ------------------------------------------------------------------ per-track plots
function selectedK() {
  const ds = S.ds, out = [];
  if (!ds) return out;
  for (const id of S.sel) { const k = ds.kOf.get(id); if (k !== undefined) out.push(k); }
  out.sort((a, b) => ds.ids[a] - ds.ids[b]);
  return out;
}

function layout2(title, ytitle) {
  return {
    paper_bgcolor: DARK.paper, plot_bgcolor: DARK.plot, font: { color: DARK.font, size: 11 },
    margin: { l: 50, r: 10, t: 24, b: 26 }, showlegend: false,
    title: { text: title, font: { size: 12 }, x: 0.01, y: 0.97 },
    xaxis: { title: { text: '' }, gridcolor: DARK.grid, zerolinecolor: DARK.zero, range: [0.5, S.nFrames + 0.5] },
    yaxis: { title: { text: ytitle, standoff: 4 }, gridcolor: DARK.grid, zerolinecolor: DARK.zero },
    hovermode: 'closest', uirevision: S.movie,
  };
}

export function drawTrackPlots() {
  if (typeof Plotly === 'undefined') return;
  if (!visTracks()) { dirty.tracks = true; return; }
  const t0 = performance.now();
  dirty.tracks = false;
  const ds = S.ds, F = S.F;
  const ks = ds && F ? selectedK() : [];
  const shown = ks.slice(0, MAX_PLOT_TRACKS);
  const note_ = ks.length > MAX_PLOT_TRACKS ? ` (first ${MAX_PLOT_TRACKS} of ${ks.length})` : ks.length ? ` (${ks.length})` : ' — select tracks';
  document.getElementById('trackPlots').classList.toggle('empty', !ks.length);
  const tz = [], txy = [], td = [];
  for (const k of shown) {
    const id = ds.ids[k], rows = F.tRows[k].length ? F.tRows[k] : ds.trackRows[k];
    if (!rows.length) continue;
    const col = trackColor(id);
    const fr = Array.from(rows, (r) => ds.frame[r]);
    const r0 = rows[0];
    const X = ds.X, Y = ds.Y, Z = ds.Z;
    const z = Array.from(rows, (r) => (Number.isFinite(Z[r]) ? Z[r] : null));
    const dx = Array.from(rows, (r) => (X[r] - X[r0]) * PX_UM);
    const dy = Array.from(rows, (r) => (Y[r] - Y[r0]) * PX_UM);
    const z0 = Z[r0];
    const d3 = Array.from(rows, (r, j) => { const dz = Z[r] - z0; return Math.hypot(dx[j], dy[j], Number.isFinite(dz) ? dz : 0); });
    const common = { type: 'scattergl', mode: 'lines+markers', marker: { size: 3, color: col }, customdata: fr.map(() => id) };
    tz.push({ ...common, x: fr, y: z, line: { color: col, width: 1.5 }, name: `${id}`, hovertemplate: `track ${id}<br>frame %{x}<br>z %{y:.3f} µm<extra></extra>` });
    // Δx and Δy: dark and light shades of the track's colour, same line style
    const cx = trackShade(id, 42), cy = trackShade(id, 78);
    txy.push({ ...common, x: fr, y: dx, line: { color: cx, width: 1.5 }, marker: { size: 3, color: cx }, name: `${id} Δx`, hovertemplate: `track ${id}<br>frame %{x}<br>Δx %{y:.3f} µm<extra></extra>` });
    txy.push({ ...common, x: fr, y: dy, line: { color: cy, width: 1.5 }, marker: { size: 3, color: cy }, name: `${id} Δy`, hovertemplate: `track ${id}<br>frame %{x}<br>Δy %{y:.3f} µm<extra></extra>` });
    td.push({ ...common, x: fr, y: d3, line: { color: col, width: 1.5 }, name: `${id}`, hovertemplate: `track ${id}<br>frame %{x}<br>|Δr| %{y:.3f} µm<extra></extra>` });
  }
  const cm = ds ? ` · ${ds.coordLabel}` : '';
  const lD = layout2(`|Δr| 3D from first localization${cm}`, 'µm');
  lD.xaxis.title.text = 'frame';
  safePlot(divZ, tz, withRanges2(divZ, layout2(`z vs frame${note_}`, 'z (µm)')));
  safePlot(divXY, txy, withRanges2(divXY, layout2(`Δx (dark) · Δy (light)${cm}`, 'µm')));
  safePlot(divD, td, withRanges2(divD, lD)).then(() => note(perf.tracks, performance.now() - t0));
  drawMotionPlot(shown);
}

// probability of the moving state vs frame for the selected tracks, over the population's
// fraction of beads moving (grey area); only when the motion analysis is loaded
function drawMotionPlot(shown) {
  if (!divM) return;
  const ds = S.ds, F = S.F;
  const has = !!(ds && ds.has.pMoving);
  divM.style.display = has ? '' : 'none';
  if (!has) { if (divM.data) Plotly.purge(divM); divM._bound = false; return; }
  const n = S.nFrames, cnt = new Float64Array(n + 1), mov = new Float64Array(n + 1);
  const p = ds.cols.pMoving;
  for (let r = 0; r < ds.n; r++) {
    if (!F.rowOk[r] || !Number.isFinite(p[r])) continue;
    cnt[ds.frame[r]]++; if (p[r] > 0.5) mov[ds.frame[r]]++;
  }
  const frames = Array.from({ length: n }, (_, i) => i + 1);
  const tr = [{ type: 'scatter', mode: 'lines', x: frames, y: frames.map((f) => (cnt[f] ? mov[f] / cnt[f] : null)), name: 'fraction of beads moving',
    fill: 'tozeroy', line: { color: 'rgba(160,170,185,0.55)', width: 1 }, fillcolor: 'rgba(160,170,185,0.16)', hovertemplate: 'frame %{x}<br>%{y:.0%} of beads moving<extra></extra>' }];
  for (const k of shown) {
    const id = ds.ids[k], rows = F.tRows[k].length ? F.tRows[k] : ds.trackRows[k];
    if (!rows.length) continue;
    const col = trackColor(id);
    tr.push({ type: 'scattergl', mode: 'lines+markers', x: Array.from(rows, (r) => ds.frame[r]), y: Array.from(rows, (r) => (Number.isFinite(p[r]) ? p[r] : null)),
      line: { color: col, width: 1.5 }, marker: { size: 3, color: col }, customdata: Array.from(rows, () => id), name: `${id}`,
      hovertemplate: `track ${id}<br>frame %{x}<br>P(moving) %{y:.2f}<extra></extra>` });
  }
  const L = layout2('probability of moving (ExaTrack-style states) · grey: fraction of all beads moving', 'P');
  L.yaxis.range = [-0.03, 1.03];
  L.xaxis.title.text = 'frame';
  safePlot(divM, tr, withRanges2(divM, L));
}

// estimated rigid drift (dx, dy px; dz µm) vs frame, when {movie}_drift.csv exists
export function drawDriftPlot() {
  if (!divDrift) return;
  const D = S.drift;
  divDrift.style.display = D ? '' : 'none';
  if (!visTracks()) { dirty.drift = true; return; }
  dirty.drift = false;
  if (!D) { if (divDrift.data) Plotly.purge(divDrift); divDrift._bound = false; return; }
  const x = D.frame_number;
  const L = layout2(`drift · dx, dy (px, left) · dz (µm, right)`, 'px');
  L.yaxis2 = { title: { text: 'µm', standoff: 2 }, overlaying: 'y', side: 'right', gridcolor: 'rgba(0,0,0,0)', zerolinecolor: DARK.zero };
  L.margin.r = 40;
  // legend in its own strip below the plot, never over the traces
  L.margin.b = 46;
  L.showlegend = true;
  L.legend = { orientation: 'h', x: 0, xanchor: 'left', y: -0.16, yanchor: 'top', font: { size: 10 }, bgcolor: 'rgba(0,0,0,0)' };
  const tr = [
    { type: 'scatter', mode: 'lines', x, y: D.dx, name: 'dx', line: { color: '#4aa3ff', width: 1.5 } },
    { type: 'scatter', mode: 'lines', x, y: D.dy, name: 'dy', line: { color: '#ffb347', width: 1.5 } },
    { type: 'scatter', mode: 'lines', x, y: D.dz || [], name: 'dz (smoothed)', yaxis: 'y2', line: { color: '#3ecf6e', width: 2 } },
  ];
  if (D.dzRaw) tr.push({ type: 'scatter', mode: 'lines', x, y: D.dzRaw, name: 'dz per frame', yaxis: 'y2', line: { color: 'rgba(255,91,91,0.85)', width: 1.5 } });
  setTimeout(() => safePlot(divDrift, tr, withRanges2(divDrift, L)), 0);
}

// current-frame cursor: a DOM line positioned from the axis mapping (no Plotly relayout)
function updateFrameLines() {
  for (const d of [divZ, divXY, divD, divDrift, divM]) {
    if (!d || !d._fullLayout || !d._fullLayout.xaxis || d.style.display === 'none') continue;
    const xa = d._fullLayout.xaxis, ya = d._fullLayout.yaxis;
    let line = d.querySelector(':scope > .fline');
    if (!line) { line = document.createElement('div'); line.className = 'fline'; d.appendChild(line); }
    const x = xa._offset + xa.l2p(S.frame);
    line.style.left = `${x}px`;
    line.style.top = `${ya._offset}px`;
    line.style.height = `${ya._length}px`;
    line.style.display = x >= xa._offset - 1 && x <= xa._offset + xa._length + 1 ? '' : 'none';
  }
}

export function download3D() {
  if (!div3.data) { emit('toast', ['Open the 3D tab first', 'err']); return; }
  Plotly.downloadImage(div3, { format: 'png', width: 1600, height: 1200, filename: `${S.movie}_3d_tracks` });
}
