// Deformation view: the displacement of every bead from a reference frame to the current frame,
// interpolated into a continuous 3D field (Gaussian-process / kriging interpolation, gp.js).
//
// Default display: a sheet of material at a chosen depth, drawn as a mesh and moved by the field
// (a warped surface, like a membrane pushed in by the indenter), or the same sheet seen from above
// as a map.
//
// Positions are the analysis coordinates (X, Y, Z of the dataset: raw / corrected / stabilized /
// refined, as chosen under Coordinates), in µm; the field is defined over the reference-frame
// positions (a Lagrangian map). Everything that sets the scene -- smoothing length, grid, axis
// ranges, colour range, arrow length -- is fixed per reference frame / coordinate set (fitted at
// the frame of largest displacement), so playback only moves the surface, never the axes.
import { S, PX_UM, withRun } from './state.js';
import { on, emit, $, throttle, debounce, percentile, inferno, rdbu, toast, downloadUrl, startTask, nextPaint } from './util.js';
import { rowAt, setCoords } from './data.js';
import { fitHyperSteps, condition } from './gp.js';

const DARK = { paper: '#0f1115', plot: '#0f1115', grid: '#262a33', font: '#c9cdd4', zero: '#3a404b' };
// the tool bar (camera, zoom, save) stays visible in this view instead of only on hover
const CFG = { displaylogo: false, responsive: true, displayModeBar: true, modeBarButtonsToRemove: ['sendDataToCloud', 'lasso2d', 'select2d'] };
const MIN_BEADS = 8;
const MAX_FIT = 600;          // beads used for the hyperparameter fit (all are used for the field)
const DENS_MIN = 0.35;        // map: nothing where there is less than about a third of a bead within ℓ

let div, dirty = true, hyper = null, exporting = false, pendingFit = null, healing = false;
const vis = () => S.ui.panel && S.ui.tab === 'deform';

export const DF = { show: 'sheet', qty: 'uz', depth: 0.5, warp: 1, ell: null, beads: true, arrows: false, arrowScale: null, pos: 'best' };

// Positions used for the field. 'best': lateral-corrected, whole-field drift removed (XY and Z) and
// refined by the motion analysis where available -- whatever the display coordinates are, since
// drift would otherwise count as deformation. 'display': the coordinates chosen under Coordinates.
let posCache = null;
function positions() {
  const ds = S.ds;
  if (DF.pos === 'display') return { X: ds.X, Y: ds.Y, Z: ds.Z, sdX: ds.sdX, sdY: ds.sdY, sdZ: ds.sdZ, label: ds.coordLabel };
  if (posCache && posCache.ds === ds && posCache.drift === S.drift) return posCache;
  const t = { n: ds.n, cols: ds.cols, frame: ds.frame, has: ds.has };
  setCoords(t, { corr: true, xy: true, z: true, refined: true }, S.drift);
  posCache = { ds, drift: S.drift, X: t.X, Y: t.Y, Z: t.Z, sdX: t.sdX, sdY: t.sdY, sdZ: t.sdZ, label: t.coordLabel };
  return posCache;
}

// beads present at both frames: reference positions P (µm), displacements U and noise variances V
function gather(a, b) {
  const ds = S.ds, F = S.F, p = positions(), P = [], U = [[], [], []], V = [[], [], []], ids = [];
  for (let k = 0; k < ds.ids.length; k++) {
    if (!F.trackOk[k]) continue;
    const ra = rowAt(F, k, a), rb = rowAt(F, k, b);
    if (ra < 0 || rb < 0) continue;
    const za = p.Z[ra], zb = p.Z[rb];
    if (!Number.isFinite(za) || !Number.isFinite(zb)) continue;
    P.push(p.X[ra] * PX_UM, p.Y[ra] * PX_UM, za);
    U[0].push((p.X[rb] - p.X[ra]) * PX_UM); U[1].push((p.Y[rb] - p.Y[ra]) * PX_UM); U[2].push(zb - za);
    V[0].push((p.sdX[ra] ** 2 + p.sdX[rb] ** 2) * PX_UM * PX_UM);
    V[1].push((p.sdY[ra] ** 2 + p.sdY[rb] ** 2) * PX_UM * PX_UM);
    V[2].push(p.sdZ[ra] ** 2 + p.sdZ[rb] ** 2);
    ids.push(ds.ids[k]);
  }
  return { P: Float64Array.from(P), U: U.map((u) => Float64Array.from(u)), V: V.map((v) => Float64Array.from(v)), ids, n: ids.length };
}

// frame with the largest displacements (98th percentile of |u|: the few beads in the dent, not one
// stray bead) relative to the reference
function busiestFrame(a) {
  let best = a, bestV = -1;
  const step = Math.max(1, Math.floor(S.nFrames / 60));
  for (let f = 1; f <= S.nFrames; f += step) {
    if (f === a) continue;
    const g = gather(a, f);
    if (g.n < MIN_BEADS) continue;
    const v = percentile(Array.from(g.U[0], (_, i) => Math.hypot(g.U[0][i], g.U[1][i], g.U[2][i])), 98);
    if (v > bestV) { bestV = v; best = f; }
  }
  return best;
}

// regular grid over all beads of the reference frame (fixed for the whole movie)
function makeGrid(P) {
  let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < P.length; i += 3) for (let c = 0; c < 3; c++) { lo[c] = Math.min(lo[c], P[i + c]); hi[c] = Math.max(hi[c], P[i + c]); }
  const pad = hi.map((h, c) => 0.03 * Math.max(h - lo[c], 1));
  lo = lo.map((v, c) => v - pad[c]); hi = hi.map((v, c) => v + pad[c]);
  const nx = 44, ny = Math.max(8, Math.min(60, Math.round(nx * (hi[1] - lo[1]) / (hi[0] - lo[0])))),
    nz = Math.max(4, Math.min(14, Math.round(nx * (hi[2] - lo[2]) / (hi[0] - lo[0])) + 4));
  const ax = (c, m) => Float64Array.from({ length: m }, (_, i) => lo[c] + (hi[c] - lo[c]) * i / (m - 1));
  return { xs: ax(0, nx), ys: ax(1, ny), zs: ax(2, nz), lo, hi };
}

// robust plane z = c0 + c1 x + c2 y through the beads (least squares, then again without outliers
// beyond 3 MADs): the undeformed "median plane" of the sample, tilt included
function medianPlane(P) {
  const n = P.length / 3;
  let keep = Array.from({ length: n }, (_, i) => i), c = [0, 0, 0];
  for (let pass = 0; pass < 3; pass++) {
    const M = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], r = [0, 0, 0];
    const x0 = P[0], y0 = P[1];
    for (const i of keep) {
      const v = [1, P[3 * i] - x0, P[3 * i + 1] - y0], z = P[3 * i + 2];
      for (let a = 0; a < 3; a++) { r[a] += v[a] * z; for (let b = 0; b < 3; b++) M[a][b] += v[a] * v[b]; }
    }
    const sol = solve3(M, r);
    if (!sol) { const zs = keep.map((i) => P[3 * i + 2]).sort((p, q) => p - q); return () => zs[zs.length >> 1]; }
    c = [sol[0] - sol[1] * x0 - sol[2] * y0, sol[1], sol[2]];
    const res = Array.from({ length: n }, (_, i) => P[3 * i + 2] - (c[0] + c[1] * P[3 * i] + c[2] * P[3 * i + 1]));
    const mad = percentile(res.map(Math.abs), 50) * 1.4826 || 1;
    keep = keep.filter((i) => Math.abs(res[i]) < 3 * mad);
    if (keep.length < 3) break;
  }
  return (x, y) => c[0] + c[1] * x + c[2] * y;
}
function solve3(M, r) {
  const A = M.map((row, i) => [...row, r[i]]);
  for (let i = 0; i < 3; i++) {
    let p = i; for (let k = i + 1; k < 3; k++) if (Math.abs(A[k][i]) > Math.abs(A[p][i])) p = k;
    [A[i], A[p]] = [A[p], A[i]];
    if (Math.abs(A[i][i]) < 1e-12) return null;
    for (let k = 0; k < 3; k++) if (k !== i) { const f = A[k][i] / A[i][i]; for (let j = i; j < 4; j++) A[k][j] -= f * A[i][j]; }
  }
  return A.map((row, i) => row[3] / row[i]);
}

// deterministic subset of at most MAX_FIT beads (hyperparameter fit and colour-range sampling only;
// the field shown always uses every bead)
function thin(g) {
  if (g.n <= MAX_FIT) return g;
  const keep = Array.from({ length: g.n }, (_, i) => i).filter((i) => (i * 7919) % g.n < MAX_FIT);
  return { P: Float64Array.from(keep.flatMap((i) => [g.P[3 * i], g.P[3 * i + 1], g.P[3 * i + 2]])),
    U: g.U.map((u) => Float64Array.from(keep, (i) => u[i])), V: g.V.map((v) => Float64Array.from(keep, (i) => v[i])),
    ids: keep.map((i) => g.ids[i]), n: keep.length };
}

function hyperKey() { return `${S.movie}|${S.disp.a}|${positions().label}|${S.F.nTracks}|${DF.ell}`; }

function ensureHyper() {
  const it = hyperSteps();
  let r;
  while (!(r = it.next()).done);
  return r.value;
}
// the same with a progress bar, letting the page paint between steps
async function ensureHyperAsync(task) {
  const it = hyperSteps();
  let r;
  while (!(r = it.next()).done) { task.set(r.value.frac, r.value.detail); await nextPaint(); }
  return r.value;
}

// fit (smoothing length, amplitudes) + colour-range samples; yields {frac, detail} as it goes
function* hyperSteps() {
  const key = hyperKey();
  if (hyper && hyper.key === key) return hyper;
  const a = S.disp.a;
  const ref = gather(a, a);
  if (ref.n < MIN_BEADS) return (hyper = { key, fail: true });
  yield { frac: 0.02, detail: 'finding the frame of largest displacement' };
  const fitFrame = busiestFrame(a);
  let g = gather(a, fitFrame);
  if (g.n < MIN_BEADS) return (hyper = { key, fail: true });
  g = thin(g);
  const t0 = performance.now();
  // 5-45 %: the smoothing length (one step per candidate length)
  const fit = fitHyperSteps(g.P, g.U, g.V, DF.ell);
  let fr;
  yield { frac: 0.05, detail: 'fitting the smoothing length' };
  while (!(fr = fit.next()).done) yield { frac: 0.05 + 0.4 * fr.value, detail: 'fitting the smoothing length' };
  const h = fr.value;
  // typical large displacement at the busiest frame: fixes colour range, arrow length and z room
  const mags = Array.from(g.U[0], (_, i) => Math.hypot(g.U[0][i], g.U[1][i], g.U[2][i]));
  // the field on the median plane (near beads) at ~30 frames across the movie, so the fixed colour
  // range reaches the deepest part of the dent whenever it happens
  const G = makeGrid(ref.P), plane = medianPlane(ref.P);
  const Q = [];
  for (const y of G.ys) for (const x of G.xs) Q.push(x, y, plane(x, y));
  const Qf = Float64Array.from(Q), samples = [];
  const step = Math.max(1, Math.round(S.nFrames / 24));
  let uMax = 0;
  for (let f = 1; f <= S.nFrames; f += step) {
    if (f === a) continue;
    yield { frac: 0.45 + 0.55 * (f - 1) / S.nFrames, detail: `colour range: frame ${f} / ${S.nFrames}` };
    const gAll = gather(a, f);
    if (gAll.n < MIN_BEADS) continue;
    for (let i = 0; i < gAll.n; i++) uMax = Math.max(uMax, Math.hypot(gAll.U[0][i], gAll.U[1][i], gAll.U[2][i]));
    const gf = thin(gAll);
    const pf = condition(gf.P, gf.U, gf.V, h.ell, h.s2);
    if (!pf) continue;
    const r = pf(Qf, true);
    const keep = Array.from(r.density.keys()).filter((i) => r.density[i] >= DENS_MIN);
    samples.push({ u: r.u.map((c) => Float64Array.from(keep, (i) => c[i])), grad: r.grad.map((c) => Float64Array.from(keep, (i) => c[i])) });
  }
  hyper = { key, ...h, fitFrame, samples, cmax: {}, G, plane, uRef: Math.max(percentile(mags, 95) || 0, 1e-3), uMax: Math.max(uMax, 1e-3),
    ms: performance.now() - t0 };
  return hyper;
}

// r: a prediction { u: [ux, uy, uz], grad: ∂u_c/∂x_j as grad[3c + j] (only for the strain quantities) }
// Strains are in-plane only: the bead layer is too thin for derivatives in depth to be measured.
function quantity(r, i) {
  const u = r.u, G = r.grad;
  switch (DF.qty) {
    case 'ux': return u[0][i];
    case 'uy': return u[1][i];
    case 'uz': return u[2][i];                        // z is height: the indenter pushes into −z
    case 'lat': return Math.hypot(u[0][i], u[1][i]);
    case 'areal': return 100 * (G[0][i] + G[4][i]);                                   // εxx + εyy
    case 'shear': return 100 * Math.hypot((G[0][i] - G[4][i]) / 2, (G[1][i] + G[3][i]) / 2);   // max in-plane shear
    case 'tilt': return (180 / Math.PI) * Math.atan(Math.hypot(G[6][i], G[7][i]));  // slope of the warped plane
    default: return Math.hypot(u[0][i], u[1][i], u[2][i]);
  }
}
const QTY = { mag: '|u| (µm)', lat: 'lateral |u| (µm)', ux: 'Δx (µm)', uy: 'Δy (µm)', uz: 'Δz (µm)',
  areal: 'areal strain (%)', shear: 'in-plane shear (%)', tilt: 'surface tilt (°)' };
const signed = () => ['ux', 'uy', 'uz', 'areal'].includes(DF.qty);
const needGrad = () => ['areal', 'shear', 'tilt'].includes(DF.qty);
// Lab z convention: z is height, up (toward the indenter, which comes from above) is +, so an
// indentation is negative Δz. zMicrons is reported in this convention (pipeline.to_lab_z; older
// result folders are converted by the server on loading), so display and data coincide.
const dispZ = (z) => z;
const dispDz = (dz) => dz;

export function initDeform() {
  div = document.getElementById('plotDeform');
  const redraw = () => { dirty = true; refresh(); };
  on('data', () => { hyper = null; posCache = null; redraw(); });
  on('filter', redraw);
  on('disp', redraw);
  on('view3d', redraw);
  on('panel', refresh);
  const during = throttle(() => { if (vis()) draw(); }, 300), settled = debounce(() => { if (vis()) draw(); }, 120);
  on('frame', () => { if (exporting) return; if (!vis()) { dirty = true; return; } if (S.playing) during(); settled(); });
  $('#dfExField').onclick = () => exportField().catch((e) => toast(`Export failed: ${e.message}`, 'err', 8000));
  $('#dfExMp4').onclick = () => exportAnimation('video').catch((e) => toast(`Animation failed: ${e.message}`, 'err', 8000));
  $('#dfExGif').onclick = () => exportAnimation('gif').catch((e) => toast(`Animation failed: ${e.message}`, 'err', 8000));
  const bind = (id, key, parse = (v) => v) => $(id).addEventListener('change', (e) => { DF[key] = parse(e.target.type === 'checkbox' ? e.target.checked : e.target.value); updateTools(); redraw(); });
  bind('#dfQty', 'qty'); bind('#dfShow', 'show'); bind('#dfPos', 'pos');
  bind('#dfDepth', 'depth', (v) => +v / 100);
  $('#dfDepth').addEventListener('input', (e) => { DF.depth = +e.target.value / 100; throttledDraw(); });
  bind('#dfWarp', 'warp', (v) => (+v > 0 ? +v : 1));
  bind('#dfBeads', 'beads'); bind('#dfArrows', 'arrows');
  bind('#dfArrowScale', 'arrowScale', (v) => (+v > 0 ? +v : null));
  bind('#dfEll', 'ell', (v) => (+v > 0 ? +v : null));
  // back to automatic: clear the typed value
  $('#dfEllAuto').onclick = () => { DF.ell = null; $('#dfEll').value = ''; updateTools(); redraw(); };
  $('#dfArrowAuto').onclick = () => { DF.arrowScale = null; $('#dfArrowScale').value = ''; updateTools(); redraw(); };
  $('#dfRef').addEventListener('change', (e) => {
    const v = Math.round(+e.target.value);
    if (!(v >= 1 && v <= S.nFrames)) return;
    S.disp.a = v; const dA = $('#dA'); if (dA) dA.value = v; emit('disp');
  });
  // the browser can drop a WebGL context (too many 3D views, GPU reset), which leaves the 3D plot
  // blank: rebuild it
  div.addEventListener('webglcontextlost', (e) => {
    e.preventDefault();
    setTimeout(() => { try { Plotly.purge(div); } catch (err) { /* ignore */ } dirty = true; refresh(); }, 300);
  }, true);
  // start over: new fit, fresh plot (also recovers a 3D view whose graphics context was lost);
  // with no data loaded, reload the movie's data
  $('#dfRefresh').onclick = () => {
    if (!S.ds || !S.F) { info('Reloading the data…'); emit('reloadData'); return; }
    emit('syncRun');
    hyper = null; posCache = null;
    try { Plotly.purge(div); } catch (e) { /* ignore */ }
    dirty = true; draw();
  };
  $('#dfRefCur').onclick =() => { S.disp.a = S.frame; $('#dfRef').value = S.frame; const dA = $('#dA'); if (dA) dA.value = S.frame; emit('disp'); };
  new ResizeObserver(throttle(() => { if (div.data) try { Plotly.Plots.resize(div); } catch (e) { /* ignore */ } }, 150)).observe(div);
  updateTools();
}
const throttledDraw = throttle(() => { if (vis()) draw(); }, 120);

// controls that only apply to some display modes
function updateTools() {
  const sheet = DF.show === 'sheet';
  $('#dfWarpWrap').classList.toggle('hidden', !sheet);
  // "auto" is lit while the value is automatic; the box then shows the value in use as its placeholder
  $('#dfEllAuto').classList.toggle('on', DF.ell === null);
  $('#dfArrowAuto').classList.toggle('on', DF.arrowScale === null);
  $('#dfArrowRow').classList.toggle('hidden', !DF.arrows);
}
function showAutoValues(h, arrowScale) {
  $('#dfEll').placeholder = DF.ell === null && h ? `auto (${h.ell.toFixed(0)})` : 'auto';
  $('#dfArrowScale').placeholder = DF.arrowScale === null && arrowScale ? `auto (${arrowScale.toFixed(arrowScale < 10 ? 1 : 0)})` : 'auto';
}

export function refresh() { if (vis() && dirty) draw(); }

function info(html, tip = '') { const el = $('#dfInfo'); el.innerHTML = html; el.title = tip; }

// the camera as shown (Plotly re-applies the layout camera on every react)
function liveCamera() {
  const sc = div && div._fullLayout && div._fullLayout.scene && div._fullLayout.scene._scene;
  if (!sc || !sc.getCamera) return null;
  try { return sc.getCamera(); } catch (e) { return null; }
}

// a sheet of material: at reference-frame height zRef(x, y), moved by the field (×w); surface + mesh lines
function sheetTraces(predict, G, zRef, w, colour, withScale) {
  const nx = G.xs.length, ny = G.ys.length, Q = new Float64Array(3 * nx * ny);
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) { const q = j * nx + i; Q[3 * q] = G.xs[i]; Q[3 * q + 1] = G.ys[j]; Q[3 * q + 2] = zRef(G.xs[i], G.ys[j]); }
  const r = predict(Q, needGrad());
  const X = [], Y = [], Z = [], C = [];
  for (let j = 0; j < ny; j++) {
    const xr = [], yr = [], zr = [], cr = [];
    for (let i = 0; i < nx; i++) {
      const q = j * nx + i;
      xr.push(G.xs[i] + w * r.u[0][q]); yr.push(G.ys[j] + w * r.u[1][q]); zr.push(dispZ(Q[3 * q + 2]) + w * dispDz(r.u[2][q]));
      cr.push(quantity(r, q));
    }
    X.push(xr); Y.push(yr); Z.push(zr); C.push(cr);
  }
  const out = [{ type: 'surface', x: X, y: Y, z: Z, surfacecolor: C, ...colour, showscale: withScale, opacity: 0.92,
    lighting: { ambient: 0.75, diffuse: 0.5, specular: 0.15, roughness: 0.8 },
    hovertemplate: `x %{x:.0f} y %{y:.0f} µm<br>${QTY[DF.qty]} %{surfacecolor:.2f}<extra></extra>` }];
  // mesh lines every few nodes, following the warped surface
  const lx = [], ly = [], lz = [], step = 3;
  for (let j = 0; j < ny; j += step) { for (let i = 0; i < nx; i++) { lx.push(X[j][i]); ly.push(Y[j][i]); lz.push(Z[j][i]); } lx.push(null); ly.push(null); lz.push(null); }
  for (let i = 0; i < nx; i += step) { for (let j = 0; j < ny; j++) { lx.push(X[j][i]); ly.push(Y[j][i]); lz.push(Z[j][i]); } lx.push(null); ly.push(null); lz.push(null); }
  out.push({ type: 'scatter3d', mode: 'lines', x: lx, y: ly, z: lz, line: { color: 'rgba(255,255,255,0.28)', width: 1 }, hoverinfo: 'skip', showlegend: false });
  return out;
}

// draw frame b (default: the movie frame); resolves when Plotly has rendered
export function draw(bOverride = null) {
  if (typeof Plotly === 'undefined' || !div) return Promise.resolve();
  if (!vis()) { dirty = true; return Promise.resolve(); }
  if (!S.ds || !S.F) {
    Plotly.purge(div);
    info('No localizations loaded for this movie (choose a result set and movie at the top).');
    return Promise.resolve();
  }
  dirty = false;
  $('#dfRef').value = S.disp.a;
  // the first draw for a reference frame fits the interpolation (a second or more with many beads):
  // say so, and let the message paint before the work starts
  if (!hyper || hyper.key !== hyperKey()) {
    if (!pendingFit) {
      info(`Fitting the interpolation for reference frame ${S.disp.a} (smoothing length, colour range)…`);
      const task = startTask('deformation map');
      pendingFit = ensureHyperAsync(task)
        .catch((e) => { console.error(e); hyper = { key: hyperKey(), fail: true }; })
        .finally(() => { task.done(); pendingFit = null; });
    }
    return pendingFit.then(() => draw(bOverride));
  }
  const a = S.disp.a, b = bOverride ?? S.frame;
  const h = hyper;
  const g = gather(a, b);
  if (!h || h.fail || g.n < MIN_BEADS) {
    Plotly.purge(div);
    info(`Needs at least ${MIN_BEADS} beads present in both the reference frame (${a}) and this frame.`);
    return Promise.resolve();
  }
  const t0 = performance.now();
  const predict = condition(g.P, g.U, g.V, h.ell, h.s2);
  if (!predict) { info('Interpolation failed (singular covariance).'); return Promise.resolve(); }
  const G = h.G;
  const traces = [];
  // colour range from the measured displacements at the busiest frame (fixed during playback)
  // colour range: the largest value of the (smooth, interpolated) field near beads over the sampled
  // frames, so the deepest part of the dent is not clipped; fixed for playback
  if (h.cmax[DF.qty] === undefined) {
    let top = 0;
    for (const r of h.samples) for (let i = 0; i < r.u[0].length; i++) top = Math.max(top, Math.abs(quantity(r, i)));
    h.cmax[DF.qty] = Math.max(top, 1e-3);
  }
  const cmax = h.cmax[DF.qty];
  const cmin = signed() ? -cmax : 0;
  const colour = { colorscale: signed() ? rdbu.plotly : inferno.plotly, cmin, cmax,
    colorbar: { title: { text: QTY[DF.qty] }, thickness: 10, len: 0.55, tickfont: { color: DARK.font } } };
  const sheet = DF.show === 'sheet';
  const w = sheet ? DF.warp : 0;
  // the beads' median plane (slider in the middle), shifted up/down by the slider
  const zSpan = G.hi[2] - G.lo[2];
  const sheetAt = (f) => (x, y) => h.plane(x, y) + (0.5 - f) * zSpan;     // slider right = deeper
  const span = Math.max(G.hi[0] - G.lo[0], G.hi[1] - G.lo[1]);
  const sc = DF.arrowScale || (0.06 * span) / h.uRef;
  if (DF.show === 'map') return drawMap({ a, b, g, h, predict, colour, zAt: sheetAt(DF.depth), sc, t0 });
  traces.push(...sheetTraces(predict, G, sheetAt(DF.depth), w, colour, true));
  // measured beads: dots where they are now (moved by the same warp as the sheet), optional arrows
  if (DF.beads || DF.arrows) {
    const bx = [], by = [], bz = [], txt = [], ax = [], ay = [], az = [];
    for (let i = 0; i < g.n; i++) {
      const px = g.P[3 * i], py = g.P[3 * i + 1], pz = dispZ(g.P[3 * i + 2]);
      bx.push(px + w * g.U[0][i]); by.push(py + w * g.U[1][i]); bz.push(pz + w * dispDz(g.U[2][i]));
      txt.push(`track ${g.ids[i]}<br>Δx ${g.U[0][i].toFixed(2)} Δy ${g.U[1][i].toFixed(2)} Δz ${dispDz(g.U[2][i]).toFixed(2)} µm`);
      if (DF.arrows) { ax.push(px, px + sc * g.U[0][i], null); ay.push(py, py + sc * g.U[1][i], null); az.push(pz, pz + sc * dispDz(g.U[2][i]), null); }
    }
    if (DF.arrows) traces.push({ type: 'scatter3d', mode: 'lines', x: ax, y: ay, z: az, line: { color: '#e8f1ff', width: 3 }, hoverinfo: 'skip', showlegend: false });
    if (DF.beads) traces.push({ type: 'scatter3d', mode: 'markers', x: bx, y: by, z: bz, // dark on the (mostly white) plane, light in the dark volume
      text: txt, marker: { size: g.n > 400 ? 1.6 : 2.2, color: sheet ? 'rgba(60,64,74,0.8)' : (g.n > 400 ? 'rgba(184,199,220,0.55)' : '#b8c7dc') },
      hovertemplate: '%{text}<extra></extra>', showlegend: false });
  }
  // fixed scene: ranges from the reference grid plus room for the largest displacement
  const room = sheet ? DF.warp * h.uMax * 1.1 : 0;
  const zr = [dispZ(G.lo[2] - room), dispZ(G.hi[2] + room)].sort((p, q) => p - q);
  const xyRoom = sheet ? DF.warp * h.uMax * 0.2 : 0;
  const spanZ = zr[1] - zr[0], spanX = G.hi[0] - G.lo[0] + 2 * xyRoom, spanY = G.hi[1] - G.lo[1] + 2 * xyRoom;
  const zStretch = Math.max(1, (0.35 * spanX) / spanZ);
  const axis = (title, range, extra = {}) => ({ title: { text: title }, range, autorange: false, backgroundcolor: DARK.plot, gridcolor: DARK.grid, zerolinecolor: DARK.zero, showbackground: true, color: DARK.font, ...extra });
  const layout = {
    paper_bgcolor: DARK.paper, plot_bgcolor: DARK.plot, font: { color: DARK.font, size: 12 },
    margin: { l: 0, r: 0, t: 28, b: 0 }, showlegend: false,
    title: { text: `deformation · frame ${a} → ${b} · ${QTY[DF.qty]}`, font: { size: 12 }, x: 0.02, y: 0.985 },
    uirevision: `${S.movie}|${S.v3.yDown}|${S.disp.a}|${S.ds.coordLabel}`,
    scene: {
      xaxis: axis('x (µm)', [G.lo[0] - xyRoom, G.hi[0] + xyRoom]),
      yaxis: axis('y (µm)', S.v3.yDown ? [G.hi[1] + xyRoom, G.lo[1] - xyRoom] : [G.lo[1] - xyRoom, G.hi[1] + xyRoom]),
      zaxis: axis('z (µm, up)', zr),
      aspectmode: 'manual', aspectratio: { x: 1, y: spanY / spanX, z: (spanZ * zStretch) / spanX }, bgcolor: DARK.plot,
    },
    hoverlabel: { bgcolor: '#222', font: { color: '#eee' } },
  };
  const cam = div._fullLayout && div._fullLayout.uirevision === layout.uirevision ? liveCamera() : null;
  if (cam) layout.scene.camera = cam;
  else layout.scene.camera = { eye: { x: 0.7, y: -1.05, z: 1.15 }, center: { x: 0, y: 0, z: -0.15 } };
  // a 3D scene whose WebGL canvas could not be created (e.g. drawn while the pane was being laid
  // out, or the browser's limit of graphics contexts) renders nothing: rebuild it once
  const done = Plotly.react(div, traces, layout, CFG).then(() => {
    if (div.querySelector('canvas') || healing) return;
    healing = true;
    try { Plotly.purge(div); } catch (e) { /* ignore */ }
    return draw(bOverride).finally(() => { healing = false; });
  });
  showAutoValues(h, (0.06 * span) / h.uRef);
  summary(g, h, t0, [sheet && DF.warp !== 1 ? `drawn ×${DF.warp}` : '', zStretch > 1.05 ? `z ×${zStretch.toFixed(0)}` : '',
    DF.arrows ? `arrows ×${sc.toFixed(sc < 10 ? 1 : 0)}` : '']);
  return done;
}

// one short status line; the full explanation is in its tooltip
function summary(g, h, t0, extra = [], note = '') {
  const sd = (c) => Math.sqrt(percentile(Array.from(g.V[c]), 50));
  const p = positions(), ms = performance.now() - t0;
  const drift = DF.pos === 'display' && S.drift && !(S.coords.xy && S.coords.z);
  info(`${g.n} beads · ℓ ${h.ell.toFixed(0)} µm${DF.ell ? ' (set)' : ''} · error ${sd(0).toFixed(2)}/${sd(1).toFixed(2)}/${sd(2).toFixed(2)} µm`
    + extra.filter(Boolean).map((s) => ` · ${s}`).join('')
    + (drift ? ' · <span class="warn">drift not removed</span>' : '')
    + `<span class="dim"> · ${ms.toFixed(0)} ms</span>` + (note ? `<br><span class="dim">${note}</span>` : ''),
    `${g.n} beads present in frames ${S.disp.a} and ${S.frame}\n`
    + `Smoothing length ℓ = ${h.ell.toFixed(1)} µm (${DF.ell ? 'set by hand' : `fitted by maximum likelihood at frame ${h.fitFrame}`}; median bead spacing ${h.nn.toFixed(0)} µm)\n`
    + `Typical displacement error of one bead, x / y / z: ${sd(0).toFixed(2)} / ${sd(1).toFixed(2)} / ${sd(2).toFixed(2)} µm\n`
    + `Positions: ${p.label}` + (drift ? '\nXY and Z stabilization are not both on, so whole-field drift counts as displacement' : ''));
}

// top view: the quantity on the (depth-shifted) median plane as a map, beads and lateral arrows on top
function drawMap({ a, b, g, h, predict, colour, zAt, sc, t0 }) {
  const G = h.G, nx = G.xs.length, ny = G.ys.length, Q = new Float64Array(3 * nx * ny);
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) { const q = j * nx + i; Q[3 * q] = G.xs[i]; Q[3 * q + 1] = G.ys[j]; Q[3 * q + 2] = zAt(G.xs[i], G.ys[j]); }
  const r = predict(Q, needGrad());
  const Z = Array.from({ length: ny }, (_, j) => Array.from({ length: nx }, (_, i) => { const q = j * nx + i; return r.density[q] >= DENS_MIN ? quantity(r, q) : null; }));
  const traces = [{ type: 'heatmap', x: Array.from(G.xs), y: Array.from(G.ys), z: Z, zsmooth: 'best', colorscale: colour.colorscale,
    zmin: colour.cmin, zmax: colour.cmax, colorbar: colour.colorbar, hoverongaps: false,
    hovertemplate: `x %{x:.0f} y %{y:.0f} µm<br>${QTY[DF.qty]} %{z:.2f}<extra></extra>` }];
  if (DF.arrows) {
    const ax = [], ay = [];
    for (let i = 0; i < g.n; i++) { const px = g.P[3 * i], py = g.P[3 * i + 1]; ax.push(px, px + sc * g.U[0][i], null); ay.push(py, py + sc * g.U[1][i], null); }
    traces.push({ type: 'scatter', mode: 'lines', x: ax, y: ay, line: { color: '#e8f1ff', width: 1.5 }, hoverinfo: 'skip', showlegend: false });
  }
  if (DF.beads) {
    traces.push({ type: 'scatter', mode: 'markers', x: Array.from({ length: g.n }, (_, i) => g.P[3 * i]), y: Array.from({ length: g.n }, (_, i) => g.P[3 * i + 1]),
      text: g.ids.map((id, i) => `track ${id}<br>Δx ${g.U[0][i].toFixed(2)} Δy ${g.U[1][i].toFixed(2)} Δz ${g.U[2][i].toFixed(2)} µm`),
      marker: { size: g.n > 400 ? 2.5 : 4, color: 'rgba(30,30,30,0.55)' },
      hovertemplate: '%{text}<extra></extra>', showlegend: false });
  }
  const ax = (title, range) => ({ title: { text: title }, range, gridcolor: DARK.grid, zerolinecolor: DARK.zero, color: DARK.font });
  const layout = {
    paper_bgcolor: DARK.paper, plot_bgcolor: DARK.plot, font: { color: DARK.font, size: 12 },
    margin: { l: 52, r: 10, t: 28, b: 40 }, showlegend: false,
    title: { text: `deformation · frame ${a} → ${b} · ${QTY[DF.qty]}`, font: { size: 12 }, x: 0.02, y: 0.985 },
    uirevision: `map|${S.movie}|${S.v3.yDown}|${S.disp.a}`,
    xaxis: { ...ax('x (µm)', [G.lo[0], G.hi[0]]), constrain: 'domain' },
    yaxis: { ...ax('y (µm)', S.v3.yDown ? [G.hi[1], G.lo[1]] : [G.lo[1], G.hi[1]]), scaleanchor: 'x', constrain: 'domain' },
    hoverlabel: { bgcolor: '#222', font: { color: '#eee' } },
  };
  const done = Plotly.react(div, traces, layout, CFG);
  showAutoValues(h, sc);
  summary(g, h, t0, [DF.arrows ? `arrows ×${sc.toFixed(sc < 10 ? 1 : 0)} (lateral)` : '']);
  return done;
}

// ------------------------------------------------------------------ exports
async function postJson(path, body) {
  const r = await fetch(withRun(path), { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const j = await r.json().catch(() => ({ error: `HTTP ${r.status}` }));
  if (!r.ok) throw new Error(j.error || `HTTP ${r.status}`);
  return j;
}
const sleep = (ms) => new Promise((res) => setTimeout(res, ms));
const exportFrames = () => {
  const f0 = Math.max(1, S.filters.fStart || 1), f1 = Math.min(S.nFrames, S.filters.fEnd || S.nFrames);
  return Array.from({ length: f1 - f0 + 1 }, (_, i) => f0 + i);
};
let exportTask = null;
// frac: fraction done (null = unknown) for the progress bar
function setBusy(on, msg = '', frac = null) {
  exporting = on;
  for (const id of ['#dfExField', '#dfExMp4', '#dfExGif']) $(id).disabled = on;
  $('#dfExInfo').textContent = msg;
  if (on) { if (!exportTask) exportTask = startTask('deformation export'); exportTask.set(frac, msg.replace(/…$/, '')); }
  else if (exportTask) { exportTask.done(); exportTask = null; }
}

// the warped median plane (current depth offset) for every frame -> CSV, MAT and a VTK series
async function exportField() {
  if (!S.ds || !S.F) return;
  setBusy(true, 'fitting the interpolation…', 0);
  const h = await ensureHyperAsync({ set: (fr) => setBusy(true, 'fitting the interpolation…', 0.1 * fr) });
  if (!h || h.fail) { setBusy(false); throw new Error('not enough beads'); }
  try {
    const G = h.G, nx = G.xs.length, ny = G.ys.length, a = S.disp.a;
    const zSpan = G.hi[2] - G.lo[2], off = (0.5 - DF.depth) * zSpan;     // as sheetAt: slider right = deeper
    const Q = new Float64Array(3 * nx * ny), z0 = [];
    for (let j = 0; j < ny; j++) {
      const row = [];
      for (let i = 0; i < nx; i++) { const q = j * nx + i, z = h.plane(G.xs[i], G.ys[j]) + off; Q[3 * q] = G.xs[i]; Q[3 * q + 1] = G.ys[j]; Q[3 * q + 2] = z; row.push(dispZ(z)); }
      z0.push(row);
    }
    const frames = exportFrames();
    const out = { ux: [], uy: [], uz: [], areal_strain_pct: [], shear_strain_pct: [], tilt_deg: [] };
    const keep = DF.qty;
    for (const [n, f] of frames.entries()) {
      setBusy(true, `field: frame ${f} (${n + 1} / ${frames.length})…`, 0.1 + 0.8 * n / frames.length);
      await nextPaint();
      const g = gather(a, f);
      const pf = g.n >= MIN_BEADS ? condition(g.P, g.U, g.V, h.ell, h.s2) : null;
      const r = pf ? pf(Q, true) : null;
      const grid = (fn) => Array.from({ length: ny }, (_, j) => Array.from({ length: nx }, (_, i) => (r ? fn(j * nx + i) : NaN)));
      out.ux.push(grid((q) => r.u[0][q])); out.uy.push(grid((q) => r.u[1][q])); out.uz.push(grid((q) => dispDz(r.u[2][q])));
      for (const [k, name] of [['areal', 'areal_strain_pct'], ['shear', 'shear_strain_pct'], ['tilt', 'tilt_deg']]) {
        DF.qty = k; out[name].push(grid((q) => quantity(r, q)));
      }
      DF.qty = keep;
    }
    const nan0 = (A) => A.map((fr) => fr.map((row) => row.map((v) => (Number.isFinite(v) ? v : null))));
    for (const k of Object.keys(out)) out[k] = nan0(out[k]);
    setBusy(true, 'writing CSV, MAT and VTK…', 0.92);
    const r = await postJson('/api/export_field', {
      movie: S.movie, label: `deformation_ref${a}`, xs: Array.from(G.xs), ys: Array.from(G.ys), z0, frames, ...out,
      info: { reference_frame: a, coordinates: S.ds.coordLabel, smoothing_length_um: h.ell, field_amplitude_um2: h.s2,
        plane: 'robust median plane of the beads at the reference frame' + (off ? `, shifted ${off.toFixed(1)} µm` : ''),
        z_convention: 'up: z is height, positive toward the indenter (which comes from above), so an indentation is negative uz; the same convention as zMicrons', units: 'x, y, z, u in µm; strains in %; tilt in degrees',
        method: 'Gaussian-process (ordinary kriging) interpolation of bead displacements, squared-exponential covariance, per-bead errors; strains from its exact derivatives (in-plane only)' },
    });
    setBusy(false, `→ ${r.csv} (+ .mat, VTK series in ${r.vtk_dir})`);
    toast(`Deformation field exported (${r.frames} frames)\n${r.csv}\n${r.mat}\n${r.vtk_dir}`, 'ok', 8000);
    downloadUrl(r.csv_url, r.csv.split(/[\\/]/).pop());
  } finally { if (exporting) setBusy(false); }
}

// animation of the current view (camera, colouring, warp) over the frame range -> MP4 (or WebM) / GIF
async function exportAnimation(kind) {
  if (!S.ds || !S.F || !div.data) return;
  if (S.playing) $('#btnPlay').click();      // pause the movie while recording
  const frames = exportFrames(), fps = 10;
  const Wd = kind === 'gif' ? 800 : 1280, H0 = Math.round(Wd * div.clientHeight / Math.max(div.clientWidth, 1));
  const W = Wd - (Wd % 2), H = H0 - (H0 % 2);            // H.264 needs even sizes
  const shots = [];
  setBusy(true, `rendering 0 / ${frames.length} frames…`, 0);
  try {
    for (const [i, f] of frames.entries()) {
      await draw(f);
      shots.push(await Plotly.toImage(div, { format: 'png', width: W, height: H }));
      setBusy(true, `rendering ${i + 1} / ${frames.length} frames…`, 0.8 * (i + 1) / frames.length);
    }
    const label = `deformation_${DF.qty}_ref${S.disp.a}`;
    let r;
    if (kind === 'gif') {
      setBusy(true, 'making the GIF…', null);
      r = await postJson('/api/export_animation', { movie: S.movie, label, kind: 'gif', fps, frames: shots });
    } else {
      const mime = ['video/mp4;codecs=avc1.42E01E', 'video/mp4', 'video/webm;codecs=vp9', 'video/webm'].find((t) => MediaRecorder.isTypeSupported(t));
      if (!mime) throw new Error('this browser cannot record video; use GIF');
      setBusy(true, `recording the video (${(frames.length / fps).toFixed(0)} s)…`, 0.8);
      const c = document.createElement('canvas'); c.width = W; c.height = H;
      const ctx = c.getContext('2d');
      const stream = c.captureStream(0), track = stream.getVideoTracks()[0];
      const rec = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 8e6 });
      const chunks = []; rec.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
      const stopped = new Promise((res) => { rec.onstop = res; });
      const imgs = await Promise.all(shots.map((s) => new Promise((res, rej) => { const im = new Image(); im.onload = () => res(im); im.onerror = rej; im.src = s; })));
      rec.start();
      for (const [k, im] of imgs.entries()) {
        ctx.drawImage(im, 0, 0, W, H); track.requestFrame();
        if (k % 5 === 0) setBusy(true, `recording the video (${(frames.length / fps).toFixed(0)} s)…`, 0.8 + 0.18 * k / imgs.length);
        await sleep(1000 / fps);
      }
      await sleep(150);
      rec.stop(); await stopped;
      const blob = new Blob(chunks, { type: mime.split(';')[0] });
      const data = await new Promise((res) => { const fr = new FileReader(); fr.onload = () => res(fr.result); fr.readAsDataURL(blob); });
      r = await postJson('/api/export_animation', { movie: S.movie, label, kind: mime.includes('mp4') ? 'mp4' : 'webm', data });
    }
    setBusy(false, `→ ${r.path}`);
    toast(`Animation saved (${(r.bytes / 1e6).toFixed(1)} MB)\n${r.path}`, 'ok', 8000);
    downloadUrl(r.url, r.path.split(/[\\/]/).pop());
  } finally {
    if (exporting) setBusy(false);
    await draw();
  }
}
