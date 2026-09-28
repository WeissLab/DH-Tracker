// Movie canvas: frame loading/caching, pan/zoom, overlays, hit-testing, ROI tools.
//
// Performance notes
//  * Overlays are built once per (frame, filters, selection, overlay options) into
//    Path2D objects in IMAGE coordinates, bucketed by colour; pan/zoom only re-strokes
//    the cached paths under a new transform (no per-bead JS work).
//  * Redraws are coalesced with requestAnimationFrame.
//  * At low zoom only bead centres are drawn (no lobes / labels).
//  * Frames: binned (fast) versions are preloaded first, full resolution on idle.
import { S, PX_UM } from './state.js';
import { on, emit, clamp, fmt, trackColor, turbo, parula, inferno, rdbu, roiContains, percentile, motionMap, stageColor } from './util.js';
import { frameRowsOk, rowAt } from './data.js';

export const perf = { draw: [], build: [] };
const note = (arr, v) => { arr.push(v); if (arr.length > 120) arr.shift(); };

// ------------------------------------------------------------------ frame cache
const MAX_CACHE = 260, MAX_INFLIGHT = 4;
const cache = new Map();   // key -> {img, url, state:'queued'|'loading'|'ok'|'err'}
const queue = [];          // pending {key,url} (front = highest priority)
let inflight = 0;

const stabOn = () => (S.stab && S.drift ? 1 : 0);
function frameKey(frame, bin) {
  const c = S.contrast;
  return `${S.run}|${S.movie}|${frame}|${bin}|${c.vmin}|${c.vmax}|${c.gamma}|${stabOn()}`;
}
function frameUrl(frame, bin) {
  const c = S.contrast;
  // the run id keeps the browser's HTTP cache apart for same-named movies of different runs
  return `/api/movie/${encodeURIComponent(S.movie)}/frame/${frame - 1}?vmin=${c.vmin}&vmax=${c.vmax}&gamma=${c.gamma}&bin=${bin}${stabOn() ? '&stab=1' : ''}&run=${encodeURIComponent(S.run || '')}`;
}
function touch(key) { const e = cache.get(key); if (e) { cache.delete(key); cache.set(key, e); } return e; }
function evict() {
  let guard = cache.size;
  while (cache.size > MAX_CACHE && guard-- > 0) {
    const [k, e] = cache.entries().next().value;
    cache.delete(k);
    if (e.state === 'loading' || e.frame === S.frame) { cache.set(k, e); continue; }
    if (e.url) URL.revokeObjectURL(e.url);
  }
}
function pump() {
  while (inflight < MAX_INFLIGHT && queue.length) {
    const { key, url } = queue.shift();
    const e = cache.get(key);
    if (!e || e.state !== 'queued') continue;
    e.state = 'loading';
    inflight++;
    fetch(url).then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.blob(); })
      .then((b) => new Promise((res, rej) => {
        const img = new Image();
        e.url = URL.createObjectURL(b);
        img.onload = () => {
          // pre-decode off the draw path when possible (decode() can stall in hidden pages,
          // so never wait on it)
          if (img.decode) img.decode().catch(() => {});
          res(img);
        };
        img.onerror = rej;
        img.src = e.url;
      }))
      .then((img) => { e.img = img; e.state = 'ok'; })
      .catch((err) => { e.state = 'err'; console.warn('frame load failed', key, err); cache.delete(key); })
      .finally(() => { inflight--; pump(); if (e.state === 'ok') onFrameLoaded(e); if (!queue.length && !inflight) idleFill(); });
  }
}
export function requestFrame(frame, bin, priority = false) {
  if (!S.movie || frame < 1 || frame > S.nFrames) return null;
  const key = frameKey(frame, bin);
  let e = touch(key);
  if (e) {
    if (priority && e.state === 'queued') { const i = queue.findIndex((q) => q.key === key); if (i > 0) queue.unshift(queue.splice(i, 1)[0]); }
    return e;
  }
  e = { key, frame, bin, state: 'queued' };
  cache.set(key, e);
  const item = { key, url: frameUrl(frame, bin) };
  if (priority) queue.unshift(item); else queue.push(item);
  evict();
  pump();
  return e;
}
export function clearQueue() { for (const q of queue) cache.delete(q.key); queue.length = 0; }
// drop every cached frame (switching to another run)
export function clearFrames() {
  clearQueue();
  for (const [k, e] of cache) { if (e.state === 'loading') continue; if (e.url) URL.revokeObjectURL(e.url); cache.delete(k); }
  lastImg = null;
}
function onFrameLoaded(e) { if (e.frame === S.frame) redraw(); cacheBarDirty(); }

// best available image for a frame: requested bin first, then any other resolution
function bestImage(frame, bin) {
  const e = cache.get(frameKey(frame, bin));
  if (e && e.state === 'ok') return e.img;
  for (const b of [1, 2, 4]) { if (b === bin) continue; const e2 = cache.get(frameKey(frame, b)); if (e2 && e2.state === 'ok') return e2.img; }
  return null;
}
export function isFrameReady(frame) { return !!bestImage(frame, currentBin()); }
export function frameEntry(frame) { return requestFrame(frame, currentBin(), true); }

let cacheBarTimer = null;
function cacheBarDirty() { if (!cacheBarTimer) cacheBarTimer = setTimeout(() => { cacheBarTimer = null; updateCacheBar(); }, 300); }
export function updateCacheBar() {
  const el = document.getElementById('cacheInfo');
  if (!el || !S.ui.drawer) return;   // hidden drawer: skip
  const cnt = (b) => { let n = 0; for (let f = 1; f <= S.nFrames; f++) { const e = cache.get(frameKey(f, b)); if (e && e.state === 'ok') n++; } return n; };
  el.textContent = `cached: ${cnt(1)} full · ${cnt(2)} ½ · ${cnt(4)} ¼ of ${S.nFrames}`;
}
export function preloadAll(bin = currentBin()) { for (let f = 1; f <= S.nFrames; f++) requestFrame(f, bin); }
// background strategy: all frames binned (cheap) first, then full-res near the current frame on idle
export function backgroundPreload() {
  preloadAll(Math.max(2, currentBin()));
}
function idleFill() {
  if (!S.movie || currentBin() !== 1) return;
  const ric = window.requestIdleCallback || ((fn) => setTimeout(fn, 200));
  ric(() => {
    for (let d = -8; d <= 16; d++) {
      const f = S.frame + d;
      if (f >= 1 && f <= S.nFrames) requestFrame(f, 1);
    }
  });
}

// ------------------------------------------------------------------ view / canvas
let canvas, ctx, wrap, tip;
let W = 2048, H = 2048;               // image size
const view = { s: 0.4, tx: 0, ty: 0 }; // screen = u*s + t  (CSS px)
let cssW = 100, cssH = 100, dpr = 1;
let lastImg = null;
let drawPending = false;
export const autoRes = { on: true };

export function currentBin() {
  if (!autoRes.on) return 1;
  const eff = view.s * dpr;
  return eff >= 0.7 ? 1 : eff >= 0.3 ? 2 : 4;
}

export function initMovie() {
  canvas = document.getElementById('movieCanvas');
  wrap = document.getElementById('movieWrap');
  tip = document.getElementById('tooltip');
  ctx = canvas.getContext('2d', { alpha: false });
  new ResizeObserver(resize).observe(wrap);
  resize();
  bindMouse();
  on('frame', () => { redraw(); prefetch(); });
  ['filter', 'selection', 'overlay', 'data'].forEach((ev) => on(ev, () => { invalidate(); redraw(); }));
  on('disp', () => { dispCache = null; redraw(); });
  on('rois', redraw);
}

function resize() {
  dpr = window.devicePixelRatio || 1;
  cssW = wrap.clientWidth; cssH = wrap.clientHeight;
  canvas.width = Math.max(1, Math.round(cssW * dpr));
  canvas.height = Math.max(1, Math.round(cssH * dpr));
  canvas.style.width = cssW + 'px'; canvas.style.height = cssH + 'px';
  redraw();
}

export function setImageSize(w, h) { W = w; H = h; lastImg = null; }
export function fitView() {
  const s = Math.min(cssW / W, cssH / H) * 0.98;
  view.s = s; view.tx = (cssW - W * s) / 2; view.ty = (cssH - H * s) / 2;
  redraw();
}
export function centerOn(u, v, minScale = 1.5) {
  view.s = Math.max(view.s, minScale);
  view.tx = cssW / 2 - u * view.s; view.ty = cssH / 2 - v * view.s;
  redraw();
}
// Pan (keeping the zoom) so image point (u, v) is on screen; no-op when it already is.
export function reveal(u, v, margin = 40) {
  const [x, y] = toScreen(u, v);
  if (x >= margin && x <= cssW - margin && y >= margin && y <= cssH - margin) return;
  view.tx = cssW / 2 - u * view.s; view.ty = cssH / 2 - v * view.s;
  redraw();
}
export function setView(s, u, v) { view.s = s; view.tx = cssW / 2 - u * s; view.ty = cssH / 2 - v * s; redraw(); }
export function getView() { return { ...view }; }
// image coords -> client (page) coords; used by tests / tooling
export function imageToClient(u, v) { const r = canvas.getBoundingClientRect(); const [x, y] = toScreen(u, v); return [r.left + x, r.top + y]; }
const toScreen = (u, v) => [u * view.s + view.tx, v * view.s + view.ty];
const toImage = (sx, sy) => [(sx - view.tx) / view.s, (sy - view.ty) / view.s];

export function prefetch() {
  const bin = currentBin();
  requestFrame(S.frame, bin, true);
  const ahead = S.playing ? 12 : 3;
  for (let d = 1; d <= ahead; d++) {
    let f = S.frame + d;
    if (f > S.nFrames) { if (S.loop && S.playing) f = ((f - 1) % S.nFrames) + 1; else break; }
    requestFrame(f, bin);
  }
  if (!S.playing) for (let d = 1; d <= 2; d++) requestFrame(S.frame - d, bin);
}

export function redraw() {
  if (drawPending) return;
  drawPending = true;
  requestAnimationFrame(() => { drawPending = false; draw(); });
}

// ------------------------------------------------------------------ colouring
export function zRange() {
  if (!S.ov.zcAuto) return [S.ov.zcMin, S.ov.zcMax];
  const r = S.ds ? S.ds.zAuto : [-5, 5];
  return Number.isFinite(r[0]) && r[1] > r[0] ? r : [-5, 5];
}
const NBIN = 32;
// colour bucket key for a row (quantized so that one Path2D serves many beads)
function rowColor(r) {
  const ds = S.ds;
  switch (S.ov.colorBy) {
    case 'track': return trackColor(ds.tid[r]);
    case 'frame': return parula(Math.round(NBIN * (ds.frame[r] - 1) / Math.max(1, S.nFrames - 1)) / NBIN);
    case 'white': return '#ffffff';
    case 'motion': {
      const p = ds.cols.pMoving ? ds.cols.pMoving[r] : NaN;
      return Number.isFinite(p) ? motionMap(Math.round(NBIN * p) / NBIN) : '#9a9a9a';
    }
    case 'stage': return stageColor(ds.cols.stage ? ds.cols.stage[r] : NaN);
    default: {
      const z = ds.Z[r];
      if (!Number.isFinite(z)) return '#9a9a9a';
      const [a, b] = zRange();
      return turbo(Math.round(NBIN * clamp((z - a) / (b - a), 0, 1)) / NBIN);
    }
  }
}

// ------------------------------------------------------------------ displacement field
let dispCache = null;
export function computeDisplacement() {
  const ds = S.ds, F = S.F;
  const out = [];
  if (!ds || !F) return out;
  const a = S.disp.a, b = S.disp.followB ? S.frame : S.disp.b;
  for (let k = 0; k < ds.ids.length; k++) {
    if (!F.trackOk[k]) continue;
    if (S.disp.onlySel && !S.sel.has(ds.ids[k])) continue;
    const ra = rowAt(F, k, a), rb = rowAt(F, k, b);
    if (ra < 0 || rb < 0) continue;
    const dxp = ds.X[rb] - ds.X[ra], dyp = ds.Y[rb] - ds.Y[ra];
    const dz = ds.Z[rb] - ds.Z[ra];
    const mag = Math.hypot(dxp * PX_UM, dyp * PX_UM, Number.isFinite(dz) ? dz : 0);
    out.push({ k, id: ds.ids[k], ra, rb, dxp, dyp, dz, mag });
  }
  return out;
}
function dispData() {
  const key = `${S.disp.a}|${S.disp.followB ? S.frame : S.disp.b}|${S.disp.scale}|${S.disp.onlySel}|${S.disp.zmapAt}`;
  if (dispCache && dispCache.key === key && dispCache.F === S.F && dispCache.ds === S.ds && dispCache.sel === S.sel) return dispCache;
  const D = computeDisplacement();
  const magMax = Math.max(1e-6, percentile(D.map((d) => d.mag), 98) || 1);
  const dzMax = Math.max(1e-6, percentile(D.map((d) => Math.abs(d.dz)), 98) || 1);
  const ds = S.ds, sc = S.disp.scale;
  // arrows bucketed by colour (8 bins), in image coords; heads are added at draw time
  const arrows = new Map(), discs = new Map();
  for (const d of D) {
    const col = inferno(0.15 + 0.85 * Math.round(8 * clamp(d.mag / magMax, 0, 1)) / 8);
    const u0 = ds.ux[d.ra], v0 = ds.uy[d.ra];
    if (!arrows.has(col)) arrows.set(col, []);
    arrows.get(col).push([u0, v0, u0 + sc * d.dxp, v0 + sc * d.dyp, d.id]);
    const r = S.disp.zmapAt === 'a' ? d.ra : d.rb;
    const dc = Number.isFinite(d.dz) ? rdbu(0.5 + 0.5 * Math.round(10 * clamp(d.dz / dzMax, -1, 1)) / 20) : '#666';
    if (!discs.has(dc)) discs.set(dc, new Path2D());
    const p = discs.get(dc); p.moveTo(ds.ux[r] + 14, ds.uy[r]); p.arc(ds.ux[r], ds.uy[r], 14, 0, 2 * Math.PI);
  }
  dispCache = { key, F: S.F, ds: S.ds, sel: S.sel, D, magMax, dzMax, arrows, discs };
  S._dispLegend = { magMax, dzMax, n: D.length };
  return dispCache;
}

// ------------------------------------------------------------------ overlay cache (Path2D, image coords)
let ovCache = null;
let ovVersion = 0;
export function invalidate() { ovVersion++; }

function trackVisible(k) {
  const id = S.ds.ids[k];
  if (!S.F.trackOk[k]) return false;
  if (S.ov.onlySel && S.sel.size && !S.sel.has(id)) return false;
  return true;
}

const R_CENTER = 12, R_LOBE = 2.5;
function bucket(map, key) { let b = map.get(key); if (!b) { b = { c: new Path2D(), l: new Path2D(), cd: new Path2D(), ld: new Path2D(), dashed: false }; map.set(key, b); } return b; }

function buildOverlay() {
  const t0 = performance.now();
  const ds = S.ds, F = S.F, f = S.frame;
  const dim = S.sel.size > 0 && S.ov.dimUnsel;
  const normal = new Map(), faded = new Map(), tails = new Map(), tailsFaded = new Map();
  const selRows = [], selTails = [];
  // full path of each selected track (all frames), drawn faintly under everything else
  const selPaths = new Path2D();
  for (const id of S.sel) {
    const k = ds.kOf.get(id);
    if (k === undefined || !trackVisible(k)) continue;
    F.tRows[k].forEach((r, j) => (j ? selPaths.lineTo(ds.ux[r], ds.uy[r]) : selPaths.moveTo(ds.ux[r], ds.uy[r])));
  }
  // tails
  if (S.ov.tails && S.ov.tailN > 0) {
    const N = S.ov.tailN;
    for (let k = 0; k < ds.ids.length; k++) {
      if (!trackVisible(k)) continue;
      const fr = F.tFrames[k], rows = F.tRows[k];
      if (!fr.length || fr[0] > f || fr[fr.length - 1] < f - N) continue;
      let j1 = fr.length - 1; while (j1 >= 0 && fr[j1] > f) j1--;
      if (j1 < 1) continue;
      const id = ds.ids[k];
      const sel = S.sel.has(id);
      if (sel) {
        const p = new Path2D(); let first = true;
        for (let j = j1; j >= 0 && fr[j] >= f - N; j--) { const r = rows[j]; first ? p.moveTo(ds.ux[r], ds.uy[r]) : p.lineTo(ds.ux[r], ds.uy[r]); first = false; }
        selTails.push([rowColor(rows[j1]), p]);
        continue;
      }
      const target = dim ? tailsFaded : tails;
      let prev = rows[j1];
      for (let j = j1 - 1; j >= 0 && fr[j] >= f - N; j--) {
        const r = rows[j];
        const col = rowColor(prev);
        let p = target.get(col); if (!p) { p = new Path2D(); target.set(col, p); }
        p.moveTo(ds.ux[prev], ds.uy[prev]); p.lineTo(ds.ux[r], ds.uy[r]);
        prev = r;
      }
    }
  }
  // current-frame localizations
  if (S.ov.beads) {
    for (const r of frameRowsOk(ds, F, f)) {
      const k = ds.rowTrack[r];
      if (k < 0 || !trackVisible(k)) continue;
      if (S.sel.has(ds.tid[r])) { selRows.push(r); continue; }
      const b = bucket(dim ? faded : normal, rowColor(r));
      const rec = ds.recovered[r] && S.ov.recoveredDashed;
      const pc = rec ? b.cd : b.c, pl = rec ? b.ld : b.l;
      if (rec) b.dashed = true;
      const u = ds.ux[r], v = ds.uy[r];
      pc.moveTo(u + R_CENTER, v); pc.arc(u, v, R_CENTER, 0, 2 * Math.PI);
      pl.moveTo(ds.ux1[r], ds.uy1[r]); pl.lineTo(ds.ux2[r], ds.uy2[r]);
      pl.moveTo(ds.ux1[r] + R_LOBE, ds.uy1[r]); pl.arc(ds.ux1[r], ds.uy1[r], R_LOBE, 0, 2 * Math.PI);
      pl.moveTo(ds.ux2[r] + R_LOBE, ds.uy2[r]); pl.arc(ds.ux2[r], ds.uy2[r], R_LOBE, 0, 2 * Math.PI);
    }
  }
  // grey sets: rejected file and short tracks from re-tracking
  const grey = { c: new Path2D(), l: new Path2D(), n: 0 };
  for (const [G, show] of [[S.rej, S.ov.rejected], [S.shortDs, S.ov.short]]) {
    if (!G || !show || f < 1 || f > G.nFrames) continue;
    for (let p = G.fstart[f]; p < G.fstart[f + 1]; p++) {
      const r = G.frameRows[p];
      grey.c.moveTo(G.ux[r] + R_CENTER * 0.8, G.uy[r]); grey.c.arc(G.ux[r], G.uy[r], R_CENTER * 0.8, 0, 2 * Math.PI);
      grey.l.moveTo(G.ux1[r], G.uy1[r]); grey.l.lineTo(G.ux2[r], G.uy2[r]);
      grey.n++;
    }
  }
  ovCache = { version: ovVersion, frame: f, F, ds, normal, faded, tails, tailsFaded, selRows, selTails, selPaths, grey };
  note(perf.build, performance.now() - t0);
  return ovCache;
}

function overlay() {
  if (ovCache && ovCache.version === ovVersion && ovCache.frame === S.frame && ovCache.F === S.F && ovCache.ds === S.ds) return ovCache;
  return buildOverlay();
}

// ------------------------------------------------------------------ drawing
function draw() {
  if (!ctx) return;
  const t0 = performance.now();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = '#07080b';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (!S.movie) return;
  const bin = currentBin();
  requestFrame(S.frame, bin, true);
  let img = bestImage(S.frame, bin);
  let stale = false;
  if (!img && lastImg) { img = lastImg; stale = true; }
  ctx.setTransform(dpr * view.s, 0, 0, dpr * view.s, dpr * view.tx, dpr * view.ty);
  ctx.imageSmoothingEnabled = view.s * dpr < 2;
  if (img) { ctx.drawImage(img, 0, 0, W, H); if (!stale) lastImg = img; }
  if (S.ds && S.F && S.ov.show) drawOverlays();
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  drawRois();
  drawHud(stale);
  note(perf.draw, performance.now() - t0);
}

function drawOverlays() {
  const s = view.s;
  const px = 1 / s;                      // one screen pixel in image units
  const O = overlay();
  const detail = s >= 0.35;             // low zoom: centres only
  // image-coordinate transform is active. Round joins: the default miter join draws long spikes
  // wherever a noisy track reverses direction, which looks like a jagged path.
  ctx.lineJoin = 'round';
  if (S.disp.on || S.disp.zmap) drawDisplacement(px);
  // grey sets
  if (O.grey.n) {
    ctx.strokeStyle = '#8a8f98'; ctx.globalAlpha = 0.55; ctx.lineWidth = px;
    ctx.setLineDash([2 * px, 3 * px]);
    ctx.stroke(O.grey.c); if (detail && S.ov.lobes) ctx.stroke(O.grey.l);
    ctx.setLineDash([]); ctx.globalAlpha = 1;
  }
  // whole path of selected tracks (round caps so separate tail segments meet without notches)
  ctx.lineCap = 'round';
  if (O.selPaths) { ctx.strokeStyle = '#ffffff'; ctx.globalAlpha = 0.55; ctx.lineWidth = 1.2 * px; ctx.stroke(O.selPaths); ctx.globalAlpha = 1; }
  // tails
  ctx.lineWidth = 2.5 * px;
  ctx.globalAlpha = 0.3 * 0.85;
  for (const [col, p] of O.tailsFaded) { ctx.strokeStyle = col; ctx.stroke(p); }
  ctx.globalAlpha = 0.85;
  for (const [col, p] of O.tails) { ctx.strokeStyle = col; ctx.stroke(p); }
  ctx.lineWidth = 3.5 * px;
  for (const [col, p] of O.selTails) { ctx.strokeStyle = col; ctx.stroke(p); }
  ctx.globalAlpha = 1; ctx.lineCap = 'butt';
  // beads
  const strokeBuckets = (map, alpha) => {
    ctx.globalAlpha = alpha;
    for (const [col, b] of map) {
      ctx.strokeStyle = col;
      ctx.lineWidth = 1.5 * px; ctx.stroke(b.c);
      if (detail && S.ov.lobes) { ctx.lineWidth = 1.2 * px; ctx.stroke(b.l); }
      if (b.dashed) {
        ctx.setLineDash([4 * px, 3 * px]);
        ctx.lineWidth = 1.5 * px; ctx.stroke(b.cd);
        if (detail && S.ov.lobes) { ctx.lineWidth = 1.2 * px; ctx.stroke(b.ld); }
        ctx.setLineDash([]);
      }
    }
  };
  strokeBuckets(O.faded, 0.3);
  strokeBuckets(O.normal, 1);
  ctx.globalAlpha = 1;
  // selected beads (few): drawn individually
  const ds = S.ds;
  for (const r of O.selRows) {
    const u = ds.ux[r], v = ds.uy[r];
    const col = rowColor(r);   // same colour scale as everything else, just bolder
    const rec = ds.recovered[r] && S.ov.recoveredDashed;
    if (detail && S.ov.lobes) {
      ctx.strokeStyle = col; ctx.lineWidth = 2.5 * px;
      ctx.beginPath(); ctx.moveTo(ds.ux1[r], ds.uy1[r]); ctx.lineTo(ds.ux2[r], ds.uy2[r]);
      ctx.moveTo(ds.ux1[r] + R_LOBE, ds.uy1[r]); ctx.arc(ds.ux1[r], ds.uy1[r], R_LOBE, 0, 2 * Math.PI);
      ctx.moveTo(ds.ux2[r] + R_LOBE, ds.uy2[r]); ctx.arc(ds.ux2[r], ds.uy2[r], R_LOBE, 0, 2 * Math.PI);
      ctx.stroke();
    }
    ctx.setLineDash(rec ? [4 * px, 3 * px] : []);
    ctx.beginPath(); ctx.arc(u, v, R_CENTER * 1.35, 0, 2 * Math.PI);
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 6 * px; ctx.stroke();
    ctx.strokeStyle = col; ctx.lineWidth = 3 * px; ctx.stroke();
    ctx.setLineDash([]);
  }
  // hover ring
  if (S.hover !== null && S.hover >= 0 && ds.frame[S.hover] === S.frame) {
    ctx.strokeStyle = '#fff'; ctx.lineWidth = px; ctx.setLineDash([2 * px, 2 * px]);
    ctx.beginPath(); ctx.arc(ds.ux[S.hover], ds.uy[S.hover], R_CENTER + 4 * px, 0, 2 * Math.PI); ctx.stroke(); ctx.setLineDash([]);
  }
  // labels for selected beads (screen space)
  if (s > 0.6 && O.selRows.length) {
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = '#fff'; ctx.font = '12px system-ui, sans-serif';
    for (const r of O.selRows) { const [x, y] = toScreen(ds.ux[r], ds.uy[r]); ctx.fillText(String(ds.tid[r]), x + R_CENTER * s + 4, y - R_CENTER * s); }
    ctx.setTransform(dpr * view.s, 0, 0, dpr * view.s, dpr * view.tx, dpr * view.ty);
  }
}

function drawDisplacement(px) {
  const Dc = dispData();
  if (S.disp.zmap) {
    ctx.globalAlpha = 0.85;
    for (const [col, p] of Dc.discs) { ctx.fillStyle = col; ctx.fill(p); }
    ctx.globalAlpha = 1;
  }
  if (!S.disp.on) return;
  // shafts + heads in screen space (heads need constant pixel size); cull to viewport
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const m = 40;
  const dim = S.sel.size > 0 && S.ov.dimUnsel;
  ctx.lineWidth = 2;
  for (const [col, list] of Dc.arrows) {
    ctx.strokeStyle = col; ctx.fillStyle = col;
    for (const pass of [0, 1]) {            // pass 0 = dimmed (unselected), 1 = full
      ctx.globalAlpha = pass ? 1 : 0.3;
      const shafts = new Path2D(), heads = new Path2D();
      let any = false;
      for (const [u0, v0, u1, v1, id] of list) {
        if (dim && (S.sel.has(id) ? 0 : 1) === pass) continue;
        if (!dim && pass === 0) continue;
        const [x0, y0] = toScreen(u0, v0), [x1, y1] = toScreen(u1, v1);
        if ((x0 < -m && x1 < -m) || (x0 > cssW + m && x1 > cssW + m) || (y0 < -m && y1 < -m) || (y0 > cssH + m && y1 > cssH + m)) continue;
        any = true;
        const L = Math.hypot(x1 - x0, y1 - y0);
        if (L < 3) { heads.moveTo(x0 + 1.5, y0); heads.arc(x0, y0, 1.5, 0, 2 * Math.PI); continue; }
        shafts.moveTo(x0, y0); shafts.lineTo(x1, y1);
        const h = Math.min(9, 0.35 * L + 3), a = Math.atan2(y1 - y0, x1 - x0);
        heads.moveTo(x1, y1);
        heads.lineTo(x1 - h * Math.cos(a - 0.4), y1 - h * Math.sin(a - 0.4));
        heads.lineTo(x1 - h * Math.cos(a + 0.4), y1 - h * Math.sin(a + 0.4));
        heads.closePath();
      }
      if (any) { ctx.stroke(shafts); ctx.fill(heads); }
    }
  }
  ctx.globalAlpha = 1;
  ctx.setTransform(dpr * view.s, 0, 0, dpr * view.s, dpr * view.tx, dpr * view.ty);
}

function drawRois() {
  ctx.lineWidth = 1.5;
  const path = (roi) => {
    ctx.beginPath();
    if (roi.type === 'rect') {
      const [[x0, y0], [x1, y1]] = roi.pts;
      const [a, b] = toScreen(x0 - 0.5, y0 - 0.5), [c, d] = toScreen(x1 - 0.5, y1 - 0.5);
      ctx.rect(a, b, c - a, d - b);
    } else {
      roi.pts.forEach(([x, y], i) => { const [a, b] = toScreen(x - 0.5, y - 0.5); i ? ctx.lineTo(a, b) : ctx.moveTo(a, b); });
      if (roi.closed !== false) ctx.closePath();
    }
  };
  ctx.setLineDash([6, 4]);
  S.rois.forEach((roi, i) => {
    const col = roi.op === 'remove' ? '#ff6b6b' : '#ffd400';
    ctx.strokeStyle = col;
    path(roi); ctx.stroke();
    const [x, y] = roi.pts[0]; const [a, b] = toScreen(x - 0.5, y - 0.5);
    const sign = roi.op === 'add' ? '+ ' : roi.op === 'remove' ? '− ' : '';
    ctx.fillStyle = col; ctx.font = '12px system-ui'; ctx.fillText(`${sign}ROI${i + 1}`, a + 3, b - 4);
  });
  if (drawing) {
    ctx.strokeStyle = '#00e5ff';
    path(drawing); ctx.stroke();
    if (drawing.type === 'poly' && drawing.closed === false && drawing.cursor) {
      const [x, y] = drawing.pts[drawing.pts.length - 1];
      const [a, b] = toScreen(x - 0.5, y - 0.5);
      ctx.beginPath(); ctx.moveTo(a, b); ctx.lineTo(drawing.cursor[0], drawing.cursor[1]); ctx.stroke();
    }
  }
  ctx.setLineDash([]);
}

function colorbar(x, y, w, h, cmap, lo, hi, label, dec = 1) {
  const g = ctx.createLinearGradient(x, 0, x + w, 0);
  for (let i = 0; i <= 16; i++) g.addColorStop(i / 16, cmap(i / 16));
  ctx.fillStyle = 'rgba(0,0,0,0.55)'; ctx.fillRect(x - 8, y - 18, w + 16, h + 38);
  ctx.fillStyle = g; ctx.fillRect(x, y, w, h);
  ctx.fillStyle = '#d8dce2'; ctx.font = '12px system-ui, sans-serif';
  ctx.textAlign = 'left'; ctx.fillText(label, x, y - 5);
  ctx.fillText(fmt(lo, dec), x, y + h + 14);
  ctx.textAlign = 'right'; ctx.fillText(fmt(hi, dec), x + w, y + h + 14);
  ctx.textAlign = 'left';
}

function drawHud(stale) {
  ctx.font = '13px system-ui, sans-serif';
  const hud = `frame ${S.frame} / ${S.nFrames}${stabOn() ? '  ·  stabilized' : ''}${S.tracking ? '  ·  re-tracked' : ''}${stale ? '  ·  loading…' : ''}`;
  ctx.fillStyle = 'rgba(0,0,0,0.5)'; ctx.fillRect(10, cssH - 34, ctx.measureText(hud).width + 18, 24);
  ctx.fillStyle = stale ? '#ffb347' : '#e8e8e8';
  ctx.fillText(hud, 19, cssH - 17);
  if (S.ov.scalebar) {
    const target = 120 / view.s * PX_UM;
    const nice = [1, 2, 5, 10, 20, 50, 100, 200, 500].reduce((b, v) => (v <= target ? v : b), 1);
    const len = nice / PX_UM * view.s;
    const x = cssW - len - 24, y = cssH - 22;
    ctx.fillStyle = 'rgba(0,0,0,0.5)'; ctx.fillRect(x - 8, y - 20, len + 16, 30);
    ctx.fillStyle = '#fff'; ctx.fillRect(x, y, len, 3);
    ctx.textAlign = 'center'; ctx.fillText(`${nice} µm`, x + len / 2, y - 6); ctx.textAlign = 'left';
  }
  if (!S.ov.legend || !S.ds || !S.ov.show) return;
  let lx = cssW - 190;
  let ly = 34;
  if (S.ov.colorBy === 'z') { const [a, b] = zRange(); colorbar(lx, ly, 160, 8, turbo, a, b, 'z (µm)'); ly += 52; }
  if (S.ov.colorBy === 'frame') { colorbar(lx, ly, 160, 8, parula, 1, S.nFrames, 'frame', 0); ly += 52; }
  if (S.ov.colorBy === 'motion') { colorbar(lx, ly, 160, 8, motionMap, 0, 1, 'P(moving)', 1); ly += 52; }
  if (S.ov.colorBy === 'stage' && S.stageNames) {       // categorical legend: one swatch per stage
    const names = S.stageNames;
    ctx.fillStyle = 'rgba(0,0,0,0.55)'; ctx.fillRect(lx - 8, ly - 16, 168, 20 + 18 * names.length);
    ctx.fillStyle = '#e8e8e8'; ctx.fillText('stage', lx, ly);
    names.forEach((nm, i) => {
      ctx.fillStyle = stageColor(i + 1); ctx.fillRect(lx, ly + 8 + 18 * i, 14, 12);
      ctx.fillStyle = '#e8e8e8'; ctx.fillText(`${i + 1}. ${nm}`, lx + 22, ly + 18 + 18 * i);
    });
    ly += 30 + 18 * names.length;
  }
  const L = S._dispLegend;
  if (L && S.disp.on) { colorbar(lx, ly, 160, 8, (t) => inferno(0.15 + 0.85 * t), 0, L.magMax, `|Δr| 3D µm (arrows ×${S.disp.scale})`, 2); ly += 52; }
  if (L && S.disp.zmap) { colorbar(lx, ly, 160, 8, rdbu, -L.dzMax, L.dzMax, 'Δz (µm)', 2); }
}

// ------------------------------------------------------------------ hit test / tooltip
function hitTest(sx, sy, maxDist = 14) {
  if (!S.ds || !S.F || !S.ov.show) return -1;
  const ds = S.ds;
  let best = -1;
  const rC = Math.max(5, R_CENTER * view.s);
  let bd = Math.max(maxDist * maxDist, rC * rC);
  for (const r of frameRowsOk(ds, S.F, S.frame)) {
    const k = ds.rowTrack[r];
    if (k < 0 || !trackVisible(k)) continue;
    for (const [u, v] of [[ds.ux[r], ds.uy[r]], [ds.ux1[r], ds.uy1[r]], [ds.ux2[r], ds.uy2[r]]]) {
      const [x, y] = toScreen(u, v);
      const d = (x - sx) ** 2 + (y - sy) ** 2;
      if (d < bd) { bd = d; best = r; }
    }
  }
  return best;
}

const TIP_SKIP = new Set(['x1', 'x2', 'xMean', 'y1', 'y2', 'yMean', 'angleDegrees', 'zMicrons', 'frame_number', 'track_number', 'xStabilized', 'yStabilized', 'zStabilized']);
export function describeRow(r, ds = S.ds, title = 'track') {
  const c = ds.cols;
  const x = c.xMean[r], y = c.yMean[r];
  const lines = [
    `<b>${title} ${ds.tid[r]}</b> · frame ${ds.frame[r]}`,
    `x ${fmt(x, 2)} px (${fmt(x * PX_UM, 2)} µm) · y ${fmt(y, 2)} px (${fmt(y * PX_UM, 2)} µm)`,
    `z ${fmt(c.zMicrons[r], 3)} µm · angle ${fmt(c.angleDegrees[r], 1)}°`,
  ];
  if (ds.pipelineTid) lines.push(`<span class="q">pipeline track ${ds.pipelineTid[r]}</span>`);
  const coordSet = ds.coords || {};
  if (coordSet.corr || coordSet.xy || coordSet.z) lines.push(`<span class="q">${ds.coordLabel}: x ${fmt(ds.X[r], 2)}, y ${fmt(ds.Y[r], 2)} px, z ${fmt(ds.Z[r], 3)} µm</span>`);
  const q = [];
  for (const k of ds.columns) {
    if (TIP_SKIP.has(k)) continue;
    const v = c[k][r];
    q.push(`${k} ${fmt(v, Number.isInteger(v) ? 0 : 3)}`);
  }
  if (q.length) lines.push(`<span class="q">${q.join(' · ')}</span>`);
  return lines.join('<br>');
}

function hitTestGrey(sx, sy, maxDist = 12) {
  const f = S.frame;
  let best = null, bd = maxDist * maxDist;
  for (const [G, show, title] of [[S.rej, S.ov.rejected, 'REJECTED track'], [S.shortDs, S.ov.short, 'SHORT track']]) {
    if (!G || !show || f < 1 || f > G.nFrames) continue;
    for (let p = G.fstart[f]; p < G.fstart[f + 1]; p++) {
      const r = G.frameRows[p];
      const [x, y] = toScreen(G.ux[r], G.uy[r]);
      const d = (x - sx) ** 2 + (y - sy) ** 2;
      if (d < bd) { bd = d; best = [G, r, title]; }
    }
  }
  return best;
}

function showTip(r, sx, sy) {
  if (r < 0) {
    const g = S.ov.show ? hitTestGrey(sx, sy) : null;
    if (!g) { tip.style.display = 'none'; return; }
    tip.innerHTML = describeRow(g[1], g[0], g[2]);
  } else tip.innerHTML = describeRow(r);
  tip.style.display = 'block';
  const tw = tip.offsetWidth, th = tip.offsetHeight;
  tip.style.left = Math.min(sx + 16, cssW - tw - 4) + 'px';
  tip.style.top = Math.min(sy + 16, cssH - th - 4) + 'px';
}

// ------------------------------------------------------------------ mouse & tools
let drawing = null;       // ROI in progress
let drag = null;          // {sx, sy, tx, ty, moved, button}
export let shiftDown = false;

function evPos(e) { const r = canvas.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; }
function toData(sx, sy) { const [u, v] = toImage(sx, sy); return [u + 0.5, v + 0.5]; } // one-based px

let hoverPending = null;
function bindMouse() {
  window.addEventListener('keydown', (e) => { if (e.key === 'Shift') shiftDown = true; });
  window.addEventListener('keyup', (e) => { if (e.key === 'Shift') shiftDown = false; });
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const [sx, sy] = evPos(e);
    const [u, v] = toImage(sx, sy);
    const k = Math.exp(-e.deltaY * (e.deltaMode === 1 ? 0.05 : 0.0015));
    view.s = clamp(view.s * k, 0.05, 40);
    view.tx = sx - u * view.s; view.ty = sy - v * view.s;
    tip.style.display = 'none';
    redraw();
    zoomSettled();
  }, { passive: false });
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  canvas.addEventListener('mousedown', (e) => {
    const [sx, sy] = evPos(e);
    canvas.focus();
    const panBtn = e.button === 1 || e.button === 2 || (e.button === 0 && (S.tool === 'pan' || (e.altKey && S.tool !== 'poly')));
    if (panBtn) { drag = { sx, sy, tx: view.tx, ty: view.ty, moved: false, button: e.button, pan: true }; canvas.style.cursor = 'grabbing'; tip.style.display = 'none'; return; }
    if (e.button !== 0) return;
    const p = toData(sx, sy);
    if (S.tool === 'rect') drawing = { type: 'rect', pts: [p, p.slice()] };
    else if (S.tool === 'lasso') drawing = { type: 'poly', pts: [p], lasso: true };
    else if (S.tool === 'poly') {
      if (!drawing) drawing = { type: 'poly', pts: [p], closed: false };
      else drawing.pts.push(p);
    }
    drag = { sx, sy, moved: false, button: 0, pan: false };
    redraw();
  });
  window.addEventListener('mousemove', (e) => {
    const [sx, sy] = evPos(e);
    if (drag) {
      if (Math.hypot(sx - drag.sx, sy - drag.sy) > 3) drag.moved = true;
      if (drag.pan) { view.tx = drag.tx + sx - drag.sx; view.ty = drag.ty + sy - drag.sy; redraw(); return; }
      if (drawing && drawing.type === 'rect') { drawing.pts[1] = toData(sx, sy); redraw(); }
      else if (drawing && drawing.lasso) {
        const last = drawing.pts[drawing.pts.length - 1];
        const [lx, ly] = toScreen(last[0] - 0.5, last[1] - 0.5);
        if (Math.hypot(sx - lx, sy - ly) > 3) { drawing.pts.push(toData(sx, sy)); redraw(); }
      }
      return;
    }
    if (e.target !== canvas) { if (tip.style.display !== 'none') tip.style.display = 'none'; return; }
    if (drawing && drawing.type === 'poly' && drawing.closed === false) { drawing.cursor = [sx, sy]; redraw(); }
    // hover work is coalesced to one per animation frame
    if (!hoverPending) requestAnimationFrame(() => {
      const [hx, hy] = hoverPending; hoverPending = null;
      const r = hitTest(hx, hy);
      const [xd, yd] = toData(hx, hy);
      document.getElementById('cursorInfo').textContent = `x ${xd.toFixed(1)}  y ${yd.toFixed(1)} px   (${(xd * PX_UM).toFixed(1)}, ${(yd * PX_UM).toFixed(1)} µm)`;
      if (r !== S.hover) { S.hover = r; redraw(); }
      showTip(r, hx, hy);
    });
    hoverPending = [sx, sy];
  });
  window.addEventListener('mouseup', (e) => {
    if (!drag) return;
    const d = drag; drag = null;
    canvas.style.cursor = '';
    const [sx, sy] = evPos(e);
    if (d.pan) {
      if (!d.moved && d.button === 0) clickSelect(sx, sy, e.shiftKey || e.ctrlKey || e.metaKey);
      else if (d.moved) zoomSettled();
      return;
    }
    if (!drawing) return;
    if (drawing.type === 'rect') {
      if (!d.moved) { drawing = null; clickSelect(sx, sy, e.shiftKey || e.ctrlKey); return; }
      finishRoi(drawing, e);
    } else if (drawing.lasso) {
      if (drawing.pts.length < 3) { drawing = null; clickSelect(sx, sy, e.shiftKey || e.ctrlKey); return; }
      finishRoi(drawing, e);
    }
  });
  canvas.addEventListener('dblclick', (e) => {
    if (S.tool === 'poly' && drawing) {
      drawing.pts.pop();   // the dblclick's two mousedowns added a duplicate vertex
      if (drawing.pts.length >= 3) finishRoi(drawing, e); else { drawing = null; redraw(); }
    } else if (S.tool === 'pan') {
      const r = hitTest(...evPos(e));
      if (r >= 0) centerOn(S.ds.ux[r], S.ds.uy[r], Math.max(view.s * 2, 2));
    }
  });
  canvas.addEventListener('mouseleave', () => { tip.style.display = 'none'; if (S.hover !== null) { S.hover = null; redraw(); } });
}

const zoomSettled = (() => { let t; return () => { clearTimeout(t); t = setTimeout(() => { prefetch(); cacheBarDirty(); }, 150); }; })();

export function polyKey(key, e) {
  if (!drawing || drawing.type !== 'poly' || drawing.closed !== false) return false;
  if (key === 'Enter') { if (drawing.pts.length >= 3) finishRoi(drawing, e); return true; }
  if (key === 'Escape') { drawing = null; redraw(); return true; }
  if (key === 'Backspace') { drawing.pts.pop(); if (!drawing.pts.length) drawing = null; redraw(); return true; }
  return false;
}
export function cancelDrawing() { drawing = null; redraw(); }

function finishRoi(roi, e) {
  const mode = e && e.altKey ? 'remove' : e && (e.shiftKey || e.ctrlKey || shiftDown) ? 'add' : 'replace';
  // op: how this ROI changed the selection. A plain ROI replaces the selection, so earlier
  // outlines no longer describe it and are dropped; Shift (+) / Alt (−) ROIs are kept.
  const r = { type: roi.type, op: mode, pts: roi.pts.map(([x, y]) => [+x.toFixed(2), +y.toFixed(2)]) };
  drawing = null;
  if (mode === 'replace') S.rois = [];
  S.rois.push(r);
  const ids = tracksInRoi(r, S.roiMode);
  emit('roiSelect', { ids, mode });
  emit('rois');
}

export function tracksInRoi(roi, mode) {
  const ds = S.ds, F = S.F, out = new Set();
  if (!ds || !F) return out;
  // ROIs are drawn on the displayed image, so test the drawn (image-plane) bead centres
  const inR = (r) => roiContains(roi, ds.ux[r] + 0.5, ds.uy[r] + 0.5);
  if (mode === 'current') {
    for (const r of frameRowsOk(ds, F, S.frame)) if (inR(r)) out.add(ds.tid[r]);
    return out;
  }
  for (let k = 0; k < ds.ids.length; k++) {
    if (!F.trackOk[k]) continue;
    for (const r of F.tRows[k]) if (inR(r)) { out.add(ds.ids[k]); break; }
  }
  return out;
}

function clickSelect(sx, sy, additive) {
  const r = hitTest(sx, sy);
  if (r < 0) return;
  emit('pickTrack', { id: S.ds.tid[r], additive, source: 'movie' });
}

// synchronous draw (benchmarks / tooling)
export function drawNow() { draw(); }
export function resizeNow() { resize(); }
