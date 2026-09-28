// Shared helpers: event bus, colormaps, small math utilities.
import { withRun } from './state.js';

// ------------------------------------------------------------------ progress
// Long work shows a pill with a label, the elapsed time and a bar: filled to the fraction done when
// it is known (task.set(0..1)), a moving stripe otherwise. Several tasks may run; the newest is shown.
const tasks = new Map();
let taskSeq = 0, taskTimer = null;
function renderTasks() {
  const el = document.getElementById('busy');
  if (!el) return;
  if (!tasks.size) { el.classList.add('hidden'); clearInterval(taskTimer); taskTimer = null; return; }
  const t = [...tasks.values()].at(-1);
  el.classList.remove('hidden');
  const s = (performance.now() - t.t0) / 1000;
  document.getElementById('busyText').textContent = `${t.label}${t.detail ? ` · ${t.detail}` : ''}${s >= 0.5 ? `  ${s.toFixed(1)} s` : ''}`;
  const bar = document.getElementById('busyBar');
  const known = Number.isFinite(t.frac);
  bar.parentElement.classList.toggle('indet', !known);
  bar.style.width = known ? `${Math.round(100 * Math.min(1, Math.max(0, t.frac)))}%` : '';
  if (!taskTimer) taskTimer = setInterval(renderTasks, 200);
}
export function startTask(label) {
  const id = ++taskSeq;
  tasks.set(id, { label, detail: '', frac: null, t0: performance.now() });
  renderTasks();
  const h = {
    set(frac, detail) { const t = tasks.get(id); if (!t) return h; t.frac = frac; if (detail !== undefined) t.detail = detail; renderTasks(); return h; },
    label(text) { const t = tasks.get(id); if (t) { t.label = text; renderTasks(); } return h; },
    done() { tasks.delete(id); renderTasks(); },
  };
  return h;
}
// let the browser paint (the bar) before the next synchronous chunk of work
// (hidden pages get no animation frames, so a timer takes over: work never stalls in a background tab)
export const nextPaint = () => new Promise((res) => {
  let fired = false;
  const go = () => { if (!fired) { fired = true; setTimeout(res, 0); } };
  requestAnimationFrame(go);
  setTimeout(go, 60);
});

// fetch JSON reporting the download: onProgress(fraction or null, detail)
export async function fetchJsonProgress(url, onProgress, opts) {
  onProgress?.(null, 'waiting for the server');
  const r = await fetch(url, opts);
  if (!r.ok) {
    const j = await r.json().catch(() => ({ error: `HTTP ${r.status}` }));
    throw new Error(j.error || `HTTP ${r.status}`);
  }
  const total = +r.headers.get('Content-Length') || 0;
  if (!r.body || !r.body.getReader) return r.json();
  const reader = r.body.getReader(), parts = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value); got += value.length;
    onProgress?.(total ? got / total : null, `${(got / 1e6).toFixed(1)}${total ? ` / ${(total / 1e6).toFixed(1)}` : ''} MB`);
  }
  const buf = new Uint8Array(got);
  let o = 0;
  for (const p of parts) { buf.set(p, o); o += p.length; }
  onProgress?.(1, 'parsing');
  await nextPaint();
  return JSON.parse(new TextDecoder().decode(buf));
}

const listeners = {};
export function on(evt, fn) { (listeners[evt] ||= []).push(fn); }
export function emit(evt, arg) { (listeners[evt] || []).forEach((fn) => { try { fn(arg); } catch (e) { console.error(`[${evt}]`, e); } }); }

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

export function clamp(v, a, b) { return v < a ? a : v > b ? b : v; }
export function fmt(v, d = 2) { return Number.isFinite(v) ? v.toFixed(d) : '–'; }

export function debounce(fn, ms) {
  let t = null;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

// throttle with trailing call
export function throttle(fn, ms) {
  let last = 0, timer = null, pendingArgs = null;
  return (...a) => {
    const now = performance.now();
    pendingArgs = a;
    if (now - last >= ms) { last = now; fn(...a); pendingArgs = null; }
    else if (!timer) {
      timer = setTimeout(() => { timer = null; last = performance.now(); if (pendingArgs) fn(...pendingArgs); pendingArgs = null; }, ms - (now - last));
    }
  };
}

// ---------------------------------------------------------------- colormaps
function hexToRgb(h) { const n = parseInt(h.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }

function makeMap(stops) {
  const s = stops.map(([t, c]) => [t, typeof c === 'string' ? hexToRgb(c) : c]);
  const lut = new Array(256);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let k = 0;
    while (k < s.length - 2 && t > s[k + 1][0]) k++;
    const [t0, c0] = s[k], [t1, c1] = s[k + 1];
    const u = t1 > t0 ? clamp((t - t0) / (t1 - t0), 0, 1) : 0;
    lut[i] = `rgb(${Math.round(c0[0] + u * (c1[0] - c0[0]))},${Math.round(c0[1] + u * (c1[1] - c0[1]))},${Math.round(c0[2] + u * (c1[2] - c0[2]))})`;
  }
  const fn = (t) => lut[Number.isFinite(t) ? Math.round(clamp(t, 0, 1) * 255) : 0];
  fn.stops = stops;
  fn.plotly = s.map(([t, c]) => [t, `rgb(${c[0]},${c[1]},${c[2]})`]);
  return fn;
}

// MATLAB parula (approximation)
export const parula = makeMap([
  [0, '#352a87'], [0.125, '#0363e1'], [0.25, '#1485d4'], [0.375, '#06a7c6'], [0.5, '#38b99e'],
  [0.625, '#92bf73'], [0.75, '#d9ba56'], [0.875, '#fcce2e'], [1, '#f9fb0e'],
]);
// Turbo (sampled)
export const turbo = makeMap([
  [0, '#30123b'], [0.07, '#4145ab'], [0.14, '#4675ed'], [0.21, '#39a2fc'], [0.28, '#1bcfd4'],
  [0.35, '#24eca6'], [0.42, '#61fc6c'], [0.5, '#a4fc3b'], [0.57, '#d1e834'], [0.64, '#f3c63a'],
  [0.71, '#fe9b2d'], [0.78, '#f36315'], [0.85, '#d93806'], [0.92, '#b11901'], [1, '#7a0403'],
]);
// probability of moving: grey (still) -> yellow -> red (moving)
export const motionMap = makeMap([[0, '#6f7a88'], [0.5, '#f2c94c'], [1, '#ff4d4d']]);
// ordered stages (track_analysis --stages), by stage number 1..K: still, indent, hold, retract, still, ...
export const STAGE_COLORS = ['#8a94a3', '#ff5b5b', '#f2c94c', '#5aa9ff', '#c9cdd4', '#c792ea', '#7ee0b2'];
export const stageColor = (k) => (Number.isFinite(k) && k >= 1 ? STAGE_COLORS[(k - 1) % STAGE_COLORS.length] : '#555a63');
export const inferno = makeMap([
  [0, '#000004'], [0.13, '#1b0c41'], [0.25, '#4a0c6b'], [0.38, '#781c6d'], [0.5, '#a52c60'],
  [0.63, '#cf4446'], [0.75, '#ed6925'], [0.88, '#fb9b06'], [1, '#fcffa4'],
]);
// diverging blue-white-red (RdBu reversed so +dz = red)
export const rdbu = makeMap([
  [0, '#2166ac'], [0.25, '#67a9cf'], [0.4, '#d1e5f0'], [0.5, '#f7f7f7'], [0.6, '#fddbc7'],
  [0.75, '#ef8a62'], [1, '#b2182b'],
]);

// ------------------------------------------------------------ track colours
function hslToRgb(h, s, l) {
  s /= 100; l /= 100;
  const k = (n) => (n + h / 30) % 12;
  const a = s * Math.min(l, 1 - l);
  const f = (n) => l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)));
  return [Math.round(255 * f(0)), Math.round(255 * f(8)), Math.round(255 * f(4))];
}
export function trackHue01(id) { return ((id * 137.50776) % 360) / 360; }
const trackColorCache = new Map();
export function trackColor(id) {
  let c = trackColorCache.get(id);
  if (!c) { const [r, g, b] = hslToRgb(trackHue01(id) * 360, 85, 60); c = `rgb(${r},${g},${b})`; trackColorCache.set(id, c); }
  return c;
}
// cyclic hue colourscale matching trackColor(), for plotly numeric colour arrays
// darker / lighter shade of a track's colour (e.g. Δx dark, Δy light in the same plot)
export function trackShade(id, lightness) {
  const [r, g, b] = hslToRgb(trackHue01(id) * 360, 85, lightness);
  return `rgb(${r},${g},${b})`;
}

export const hueScale =Array.from({ length: 25 }, (_, i) => {
  const [r, g, b] = hslToRgb(360 * i / 24, 85, 60);
  return [i / 24, `rgb(${r},${g},${b})`];
});

// ------------------------------------------------------------------- geometry
export function pointInPoly(x, y, pts) {
  let inside = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const [xi, yi] = pts[i], [xj, yj] = pts[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}
export function roiContains(roi, x, y) {
  if (roi.type === 'rect') {
    const [[x0, y0], [x1, y1]] = roi.pts;
    return x >= Math.min(x0, x1) && x <= Math.max(x0, x1) && y >= Math.min(y0, y1) && y <= Math.max(y0, y1);
  }
  return pointInPoly(x, y, roi.pts);
}

export function percentile(arr, p) {
  const a = Array.from(arr).filter(Number.isFinite).sort((u, v) => u - v);
  if (!a.length) return NaN;
  const k = clamp((a.length - 1) * p / 100, 0, a.length - 1);
  const lo = Math.floor(k), hi = Math.ceil(k);
  return a[lo] + (a[hi] - a[lo]) * (k - lo);
}

// binary search: first index with arr[i] >= v
export function lowerBound(arr, v) {
  let lo = 0, hi = arr.length;
  while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] < v) lo = m + 1; else hi = m; }
  return lo;
}

export function toast(msg, kind = 'info', ms = 4000) {
  const el = document.createElement('div');
  el.className = `toast ${kind}`;
  el.textContent = msg;
  document.getElementById('toasts').appendChild(el);
  setTimeout(() => el.remove(), ms);
}

export function downloadBlob(blob, name) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
}
export function downloadUrl(url, name) {
  const a = document.createElement('a');
  a.href = withRun(url); a.download = name || '';
  document.body.appendChild(a); a.click(); setTimeout(() => a.remove(), 500);
}
