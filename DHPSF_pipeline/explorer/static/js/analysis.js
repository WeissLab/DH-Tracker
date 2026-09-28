// "New analysis" modal: file browser, TIFF inspection, job launch and progress polling.
import { $, toast } from './util.js';

const PRESET_PX = { '20x': 0.325, '10x': 0.63 };
const POLL_MS = 2000;
// jittered so polling cannot phase-lock with the engine's status writes (Windows file locking)
const pollDelay = () => POLL_MS * (0.8 + 0.4 * Math.random());
let api, post, openRun, refreshRuns;

const form = { movies: [], calInfo: null, calCheck: null, objTouched: false, nameTouched: false, zTouched: false };
let job = null;          // {id, run_id, name, state, acknowledged}
let pollTimer = null;
let lastLog = '';

export function isModalOpen() {
  return !$('#anModal').classList.contains('hidden') || !$('#fbModal').classList.contains('hidden');
}

// ------------------------------------------------------------------ small helpers
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const baseName = (p) => String(p).split(/[\\/]/).filter(Boolean).pop() || p;
function fmtSize(b) {
  if (b >= 1e9) return `${(b / 1e9).toFixed(1)} GB`;
  if (b >= 1e6) return `${(b / 1e6).toFixed(0)} MB`;
  if (b >= 1e3) return `${(b / 1e3).toFixed(0)} kB`;
  return `${b} B`;
}
function fmtDate(sec) {
  const d = new Date(sec * 1000), p = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
}
function fmtDur(s) {
  s = Math.max(0, Math.round(s));
  const m = Math.floor(s / 60), r = s % 60;
  return m ? `${m} min ${String(r).padStart(2, '0')} s` : `${r} s`;
}
const infoText = (i) => `${i.frames} frames · ${i.width}×${i.height} px${i.exposure_ms ? ` · ${i.exposure_ms} ms exposure` : ''}`;
// pasted Windows "Copy as path" values come wrapped in quotes, possibly several per line
const cleanPath = (p) => String(p || '').trim().replace(/^["']+|["']+$/g, '').trim();
const splitPaths = (text) => String(text || '').split(/\r?\n|"\s+"/).map(cleanPath).filter(Boolean);
const store = {
  get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* storage unavailable */ } },
};

// ------------------------------------------------------------------ file browser
const fb = { cwd: null, parent: null, dirs: [], files: [], sel: new Set(), anchor: null, multi: false, resolve: null };

function joinPath(dir, name) {
  if (dir === 'drives') return name;
  const sep = dir.includes('\\') || /^[A-Za-z]:/.test(dir) ? '\\' : '/';
  return dir.replace(/[\\/]+$/, '') + sep + name;
}

function browseFiles({ title, multi = false, start = null }) {
  fb.multi = multi; fb.sel = new Set(); fb.anchor = null;
  $('#fbTitle').textContent = title;
  $('#fbModal').classList.remove('hidden');
  // start: the folder of a path already typed, else the last folder used
  let dir = start && /[\\/]/.test(start) ? start.replace(/[\\/][^\\/]*\.tiff?$/i, '') : null;
  fbLoad(dir || store.get('dhpsf.browseDir') || '', true);
  setTimeout(() => $('#fbList').focus(), 0);
  return new Promise((res) => { fb.resolve = res; });
}

async function fbLoad(path, fallback = false) {
  $('#fbList').innerHTML = '<div class="fb-empty">loading…</div>';
  try {
    const r = await api(`/api/browse?path=${encodeURIComponent(path)}`);
    Object.assign(fb, { cwd: r.path, parent: r.parent, dirs: r.dirs, files: r.files, sel: new Set(), anchor: null });
    if (r.path !== 'drives') store.set('dhpsf.browseDir', r.path);
    fbRender();
  } catch (e) {
    if (fallback && path) return fbLoad('', false);
    $('#fbList').innerHTML = `<div class="fb-empty">${esc(e.message)}</div>`;
  }
}

function fbCrumbs() {
  const el = $('#fbCrumbs');
  const items = [['Drives', 'drives']];
  if (fb.cwd && fb.cwd !== 'drives') {
    const win = /^[A-Za-z]:/.test(fb.cwd);
    const parts = fb.cwd.split(/[\\/]/).filter(Boolean);
    let acc = win ? '' : '/';
    parts.forEach((p, i) => {
      acc = win ? (i === 0 ? `${p}\\` : `${acc.replace(/\\$/, '')}\\${p}`) : `${acc.replace(/\/$/, '')}/${p}`;
      items.push([p, acc]);
    });
  }
  el.innerHTML = items.map(([label, path], i) => `${i ? '<span class="sl">›</span>' : ''}<button data-path="${esc(path)}">${esc(label)}</button>`).join('');
}

function fbRender() {
  fbCrumbs();
  const rows = [];
  if (fb.parent) rows.push(`<div class="fb-row" data-kind="up" title="Up one folder"><span class="ic">↰</span><span class="nm">..</span><span></span><span></span></div>`);
  for (const d of fb.dirs) rows.push(`<div class="fb-row" data-kind="dir" data-name="${esc(d)}"><span class="ic">📁</span><span class="nm">${esc(d)}</span><span></span><span></span></div>`);
  for (const f of fb.files) {
    rows.push(`<div class="fb-row${fb.sel.has(f.name) ? ' sel' : ''}" data-kind="file" data-name="${esc(f.name)}" title="${esc(f.name)}"><span class="ic">▦</span><span class="nm">${esc(f.name)}</span><span class="sz">${fmtSize(f.size)}</span><span class="dt">${fmtDate(f.mtime)}</span></div>`);
  }
  if (!fb.dirs.length && !fb.files.length) rows.push('<div class="fb-empty">no sub-folders or .tif files here</div>');
  $('#fbList').innerHTML = rows.join('');
  fbSelChanged();
}

function fbSelChanged() {
  const n = fb.sel.size;
  $('#fbOk').disabled = !n;
  $('#fbSelInfo').textContent = n === 0 ? (fb.multi ? 'Ctrl/Shift-click to pick several files' : '') : n === 1 ? [...fb.sel][0] : `${n} files`;
  document.querySelectorAll('#fbList .fb-row[data-kind=file]').forEach((r) => r.classList.toggle('sel', fb.sel.has(r.dataset.name)));
}

function fbClose(result) {
  $('#fbModal').classList.add('hidden');
  const r = fb.resolve; fb.resolve = null;
  if (r) r(result);
}
const fbChoose = () => { if (fb.sel.size) fbClose([...fb.sel].map((n) => joinPath(fb.cwd, n))); };

function bindBrowser() {
  $('#fbCrumbs').addEventListener('click', (e) => { const b = e.target.closest('button[data-path]'); if (b) fbLoad(b.dataset.path); });
  const list = $('#fbList');
  list.addEventListener('click', (e) => {
    const row = e.target.closest('.fb-row'); if (!row || row.dataset.kind !== 'file') return;
    const name = row.dataset.name;
    if (fb.multi && (e.ctrlKey || e.metaKey)) { fb.sel.has(name) ? fb.sel.delete(name) : fb.sel.add(name); fb.anchor = name; }
    else if (fb.multi && e.shiftKey && fb.anchor) {
      const names = fb.files.map((f) => f.name), a = names.indexOf(fb.anchor), b = names.indexOf(name);
      fb.sel = new Set(names.slice(Math.min(a, b), Math.max(a, b) + 1));
    } else { fb.sel = new Set([name]); fb.anchor = name; }
    fbSelChanged();
  });
  list.addEventListener('dblclick', (e) => {
    const row = e.target.closest('.fb-row'); if (!row) return;
    if (row.dataset.kind === 'up') fbLoad(fb.parent);
    else if (row.dataset.kind === 'dir') fbLoad(joinPath(fb.cwd, row.dataset.name));
    else { fb.sel.add(row.dataset.name); fbChoose(); }
  });
  list.addEventListener('keydown', (e) => {
    if (e.key === 'Backspace' && fb.parent) { e.preventDefault(); fbLoad(fb.parent); }
    else if (e.key === 'Enter') { e.preventDefault(); fbChoose(); }
  });
  $('#fbOk').onclick = fbChoose;
  $('#fbCancel').onclick = () => fbClose(null);
  $('#fbClose').onclick = () => fbClose(null);
}

// ------------------------------------------------------------------ calibration range
const range = { n: 0, a: 1, b: 1, path: null, width: 0 };
const clampInt = (v, lo, hi) => Math.min(hi, Math.max(lo, Math.round(+v || lo)));
const fmtUm = (v) => (Math.abs(v - Math.round(v)) < 1e-6 ? String(Math.round(v)) : v.toFixed(1));
function binFor(width) { let b = 1; while (b < 8 && width / (b * 2) >= 200) b *= 2; return b; }
const frameUrl = (path, frame, bin) => `/api/preview_frame?path=${encodeURIComponent(path)}&frame=${frame}&bin=${bin}`;

function seedRange(info, path) {
  const n = info && info.frames > 1 ? info.frames : 0;
  Object.assign(range, { n, path: n ? path : null, width: info ? info.width : 0 });
  $('#anRange').classList.toggle('hidden', !n);
  $('#anRangeEmpty').classList.toggle('hidden', !!n);
  $('#anRangeEmpty').textContent = info ? 'Single-plane file: no range to choose.' : 'Choose the calibration stack first.';
  if (!n) { updateSpan(); return; }
  for (const id of ['#anPlaneA', '#anPlaneB', '#anFirst', '#anLast']) { $(id).min = 1; $(id).max = n; }
  const sp = Array.isArray(info.suggested_planes) && info.suggested_planes.length === 2 ? info.suggested_planes : [1, n];
  setRange(sp[0], sp[1]);
}

let thumbTimer = null;
function setRange(a, b, moved = null) {
  const n = range.n;
  a = clampInt(a, 1, n); b = clampInt(b, 1, n);
  if (a > b) { if (moved === 'b') a = b; else b = a; }
  range.a = a; range.b = b;
  $('#anPlaneA').value = a; $('#anPlaneB').value = b; $('#anFirst').value = a; $('#anLast').value = b;
  const pct = (v) => (n > 1 ? ((v - 1) / (n - 1)) * 100 : 0);
  $('#anDualFill').style.left = `${pct(a)}%`; $('#anDualFill').style.right = `${100 - pct(b)}%`;
  $('#anPlaneA').style.zIndex = a > n / 2 ? 3 : 1;   // keep both handles grabbable when they meet
  updateSpan();
  clearTimeout(thumbTimer);
  thumbTimer = setTimeout(() => {
    const bin = binFor(range.width);
    $('#anThumbA').src = frameUrl(range.path, range.a, bin);
    $('#anThumbB').src = frameUrl(range.path, range.b, bin);
  }, moved ? 90 : 0);
}

function updateSpan() {
  if (!range.n) { $('#anSpan').textContent = ''; return; }
  const dz = parseFloat($('#anZStep').value) > 0 ? parseFloat($('#anZStep').value) : 1;
  const span = (range.b - range.a) * dz;
  $('#anSpan').textContent = `planes ${range.a}–${range.b} of ${range.n} · ${fmtUm(span)} µm (±${fmtUm(span / 2)} µm)`;
  updateSteps();
}

// ------------------------------------------------------------------ detection threshold preview
const thr = { path: null, info: null, frame: 1, img: null, data: null, token: 0, touched: false,
  view: { s: 0, tx: 0, ty: 0 }, infos: new Map(), cache: new Map() };

function snrValue() { return parseFloat($('#anSnr').value); }
function updateThrSummary() {
  const d = thr.data;
  $('#anSnrVal').textContent = snrValue().toFixed(1);
  $('#anThrSummary').textContent = thr.touched ? `strictness ${snrValue().toFixed(1)} × noise`
    : d && Number.isFinite(d.suggested_min_snr) ? `auto (suggested ≥ ${(+d.suggested_min_snr).toFixed(1)})` : 'auto';
  $('#anSnrAuto').classList.toggle('hidden', !thr.touched);
}

async function thrInit() {
  const has = form.movies.length > 0;
  $('#anThr').classList.toggle('hidden', !has);
  $('#anThrEmpty').classList.toggle('hidden', has);
  if (!has || !$('#anThrSec').open) return;
  const sel = $('#anThrMovie');
  sel.innerHTML = form.movies.map((m) => `<option value="${esc(m.path)}">${esc(baseName(m.path))}</option>`).join('');
  sel.classList.toggle('hidden', form.movies.length < 2);
  const keep = form.movies.some((m) => m.path === thr.path);
  if (!keep) thr.path = form.movies[0].path;
  sel.value = thr.path;
  if (!keep || !thr.info) await thrLoadMovie();
  else thrDraw();
}

async function thrLoadMovie() {
  const path = thr.path;
  thr.info = null; thr.img = null; thr.data = null; thr.view.s = 0;
  thrDraw();
  try {
    let info = thr.infos.get(path);
    if (!info) { info = await api(`/api/file_info?path=${encodeURIComponent(path)}`); thr.infos.set(path, info); }
    if (thr.path !== path) return;
    thr.info = info;
    $('#anThrFrame').max = info.frames;
    thrSetFrame(Math.ceil(info.frames / 2), true);
  } catch (e) { thrMessage(`Cannot read ${baseName(path)}: ${e.message}`); }
}

function thrSetFrame(f, detect) {
  const info = thr.info; if (!info) return;
  thr.frame = clampInt(f, 1, info.frames);
  $('#anThrFrame').value = thr.frame;
  $('#anThrFrameLbl').textContent = `${thr.frame} / ${info.frames}`;
  const img = new Image();
  img.onload = () => { if (thr.img === img) thrDraw(); };
  img.src = frameUrl(thr.path, thr.frame, info.width > 1400 ? 2 : 1);
  thr.img = img;
  if (detect) thrDetect();
}

async function thrDetect() {
  const key = `${thr.path}|${thr.frame}`;
  const tok = ++thr.token;
  thrMessage('');
  const cached = thr.cache.get(key);
  if (cached) { thrGotData(cached); return; }
  thr.data = null; thrDraw();
  $('#anThrBusy').classList.remove('hidden');
  try {
    const d = await api(`/api/preview?path=${encodeURIComponent(thr.path)}&frame=${thr.frame}`);
    thr.cache.set(key, d);
    if (tok === thr.token) thrGotData(d);
  } catch (e) {
    if (tok === thr.token) thrMessage(`Bead detection failed: ${e.message}`);
  } finally {
    if (tok === thr.token) $('#anThrBusy').classList.add('hidden');
  }
}

function thrGotData(d) {
  thr.data = d;
  if (!thr.touched) {
    const v = Number.isFinite(d.suggested_min_snr) ? d.suggested_min_snr : Number.isFinite(d.default_min_snr) ? d.default_min_snr : 4;
    $('#anSnr').value = v;
  }
  updateThrSummary();
  thrDraw();
}

function thrMessage(msg) {
  let el = $('#anThrView .an-thr-msg');
  if (!msg) { if (el) el.remove(); return; }
  if (!el) { el = document.createElement('div'); el.className = 'an-thr-msg'; $('#anThrView').appendChild(el); }
  el.textContent = msg;
}

function thrDraw() {
  const cv = $('#anThrCanvas'), box = $('#anThrView');
  const cw = box.clientWidth, ch = box.clientHeight, dpr = window.devicePixelRatio || 1;
  if (!cw || !ch) return;
  if (cv.width !== Math.round(cw * dpr) || cv.height !== Math.round(ch * dpr)) { cv.width = Math.round(cw * dpr); cv.height = Math.round(ch * dpr); }
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = '#07080b'; ctx.fillRect(0, 0, cw, ch);
  const info = thr.info;
  if (!info) { $('#anSnrCount').textContent = ''; return; }
  const W = info.width, H = info.height, v = thr.view;
  if (!v.s) { v.s = Math.min(cw / W, ch / H); v.tx = (cw - W * v.s) / 2; v.ty = (ch - H * v.s) / 2; }
  if (thr.img && thr.img.complete && thr.img.naturalWidth) {
    ctx.imageSmoothingEnabled = v.s < 2;
    ctx.drawImage(thr.img, v.tx, v.ty, W * v.s, H * v.s);
  }
  const d = thr.data;
  if (!d || !Array.isArray(d.beads)) { $('#anSnrCount').textContent = ''; return; }
  const min = snrValue();
  const X = (x) => (x - 0.5) * v.s + v.tx, Y = (y) => (y - 0.5) * v.s + v.ty;   // one-based pixel centres
  const r = Math.max(2, Math.min(8, 2.5 * v.s));
  const kept = new Path2D(), dropped = new Path2D();
  let nKept = 0;
  for (const b of d.beads) {
    const ok = Number.isFinite(b.snr) && b.snr >= min;
    if (ok) nKept++;
    const p = ok ? kept : dropped;
    const x1 = X(b.x1), y1 = Y(b.y1), x2 = X(b.x2), y2 = Y(b.y2);
    p.moveTo(x1, y1); p.lineTo(x2, y2);
    p.moveTo(x1 + r, y1); p.arc(x1, y1, r, 0, 2 * Math.PI);
    p.moveTo(x2 + r, y2); p.arc(x2, y2, r, 0, 2 * Math.PI);
  }
  ctx.lineWidth = 1;
  ctx.strokeStyle = 'rgba(255, 91, 91, 0.45)'; ctx.stroke(dropped);
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = '#3ecf6e'; ctx.stroke(kept);
  $('#anSnrCount').textContent = `${nKept} of ${d.beads.length} beads kept`;
}

function bindThreshold() {
  $('#anThrSec').addEventListener('toggle', () => { if ($('#anThrSec').open) thrInit(); });
  $('#anThrMovie').addEventListener('change', (e) => { thr.path = e.target.value; thrLoadMovie(); });
  $('#anThrFrame').addEventListener('input', (e) => thrSetFrame(+e.target.value, false));   // image follows at once
  $('#anThrFrame').addEventListener('change', (e) => thrSetFrame(+e.target.value, true));  // detection on release
  $('#anSnr').addEventListener('input', () => { thr.touched = true; updateThrSummary(); thrDraw(); });
  $('#anSnrAuto').onclick = () => { thr.touched = false; if (thr.data) thrGotData(thr.data); else updateThrSummary(); };
  const box = $('#anThrView');
  box.addEventListener('wheel', (e) => {
    if (!thr.info) return;
    e.preventDefault();
    const rc = box.getBoundingClientRect(), mx = e.clientX - rc.left, my = e.clientY - rc.top, v = thr.view;
    const f = e.deltaY < 0 ? 1.25 : 0.8, s2 = Math.min(40, Math.max(0.05, v.s * f));
    v.tx = mx - (mx - v.tx) * (s2 / v.s); v.ty = my - (my - v.ty) * (s2 / v.s); v.s = s2;
    thrDraw();
  }, { passive: false });
  let drag = null;
  box.addEventListener('pointerdown', (e) => { drag = { x: e.clientX, y: e.clientY, tx: thr.view.tx, ty: thr.view.ty }; box.setPointerCapture(e.pointerId); box.style.cursor = 'grabbing'; });
  box.addEventListener('pointermove', (e) => { if (!drag) return; thr.view.tx = drag.tx + e.clientX - drag.x; thr.view.ty = drag.ty + e.clientY - drag.y; thrDraw(); });
  box.addEventListener('pointerup', () => { drag = null; box.style.cursor = ''; });
  box.addEventListener('dblclick', () => { thr.view.s = 0; thrDraw(); });
}

// ------------------------------------------------------------------ form
async function inspect(path) {
  return api(`/api/inspect?path=${encodeURIComponent(path)}`);
}

function updatePx() {
  const v = $('#anObj').value;
  $('#anPxWrap').classList.toggle('hidden', v !== 'custom');
  if (v !== 'custom') $('#anPx').value = v ? PRESET_PX[v] : '';
  updateSteps();
}

// Magnification from the file / folder names. Never silently assumed: if nothing says it, the
// user must choose; if the calibration and a movie disagree, say so.
function autoObjective() {
  const infos = [form.calInfo, ...form.movies.map((m) => m.info)].filter(Boolean);
  const guesses = infos.filter((i) => i.objective_guess);
  const kinds = [...new Set(guesses.map((i) => i.objective_guess))];
  const note = $('#anObjNote');
  note.classList.remove('an-warn');
  if (kinds.length > 1) {
    note.textContent = `⚠ The files look like different objectives (${guesses.map((i) => `${baseName(i.path)}: ${i.objective_guess.replace('x', '×')}`).join(', ')}). A calibration only applies to movies taken with the same objective.`;
    note.classList.add('an-warn');
  } else if (kinds.length === 1) {
    const g = guesses[0];
    note.textContent = `${g.objective_guess.replace('x', '×')}, from the name “${g.objective_source || baseName(g.path)}”.`;
  } else note.textContent = infos.length ? 'The magnification is not in the file or folder names; please choose it.' : '';
  if (form.objTouched) return;
  if (kinds.length !== 1) { $('#anObj').value = ''; updatePx(); return; }   // unknown or conflicting: the user decides
  if (PRESET_PX[kinds[0]]) $('#anObj').value = kinds[0];
  else if (guesses[0].pixel_size_guess_um) { $('#anObj').value = 'custom'; $('#anPx').value = guesses[0].pixel_size_guess_um; }
  else return;
  updatePx();
}

const zStepValue = () => { const v = parseFloat($('#anZStep').value); return v > 0 ? v : null; };

// z step between calibration planes from the stack's name and metadata (unless the user typed one)
function autoZStep(i) {
  const note = $('#anZNote');
  if (!i) { note.textContent = ''; return; }
  if (i.z_step_guess_um) {
    if (!form.zTouched) $('#anZStep').value = i.z_step_guess_um;
    note.textContent = `z step ${i.z_step_guess_um} µm: ${i.z_step_source}.`;
  } else {
    // never assume a value: a wrong z step scales every z by the same wrong factor
    note.textContent = `z step not found (${i.z_step_source || 'unknown'}). Enter the distance between calibration planes in µm `
      + '(stage speed × frame interval, e.g. 25 µm/s × 0.04 s = 1 µm).';
    $('#anCalMore').open = true;
  }
  $('#anZStep').classList.toggle('need', !zStepValue());
  updateSpan();
}

// is the chosen calibration a usable z-scan? (a few seconds: beads and lobe rotation on 3 planes)
async function checkCalibration(path) {
  const el = $('#anCalCheck');
  form.calCheck = null;
  el.classList.remove('hidden', 'an-warn', 'an-ok');
  el.textContent = 'checking the calibration… (10–30 s)';
  try {
    const c = await api(`/api/check_calibration?path=${encodeURIComponent(path)}`);
    if (cleanPath($('#anCal').value) !== path) return;
    form.calCheck = c;
    const cached = c.cache === 'current' ? ' Its fit is cached, so the analysis starts right away.' : '';
    el.textContent = (c.verdict === 'ok' ? '✓ ' : '⚠ ') + c.message + (c.verdict === 'ok' ? cached : '');
    el.classList.add(c.verdict === 'ok' ? 'an-ok' : 'an-warn');
  } catch (e) {
    el.textContent = `calibration check unavailable: ${e.message}`;
    form.calCheck = { verdict: 'ok', message: '', cache: 'none', unchecked: true };   // do not block on a failed check
  }
  updateSteps();
}

function autoName() {
  if (form.nameTouched) return;
  const m = form.movies.find((x) => x.info && x.info.default_run_name);
  $('#anName').value = m ? m.info.default_run_name : '';
}

async function setCalibration(path) {
  path = cleanPath(path);
  $('#anCal').value = path;
  form.calInfo = null; form.calCheck = null;
  const el = $('#anCalInfo');
  seedRange(null);
  $('#anCalCheck').classList.add('hidden');
  updateSteps();
  if (!path) { el.textContent = ''; autoZStep(null); autoObjective(); loadRecent(); return; }
  el.textContent = 'reading…'; el.title = '';
  $('#anRecent').classList.add('hidden');
  try {
    const i = await inspect(path);
    if (cleanPath($('#anCal').value) !== path) return;   // changed meanwhile
    form.calInfo = i;
    el.textContent = infoText(i);
    el.title = i.warning || '';
    autoZStep(i);
    seedRange(i, path);
    autoObjective();
    updateSteps();
    checkCalibration(path);
    updateSiblings();
  } catch (e) { el.textContent = `⚠ ${e.message}`; updateSteps(); }
}

function movieTag(m) {
  if (m.error) return '<span class="tag err">unreadable</span>';
  if (!m.info) return '<span class="tag">reading…</span>';
  if (m.checkError) return '<span class="tag">not checked</span>';
  if (!m.check) return '<span class="tag">checking…</span>';
  const c = m.check;
  return c.verdict === 'bright' ? `<span class="tag ok" title="${esc(c.message)}">bright</span>`
    : c.verdict === 'dim' ? `<span class="tag warn" title="${esc(c.message)}">dim · dim-data settings</span>`
      : `<span class="tag err" title="${esc(c.message)}">few beads</span>`;
}
function renderMovies() {
  $('#anMovies').innerHTML = form.movies.map((m, k) => {
    const meta = m.error ? m.error : m.info ? infoText(m.info) : '';
    return `<li title="${esc(m.path)}"><span class="nm">${esc(baseName(m.path))}</span><span class="meta">${esc(meta)}</span>${movieTag(m)}<button data-k="${k}" title="Remove">✕</button></li>`;
  }).join('');
  updateSteps();
}
function moviesChanged() { renderMovies(); thrInit(); updateSiblings(); }

// quick brightness check of an added movie (a few seconds; the engine picks dim-data settings from it)
async function checkMovie(m) {
  try { m.check = await api(`/api/check_movie?path=${encodeURIComponent(m.path)}`); }
  catch (e) { m.checkError = e.message; }
  if (form.movies.includes(m)) renderMovies();
}

// "add the other movies in this folder": .tif files next to the last movie that are not yet in the list
let siblingToken = 0;
async function updateSiblings() {
  const box = $('#anSiblings'), btn = $('#anAddSiblings');
  const last = form.movies[form.movies.length - 1];
  const token = ++siblingToken;
  box.classList.add('hidden');
  if (!last) return;
  const dir = last.path.replace(/[\\/][^\\/]*$/, '');
  try {
    const r = await api(`/api/browse?path=${encodeURIComponent(dir)}`);
    if (token !== siblingToken) return;
    const have = new Set([...form.movies.map((m) => m.path.toLowerCase()), cleanPath($('#anCal').value).toLowerCase()]);
    // skip continuation parts of multi-file Micro-Manager stacks (..._Pos0_1.ome.tif)
    const others = r.files.map((f) => joinPath(r.path, f.name))
      .filter((p) => !have.has(p.toLowerCase()) && !/_Pos\d+_\d+\.ome\.tiff?$/i.test(p));
    if (!others.length || others.length > 20) return;
    btn.textContent = `+ add the other ${others.length} .tif file${others.length > 1 ? 's' : ''} in this folder`;
    btn.title = others.map(baseName).join('\n');
    btn.onclick = () => addMovies(others);
    box.classList.remove('hidden');
  } catch (e) { /* folder not listable: no suggestion */ }
}

// recently used calibration stacks, newest first; a cached fit starts right away
async function loadRecent() {
  const box = $('#anRecent');
  try {
    const { calibrations } = await api('/api/recent_calibrations');
    const current = cleanPath($('#anCal').value).toLowerCase();
    const list = (calibrations || []).filter((c) => c.exists && c.path.toLowerCase() !== current).slice(0, 4);
    if (!list.length) { box.classList.add('hidden'); return; }
    box.innerHTML = '<div class="an-recent-h">Used before:</div>' + list.map((c) => {
      const tag = c.cache === 'current' ? '<span class="tag ok">fitted · starts right away</span>' : '<span class="tag">needs fitting</span>';
      const short = c.name.replace(/_MMStack_Pos\d+/i, '').replace(/(\.ome)?\.tiff?$/i, '');
      return `<button data-path="${esc(c.path)}" title="${esc(c.path)}"><span class="nm">${esc(short)}</span>${c.objective_guess ? `<span class="tag">${esc(c.objective_guess.replace('x', '×'))}</span>` : ''}${tag}</button>`;
    }).join('');
    box.classList.remove('hidden');
  } catch (e) { box.classList.add('hidden'); }
}

// ------------------------------------------------------------------ steps, summary and time estimate
const fmtMin = (m) => (m < 1.5 ? 'about 1 minute' : `about ${Math.round(m)} minutes`);
function estimateMinutes() {
  const c = form.calInfo, check = form.calCheck;
  let cal = 0;
  if (c && !(check && check.cache === 'current')) {
    const planes = range.n && range.path ? range.b - range.a + 1 : c.frames;
    cal = planes * (c.width * c.height / 1.05e6) * 1.7 / 60;   // measured: ~1.7 s per 1024² plane (16 cores)
  }
  const mov = form.movies.reduce((s, m) => s + (m.info ? m.info.frames * (m.info.width * m.info.height / 1.05e6) * 1.3 / 60 + 1 : 0), 0);
  return cal + mov + 0.3;
}
function stepState(el, cls, text) {
  el.classList.remove('done', 'attention');
  if (cls) el.classList.add(cls);
  el.querySelector('.an-state').textContent = text;
}
function updateSteps() {
  const cal = cleanPath($('#anCal').value), obj = $('#anObj').value;
  const check = form.calCheck, info = form.calInfo;
  // step 1
  let s1 = ['', 'choose a calibration stack'];
  if (cal && !info) s1 = ['', 'reading…'];
  else if (info && !check) s1 = ['', 'checking…'];
  else if (check && check.verdict !== 'ok') s1 = ['attention', check.verdict === 'too_dim' ? 'too dim' : 'not a z-scan'];
  else if (check && !obj) s1 = ['attention', 'choose the magnification'];
  else if (check && !zStepValue()) s1 = ['attention', 'enter the z step between planes'];
  else if (check) s1 = ['done', check.cache === 'current' ? 'ready · fit cached' : 'ready'];
  stepState($('#anStep1'), ...s1);
  // step 2
  const n = form.movies.length, bad = form.movies.filter((m) => m.error).length;
  const reading = form.movies.filter((m) => !m.error && !m.info).length;
  // the brightness check is informative only (the engine checks each movie again), so it never blocks Start
  const checking = form.movies.filter((m) => m.info && !m.check && !m.checkError).length;
  let s2 = ['', 'add at least one movie'];
  if (n && bad) s2 = ['attention', `${bad} unreadable`];
  else if (n && reading) s2 = ['', 'reading…'];
  else if (n) s2 = ['done', `${n} movie${n > 1 ? 's' : ''}${checking ? ' · checking brightness…' : ''}`];
  stepState($('#anStep2'), ...s2);
  // step 3: summary
  const todo = [];
  if (s1[0] !== 'done') todo.push(`Calibration: ${s1[1]}`);
  if (s2[0] !== 'done') todo.push(`Movies: ${s2[1]}`);
  const rows = [];
  if (info) {
    const planes = range.n && range.path ? `planes ${range.a}–${range.b} of ${range.n}` : `${info.frames} planes`;
    rows.push(['Calibration', `${baseName(cal)} · ${planes}${check && check.cache === 'current' ? ' · fit cached, starts right away' : ''}`]);
  }
  if (n) {
    const frames = form.movies.reduce((s, m) => s + (m.info ? m.info.frames : 0), 0);
    const dim = form.movies.filter((m) => m.check && m.check.dim).length;
    rows.push(['Movies', `${n} · ${frames} frames${dim ? ` · ${dim} dim (dim-data settings)` : ''}`]);
  }
  if (obj) rows.push(['Scale', `${obj === 'custom' ? `${$('#anPx').value || '?'} µm/px` : `${obj.replace('x', '×')} · ${PRESET_PX[obj]} µm/px`} · z in ${$('#anTrueDepth').checked ? 'true depth (× 1.33)' : 'stage µm'}`]);
  if (info && zStepValue()) rows.push(['z step', `${zStepValue()} µm between calibration planes${form.zTouched ? ' (entered)' : ' (from the file name)'}`]);
  const name = $('#anName').value.trim();
  if (name) rows.push(['Results', `runs/${name}`]);
  if (info && n) rows.push(['Time', `${fmtMin(estimateMinutes())} (you can close this window meanwhile)`]);
  $('#anSummary').innerHTML = rows.map(([k, v]) => `<div class="row"><span class="k">${esc(k)}</span><span class="v">${esc(v)}</span></div>`).join('')
    + todo.map((t) => `<div class="todo">• ${esc(t)}</div>`).join('');
  stepState($('#anStep3'), todo.length ? '' : 'done', todo.length ? '' : 'ready to start');
  $('#anStart').disabled = !!todo.length;
}

async function addMovies(paths) {
  for (const p of paths.flatMap(splitPaths)) {
    if (!p || form.movies.some((m) => m.path === p)) continue;
    const m = { path: p, info: null, error: null };
    form.movies.push(m);
    moviesChanged();
    inspect(p).then((i) => { m.info = i; checkMovie(m); }).catch((e) => { m.error = e.message; })
      .finally(() => { renderMovies(); autoObjective(); autoName(); updateSteps(); });
  }
}

function showError(msg) { const el = $('#anErr'); el.textContent = msg || ''; el.classList.toggle('hidden', !msg); }

async function start() {
  showError('');
  const cal = cleanPath($('#anCal').value);
  const pending = $('#anMovieIn').value.trim();
  if (pending) { await addMovies([pending]); $('#anMovieIn').value = ''; }
  const obj = $('#anObj').value;
  const px = parseFloat(obj === 'custom' ? $('#anPx').value : PRESET_PX[obj]);
  const zText = $('#anZStep').value.trim(), zStep = parseFloat(zText);
  const zRange = parseFloat($('#anZRange').value);
  if (!cal) return showError('Choose the calibration z-stack.');
  if (!form.movies.length) return showError('Add at least one movie.');
  if (!obj) return showError('Choose the magnification (it sets the µm per pixel).');
  if (!(px > 0)) return showError('Pixel size must be a positive number.');
  if (!(zStep > 0)) { $('#anCalMore').open = true; return showError('Enter the z step: the distance between calibration planes in µm.'); }
  if (form.calCheck && form.calCheck.verdict !== 'ok') return showError(form.calCheck.message);
  const body = {
    calibration: cal, movies: form.movies.map((m) => m.path), pixel_size_um: px, z_step_um: zStep,
    z_range_um: zRange > 0 ? zRange : null, name: $('#anName').value.trim() || null, matlab: $('#anMatlab').checked, true_depth: $('#anTrueDepth').checked,
    frame_interval_ms: parseFloat($('#anFrameDt').value) > 0 ? parseFloat($('#anFrameDt').value) : null,
    calibration_planes: range.n && range.path === cal ? [range.a, range.b] : null,
    min_snr: thr.touched ? snrValue() : null,
  };
  $('#anStart').disabled = true;
  try {
    const r = await post('/api/analyze', body);
    attachJob({ id: r.job_id, run_id: r.run_id, name: r.name, output: r.output });
  } catch (e) {
    showError(e.message);
    if (/already running/.test(e.message)) resumeCurrent();
  } finally { $('#anStart').disabled = false; }
}

function resetForm(keepCalibration = true) {
  form.movies = []; form.nameTouched = false;
  $('#anName').value = ''; $('#anMovieIn').value = '';
  if (!keepCalibration) { form.objTouched = false; form.zTouched = false; $('#anZStep').value = ''; setCalibration(''); }
  Object.assign(thr, { path: null, info: null, img: null, data: null, touched: false });
  thr.token++;
  $('#anThrSec').open = false;
  $('#anThrBusy').classList.add('hidden');
  updateThrSummary();
  moviesChanged(); showError('');
}

// ------------------------------------------------------------------ job progress
function showView(which) {
  $('#anForm').classList.toggle('hidden', which !== 'form');
  $('#anProgress').classList.toggle('hidden', which !== 'progress');
}

function attachJob(j) {
  job = { ...j, state: 'running', acknowledged: false, toasted: false };
  lastLog = '';
  $('#anLog').textContent = ''; $('#anLogWrap').open = false;
  $('#anErrBox').classList.add('hidden');
  $('#anOut').textContent = j.output || ''; $('#anOut').title = j.output || '';
  showView('progress');
  renderStatus({ state: 'running', stage: 'starting', progress: 0, elapsed_s: 0, message: 'launching…' });
  poll();
}

async function poll() {
  clearTimeout(pollTimer);
  if (!job) return;
  try {
    const st = await api(`/api/jobs/${job.id}`);
    renderStatus(st);
    if (st.state === 'running') pollTimer = setTimeout(poll, pollDelay());
    else finished(st);
  } catch (e) {
    $('#anMsg').textContent = `status unavailable: ${e.message}`;
    pollTimer = setTimeout(poll, 2 * pollDelay());
  }
}

function renderStatus(st) {
  const p = Math.max(0, Math.min(1, +st.progress || 0));
  const pct = Math.round(p * 100);
  const state = st.state || 'running';
  $('#anStage').textContent = state === 'running' ? (st.stage || 'running') : state === 'done' ? 'finished' : state === 'cancelled' ? 'cancelled' : 'failed';
  $('#anPct').textContent = `${pct}%`;
  const fill = $('#anBarFill');
  fill.style.width = `${state === 'done' ? 100 : pct}%`;
  fill.className = state === 'done' ? 'ok' : state === 'error' || state === 'cancelled' ? 'err' : '';
  const el = +st.elapsed_s || 0;
  $('#anElapsed').textContent = `elapsed ${fmtDur(el)}`;
  $('#anEta').textContent = state === 'running' ? etaText(st.job_id, el, p) : '';
  $('#anMsg').textContent = st.message || '';
  const log = (st.log_tail || []).join('\n');
  if (log !== lastLog) {
    lastLog = log;
    const pre = $('#anLog'); pre.textContent = log; pre.scrollTop = pre.scrollHeight;
  }
  const running = state === 'running';
  $('#anCancel').classList.toggle('hidden', !running);
  $('#anOpen').classList.toggle('hidden', state !== 'done');
  $('#anAgain').classList.toggle('hidden', running);
  if (state === 'error') {
    const box = $('#anErrBox'); box.textContent = st.error || st.message || 'analysis failed'; box.classList.remove('hidden');
    $('#anLogWrap').open = true;
  }
  if (job) job.state = state;
  updateBadge(state, pct);
}

// Time left from how fast the bar moved over the last few minutes (the bar's steps are sized by their
// expected share of the time, but crowded movies run slower than expected; a recent rate adapts to that).
let etaHist = { id: null, pts: [] };
function etaText(id, el, p) {
  if (etaHist.id !== id) etaHist = { id, pts: [] };
  const pts = etaHist.pts;
  if (!pts.length || el > pts[pts.length - 1][0]) pts.push([el, p]);
  while (pts.length > 2 && el - pts[0][0] > 180) pts.shift();          // keep about the last 3 minutes
  const [t0, p0] = pts[0];
  let left = null;
  if (el - t0 >= 20 && p - p0 >= 0.005) left = (1 - p) * (el - t0) / (p - p0);
  else if (p >= 0.05 && el > 20) left = el * (1 - p) / p;               // early on: the average so far
  if (left === null) return 'estimating the time left…';
  return left < 60 ? 'less than a minute left' : `about ${Math.round(left / 60)} min left`;
}

function updateBadge(state, pct) {
  const b = $('#jobBadge');
  if (!job || job.acknowledged || state === 'cancelled') { b.classList.add('hidden'); return; }
  b.classList.remove('hidden', 'ok', 'err');
  const eta = $('#anEta').textContent;
  if (state === 'running') { b.textContent = `analysing… ${pct}%${/^about|^less/.test(eta) ? ` · ${eta.replace(/^about /, '~')}` : ''}`; b.title = `Analysis “${job.name}” running; click to show`; }
  else if (state === 'done') { b.textContent = 'analysis done'; b.classList.add('ok'); b.title = 'Click to open the new results'; }
  else { b.textContent = 'analysis failed'; b.classList.add('err'); b.title = 'Click to see the error'; }
}

async function finished(st) {
  if (!job || job.toasted) return;
  job.toasted = true;
  if (st.state === 'done') toast(`Analysis finished: ${job.name}`, 'ok', 8000);
  else if (st.state === 'error') toast(`Analysis failed: ${st.message || st.error || ''}`, 'err', 10000);
  else if (st.state === 'cancelled') toast('Analysis cancelled', 'info');
  try { await refreshRuns(); } catch (e) { /* ignore */ }
}

async function cancel() {
  if (!job) return;
  clearTimeout(pollTimer);
  $('#anCancel').disabled = true;
  try { renderStatus(await post(`/api/jobs/${job.id}/cancel`, {})); finished({ state: 'cancelled' }); }
  catch (e) { toast(`Cancel failed: ${e.message}`, 'err'); poll(); }
  finally { $('#anCancel').disabled = false; }
}

async function resumeCurrent() {
  try {
    const { job: st } = await api('/api/jobs/current');
    if (st && st.state === 'running') { attachJob({ id: st.job_id, run_id: st.run_id, name: st.name, output: st.output }); }
  } catch (e) { /* no job manager */ }
}

// ------------------------------------------------------------------ modal
function openModal() {
  showView(job && !job.acknowledged ? 'progress' : 'form');
  $('#anModal').classList.remove('hidden');
  if (!cleanPath($('#anCal').value)) loadRecent();
  updateSteps();
  if (!job || job.acknowledged) setTimeout(() => (!$('#anCal').value ? $('#anCal') : $('#anMovieIn')).focus(), 0);
}
function closeModal() {
  $('#anModal').classList.add('hidden');
  if (job && job.state !== 'running') { job.acknowledged = true; updateBadge(job.state, 100); showView('form'); }
}

export function initAnalysis(deps) {
  ({ api, post, openRun, refreshRuns } = deps);
  bindBrowser();
  updatePx();
  $('#btnNewAnalysis').onclick = openModal;
  $('#jobBadge').onclick = openModal;
  $('#anClose').onclick = closeModal;
  $('#anModal').addEventListener('mousedown', (e) => { if (e.target.id === 'anModal') closeModal(); });
  $('#anCal').addEventListener('change', (e) => setCalibration(e.target.value));
  $('#anCalBrowse').onclick = async () => {
    const r = await browseFiles({ title: 'Calibration z-stack', start: $('#anCal').value });
    if (r) setCalibration(r[0]);
  };
  $('#anMovieIn').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && e.target.value.trim()) { addMovies([e.target.value]); e.target.value = ''; }
  });
  $('#anMovieBrowse').onclick = async () => {
    const last = form.movies.length ? form.movies[form.movies.length - 1].path : $('#anCal').value;
    const r = await browseFiles({ title: 'Movie(s) to analyse', multi: true, start: last });
    if (r) addMovies(r);
  };
  $('#anMovies').addEventListener('click', (e) => {
    const b = e.target.closest('button[data-k]'); if (!b) return;
    form.movies.splice(+b.dataset.k, 1); moviesChanged(); autoName();
  });
  // calibration range
  $('#anPlaneA').addEventListener('input', (e) => setRange(+e.target.value, range.b, 'a'));
  $('#anPlaneB').addEventListener('input', (e) => setRange(range.a, +e.target.value, 'b'));
  $('#anFirst').addEventListener('change', (e) => setRange(+e.target.value, range.b, 'a'));
  $('#anLast').addEventListener('change', (e) => setRange(range.a, +e.target.value, 'b'));
  $('#anZStep').addEventListener('input', (e) => {
    form.zTouched = !!e.target.value.trim();
    e.target.classList.toggle('need', !zStepValue());
    updateSpan(); updateSteps();
  });
  bindThreshold();
  updateThrSummary();
  $('#anObj').addEventListener('change', () => { form.objTouched = true; updatePx(); });
  $('#anName').addEventListener('input', (e) => { form.nameTouched = !!e.target.value.trim(); updateSteps(); });
  $('#anRecent').addEventListener('click', (e) => { const b = e.target.closest('button[data-path]'); if (b) setCalibration(b.dataset.path); });
  // remembered between sessions: these are lab-wide choices, not per-dataset guesses
  for (const [id, key] of [['#anTrueDepth', 'dhpsf.an.trueDepth'], ['#anMatlab', 'dhpsf.an.matlab']]) {
    $(id).checked = store.get(key) === '1';
    $(id).addEventListener('change', (e) => { store.set(key, e.target.checked ? '1' : '0'); updateSteps(); });
  }
  $('#anPx').addEventListener('input', updateSteps);
  $('#anStart').onclick = start;
  $('#anCancel').onclick = cancel;
  $('#anOpen').onclick = async () => {
    if (!job) return;
    const id = job.run_id;
    job.acknowledged = true; updateBadge('done', 100);
    $('#anModal').classList.add('hidden'); showView('form'); resetForm();
    await openRun(id);
  };
  $('#anAgain').onclick = () => { if (job) { job.acknowledged = true; updateBadge(job.state, 100); } resetForm(); showView('form'); };
  // Esc closes the file browser first, then the analysis window
  window.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    if (!$('#fbModal').classList.contains('hidden')) { fbClose(null); e.stopPropagation(); e.preventDefault(); }
    else if (!$('#anModal').classList.contains('hidden')) { closeModal(); e.stopPropagation(); e.preventDefault(); }
  }, true);
  resumeCurrent();   // page reloaded while an analysis runs
}
