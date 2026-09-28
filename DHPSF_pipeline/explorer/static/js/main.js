// Entry point: loads data, wires UI controls, playback, selection, tracking, export.
import { S, PX_UM, setPxUm, withRun } from './state.js';
import { on, emit, $, $$, clamp, debounce, toast, downloadBlob, downloadUrl, startTask, nextPaint, fetchJsonProgress } from './util.js';
import { buildDataset, applyFilters, filteredRows, rowAt, setCoords, buildDrift, subsetJson, origRow } from './data.js';
import * as M from './movie.js';
import { initPlots, download3D, perf as plotPerf } from './plots.js';
import { initTable } from './table.js';
import { initDeform } from './deform.js';
import { initAnalysis, isModalOpen } from './analysis.js';

// ------------------------------------------------------------------ API
async function api(path, opts) {
  const r = await fetch(withRun(path), opts);
  const j = await r.json().catch(() => ({ error: `HTTP ${r.status}` }));
  if (!r.ok) throw new Error(j.error || `HTTP ${r.status}`);
  return j;
}
const post = (path, body) => api(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
on('toast', ([m, k]) => toast(m, k));

// ------------------------------------------------------------------ busy indicator
// work without a measurable fraction (a moving bar); see util.startTask for measured progress
let busyTask = null;
function busy(text) {
  if (busyTask) { busyTask.done(); busyTask = null; }
  if (text) busyTask = startTask(text);
}

// ------------------------------------------------------------------ movie loading
let coordUserChoice = null;
async function loadMovie(name) {
  S.playing = false; updatePlayBtn();
  M.clearQueue();
  const task = startTask(`loading ${name}`);
  try {
    task.set(0.02, 'movie file');
    const info = await api(`/api/movie/${encodeURIComponent(name)}/info`);
    S.movie = name; S.info = info; S.nFrames = info.frames;
    setPxUm(info.pixel_size_um);
    S.contrast = { vmin: Math.round(info.default_vmin), vmax: Math.round(info.default_vmax), gamma: 1 };
    syncContrastInputs();
    M.setImageSize(info.width, info.height);
    S.sel = new Set(); S.rois = []; S.hover = null; S.tracking = null;
    S.ds = null; S.F = null; S.json = null; S.shortDs = null;
    setFrame(clamp(S.frame, 1, S.nFrames), true);
    $('#frameSlider').max = S.nFrames;
    M.fitView();
    const [json, drift, rej] = await Promise.all([
      // the localization table dominates: 5-80 % of the bar (the server reads the CSV first)
      fetchJsonProgress(withRun(`/api/movie/${encodeURIComponent(name)}/localizations`),
        (fr, detail) => task.set(fr === null ? null : 0.05 + 0.75 * fr, fr === null ? 'the server is reading the localizations' : `localizations ${detail}`))
        .catch((e) => { toast(`No localizations: ${e.message}`, 'err', 8000); return null; }),
      info.drift ? api(`/api/movie/${encodeURIComponent(name)}/drift`).catch((e) => { toast(`Drift file unreadable: ${e.message}`, 'err'); return null; }) : null,
      info.rejected ? api(`/api/movie/${encodeURIComponent(name)}/rejected`).catch((e) => { toast(`Rejected file unreadable: ${e.message}`, 'err'); return null; }) : null,
    ]);
    task.set(0.82, 'building tracks'); await nextPaint();
    S.drift = drift ? buildDrift(drift) : null;
    S.rej = rej ? buildDataset(rej, S.nFrames) : null;
    S.json = json;
    if (json) {
      const hasCorr = json.columns.includes('xCorrected') && json.columns.includes('yCorrected');
      S.coords = { ...(coordUserChoice || { corr: hasCorr, xy: false, z: false }) };
      buildMain();
      task.set(0.93, 'filters and views'); await nextPaint();
      S.filters = { ...S.filters, fStart: 1, fEnd: S.nFrames };
      S.disp.a = 1; S.disp.b = S.nFrames;
      syncFilterInputs(); syncDispInputs();
      loadStars();
      $('#fRecWrap').classList.toggle('hidden', !S.ds.has.recovered);
      $('#dataInfo').textContent = `${json.source} · ${S.ds.n.toLocaleString()} localizations · ${S.ds.ids.length} tracks`;
    } else $('#dataInfo').textContent = 'no localization file';
    updateCoordControls();
    recompute(false);
    updateTrackingUi();
    updateMotionUi(info.motion);
    emit('data');
    emit('selection');
    emit('frame');
    M.backgroundPreload();
  } catch (e) {
    toast(`Failed to load ${name}: ${e.message}`, 'err', 10000);
    console.error(e);
  } finally { task.done(); }
}

// (re)build the main dataset from the raw table, honouring an active re-tracking
function buildMain() {
  const json = S.json;
  if (!json) return;
  if (S.tracking) {
    const { ids, short } = S.tracking;
    const keep = [], drop = [];
    for (let i = 0; i < json.n; i++) (short[i] ? drop : keep).push(i);
    S.ds = buildDataset(subsetJson(json, keep, ids), S.nFrames);
    S.shortDs = drop.length ? buildDataset(subsetJson(json, drop, ids), S.nFrames) : null;
  } else {
    S.ds = buildDataset(json, S.nFrames);
    S.shortDs = null;
  }
  S.nFrames = Math.max(S.nFrames, S.ds.nFrames);
  applyCoords();
}

function applyCoords() {
  const c = { ...S.coords, refined: !!S.coords.refined && !S.tracking };   // refinement belongs to the pipeline tracks
  for (const d of [S.ds, S.rej, S.shortDs]) if (d) setCoords(d, c, S.drift);
  if (S.ds) S.coords = { ...S.ds.coords };      // may have fallen back (missing columns / drift)
  S.stab = S.coords.xy;
  updateBadges();
}

function updateCoordControls() {
  const hasCorr = !!(S.ds && S.ds.has.corrCols);
  $('#cCorr').disabled = !hasCorr;
  $('#cXY').disabled = $('#cZ').disabled = !S.drift;
  $('#cCorr').checked = S.coords.corr; $('#cXY').checked = S.coords.xy; $('#cZ').checked = S.coords.z;
  $('#cRef').disabled = !hasRefined(); $('#cRef').checked = !!S.coords.refined;
  $('#cRefWrap').classList.toggle('disabled', !hasRefined());
  const notes = [];
  if (S.ds && !S.ds.has.refDx) notes.push('refined positions need the motion analysis');
  else if (S.tracking) notes.push('refined positions belong to the pipeline tracks (reset re-tracking to use them)');
  if (!hasCorr) notes.push('lateral correction needs xCorrected/yCorrected columns (v2)');
  if (!S.drift) notes.push('stabilization needs {movie}_drift.csv');
  $('#coordInfo').textContent = notes.join(' · ');
  const rj = $('#ovRej'), rw = $('#ovRejWrap');
  rj.disabled = !S.rej;
  if (!S.rej) { rj.checked = false; S.ov.rejected = false; }
  rw.classList.toggle('disabled', !S.rej);
  rw.title = S.rej ? `${S.rej.n} rejected (transient / likely-noise) localizations from ${S.rej.source}` : 'Needs {movie}_rejected.csv in the results folder';
}

function changeCoords(change) {
  const wasStab = S.stab;
  S.coords = { ...S.coords, ...change };
  coordUserChoice = { ...S.coords };
  applyCoords();
  updateCoordControls();
  if (wasStab !== S.stab) { M.clearQueue(); M.prefetch(); M.backgroundPreload(); }
  recompute(false);
  emit('data');
  emit('frame');
}

function recompute(fire = true) {
  S.F = S.ds ? applyFilters(S.ds, S.filters) : null;
  if (S.F) $('#filterInfo').textContent = `${S.F.nRows.toLocaleString()} loc · ${S.F.nTracks}/${S.ds.ids.length} tracks`;
  const f = S.filters;
  const active = f.minLen > 1 || f.validOnly || Number.isFinite(f.zMin) || Number.isFinite(f.zMax) || f.recovered !== 'all' || f.fStart > 1 || f.fEnd < S.nFrames || f.minBright > 0 || (f.motion && f.motion !== 'all');
  $('#filterBadge').classList.toggle('hidden', !active);
  $('#filterBadge').textContent = 'on';
  if (fire) emit('filter');
  updateDispInfo();
  updateBadges();
}

function updateBadges() {
  const b = [];
  const toggles = [
    ['corr', 'correction', 'Lateral correction: remove the calibrated sideways shift of the bead midpoint with depth (xCorrected/yCorrected). Click to toggle.', S.ds && S.ds.has.corrCols],
    ['xy', 'XY stab', 'XY stabilization: remove whole-field XY shake (median bead motion per frame); the movie frames are shifted too. Click to toggle.', !!S.drift],
    ['z', 'Z stab', 'Z stabilization: remove whole-field z drift (median bead z change, smoothed over time). Click to toggle.', !!S.drift],
    ['refined', 'refined', 'Refined positions: Kalman smoother over each whole track (motion analysis). Click to toggle.', hasRefined() && S.coords.refined],
  ];
  for (const [key, text, tip, available] of toggles) {
    if (!available) continue;
    b.push(`<span class="badge toggle${S.coords[key] ? '' : ' off'}" data-coord="${key}" title="${tip}">${text} ${S.coords[key] ? 'on' : 'off'}</span>`);
  }
  if (S.tracking) b.push(`<span class="badge warn" title="Track ids from re-tracking (${S.tracking.summary.method}); Controls › Tracking to reset">re-tracked · ${S.tracking.summary.method}</span>`);
  if (S.F && S.ds && S.F.nTracks < S.ds.ids.length) b.push(`<span class="badge" title="Filters active (Controls › Filters)">${S.F.nTracks}/${S.ds.ids.length} tracks</span>`);
  $('#badges').innerHTML = b.join('');
}

// ------------------------------------------------------------------ frames / playback
function setFrame(f, silent = false) {
  f = Math.round(f);
  if (!Number.isFinite(f)) return;
  if (f < 1) f = S.loop && S.playing ? S.nFrames : 1;
  if (f > S.nFrames) f = S.loop && S.playing ? 1 : S.nFrames;
  S.frame = f;
  $('#frameSlider').value = f;
  $('#frameLabel').textContent = `${f} / ${S.nFrames}`;
  if (S.disp.followB) { S.disp.b = f; $('#dB').value = f; }
  if (!silent) { emit('frame'); if ((S.disp.on || S.disp.zmap) && S.disp.followB) updateDispInfo(); }
}
on('setFrame', (f) => setFrame(f));

let lastTick = 0;
function tick(t) {
  if (!S.playing) return;
  const dt = 1000 / S.fps;
  if (t - lastTick >= dt - 2) {
    let next = S.frame + 1;
    if (next > S.nFrames) { if (!S.loop) { S.playing = false; updatePlayBtn(); return; } next = 1; }
    if (M.isFrameReady(next)) { lastTick = t; setFrame(next); }
    else M.frameEntry(next);
  }
  requestAnimationFrame(tick);
}
function togglePlay() {
  S.playing = !S.playing; updatePlayBtn();
  if (S.playing) { lastTick = 0; M.prefetch(); requestAnimationFrame(tick); }
  else emit('frame');   // final, unthrottled update of dependent views
}
function updatePlayBtn() { $('#btnPlay').textContent = S.playing ? '❚❚' : '▶'; }

// ------------------------------------------------------------------ selection
function setSel(ids, mode = 'replace') {
  const before = S.sel;
  S.sel = new Set(before);
  if (mode === 'replace') S.sel = new Set(ids);
  else if (mode === 'add') for (const i of ids) S.sel.add(i);
  else if (mode === 'remove') for (const i of ids) S.sel.delete(i);
  else if (mode === 'toggle') for (const i of ids) S.sel.has(i) ? S.sel.delete(i) : S.sel.add(i);
  updateSelPill();
  if (before.size === S.sel.size && [...before].every((i) => S.sel.has(i))) return; // unchanged
  emit('selection');
}
function updateSelPill() {
  const nR = S.rois.length;
  $('#selPill').classList.toggle('hidden', !S.sel.size && !nR);
  const sel = S.sel.size === 1 ? `track ${[...S.sel][0]}` : `${S.sel.size} tracks selected`;
  $('#selCount').textContent = nR ? `${sel} · ${nR} ROI${nR > 1 ? 's' : ''}` : sel;
  const allStarred = S.sel.size > 0 && [...S.sel].every((id) => S.starred.has(id));
  const star = $('#btnStar');
  star.classList.toggle('hidden', !S.sel.size);
  star.textContent = allStarred ? '★' : '☆';
  star.title = allStarred ? 'Unstar the selected track(s)' : 'Star the selected track(s): starred tracks stay in the bar below for quick selection';
  renderStars();
}
on('selection', updateSelPill);
on('rois', updateSelPill);

// ---- starred tracks: a bar of chips under the selection pill, saved per run + movie.
// Stars refer to pipeline track numbers, so they are hidden (not lost) while re-tracked.
const starKey = () => `dhpsf.starred|${S.run || ''}|${S.movie || ''}`;
function loadStars() {
  S.starred = new Set();
  try { const v = JSON.parse(localStorage.getItem(starKey()) || '[]'); if (Array.isArray(v)) S.starred = new Set(v); } catch (e) { /* storage unavailable */ }
  updateSelPill();
}
function saveStars() {
  try { localStorage.setItem(starKey(), JSON.stringify([...S.starred])); } catch (e) { /* storage unavailable */ }
}
function starredIds() {
  if (!S.ds || S.tracking) return [];
  return [...S.starred].filter((id) => S.ds.kOf.has(id)).sort((a, b) => a - b);
}
function toggleStarSel() {
  if (!S.sel.size || S.tracking) { if (S.tracking) toast('Stars use the pipeline track numbers: reset the re-tracking first', 'err'); return; }
  const ids = [...S.sel], all = ids.every((id) => S.starred.has(id));
  for (const id of ids) all ? S.starred.delete(id) : S.starred.add(id);
  saveStars(); updateSelPill();
}
function renderStars() {
  const bar = $('#starBar'), ids = starredIds();
  bar.classList.toggle('hidden', !ids.length);
  bar.innerHTML = ids.map((id) => `<button class="star-chip${S.sel.has(id) ? ' on' : ''}" data-id="${id}" title="Track ${id}: click to select, Shift/Ctrl-click to add or remove">★ ${id}</button>`).join('')
    + (ids.length > 1 ? '<button class="star-chip all" data-id="all" title="Select all starred tracks">all ★</button>' : '');
}
// the pill's ✕ resets everything: selection and ROI outlines
function clearSelAndRois() { S.rois = []; emit('rois'); setSel([]); }
on('pickTrack', ({ id, additive, source }) => {
  setSel([id], additive ? 'toggle' : 'replace');
  // Picked in the 3D view or the table: make sure the bead is visible on the movie too.
  if (source !== 'movie' && S.sel.has(id) && S.ds) {
    const k = S.ds.kOf.get(id); if (k === undefined) return;
    let r = rowAt(S.F, k, S.frame);
    if (r < 0) {   // not in this frame: use the track's localization nearest in time
      const rows = S.ds.trackRows[k];
      r = rows.reduce((best, q) => (Math.abs(S.ds.frame[q] - S.frame) < Math.abs(S.ds.frame[best] - S.frame) ? q : best), rows[0]);
    }
    M.reveal(S.ds.ux[r], S.ds.uy[r]);
  }
});
on('setSelection', ({ ids, mode }) => setSel(ids, mode));
on('roiSelect', ({ ids, mode }) => {
  setSel(ids, mode);
  const verb = mode === 'add' ? 'added' : mode === 'remove' ? 'removed' : 'selected';
  toast(`ROI: ${ids.size} tracks ${verb} (${S.roiMode === 'any' ? 'any frame' : 'current frame'}) · ${S.sel.size} selected in total`);
});
function invertSel() {
  if (!S.F) return;
  const out = [];
  S.ds.ids.forEach((id, k) => { if (S.F.trackOk[k] && !S.sel.has(id)) out.push(id); });
  setSel(out);
}
on('gotoTrack', (id) => {
  const k = S.ds.kOf.get(id); if (k === undefined) return;
  let r = rowAt(S.F, k, S.frame);
  if (r < 0) { const rows = S.F.tRows[k].length ? S.F.tRows[k] : S.ds.trackRows[k]; r = rows[0]; setFrame(S.ds.frame[r]); }
  M.centerOn(S.ds.ux[r], S.ds.uy[r], 1.5);
  setSel([id], 'add');
});

// ------------------------------------------------------------------ drawer / panel
function setDrawer(open) {
  S.ui.drawer = open;
  $('#drawer').classList.toggle('closed', !open);
  $('#drawer').setAttribute('aria-hidden', String(!open));
  if (open) { M.updateCacheBar(); updateDispInfo(); }
}
function setPanel(show) {
  S.ui.panel = show;
  $('#layout').classList.toggle('nopanel', !show);
  emit('panel');
}
const hasRefined = () => !!(S.ds && S.ds.has.refDx && !S.tracking);
function toggleCoord(key) {
  const available = key === 'corr' ? S.ds && S.ds.has.corrCols : key === 'refined' ? hasRefined() : !!S.drift;
  if (!available) return toast(key === 'corr' ? 'No lateral-corrected columns in this file' : key === 'refined' ? 'No refined positions (run the motion analysis)' : 'No drift file for this movie', 'err');
  changeCoords({ [key]: !S.coords[key] });
  toast(`coordinates: ${S.ds ? S.ds.coordLabel : ''}`);
}

// Draggable divider between the movie and the side panel; width remembered per browser.
function initSplitter() {
  const layout = $('#layout'), bar = $('#splitter');
  const apply = (px) => layout.style.setProperty('--panel-w', px == null ? '' : `${Math.round(px)}px`);
  try { const saved = parseFloat(localStorage.getItem('dhpsf.panelWidth')); if (saved > 0) apply(saved); } catch (e) { /* storage unavailable */ }
  bar.addEventListener('pointerdown', (e) => {
    e.preventDefault();
    bar.setPointerCapture(e.pointerId);
    bar.classList.add('dragging'); document.body.classList.add('resizing');
    const move = (ev) => {
      const r = layout.getBoundingClientRect();
      apply(clamp(r.right - ev.clientX, 260, r.width - 320));
    };
    const up = () => {
      bar.removeEventListener('pointermove', move); bar.removeEventListener('pointerup', up);
      bar.classList.remove('dragging'); document.body.classList.remove('resizing');
      try { localStorage.setItem('dhpsf.panelWidth', parseFloat(layout.style.getPropertyValue('--panel-w')) || ''); } catch (e) { /* ignore */ }
      emit('panel');
    };
    bar.addEventListener('pointermove', move); bar.addEventListener('pointerup', up);
  });
  bar.addEventListener('dblclick', () => {
    apply(null);
    try { localStorage.removeItem('dhpsf.panelWidth'); } catch (e) { /* ignore */ }
    emit('panel');
  });
}

function setTab(tab) {
  S.ui.tab = tab;
  if (!S.ui.panel) setPanel(true);
  $$('.tab').forEach((b) => b.classList.toggle('active', b.dataset.tab === tab));
  $('#pane3d').classList.toggle('hidden', tab !== '3d');
  $('#paneTracks').classList.toggle('hidden', tab !== 'tracks');
  $('#paneDeform').classList.toggle('hidden', tab !== 'deform');
  // let layout settle before plotting into the newly visible pane
  setTimeout(() => emit('panel'), 0);
}

// ------------------------------------------------------------------ UI bindings
function bindCheck(id, obj, key, evt) { const el = $(id); el.checked = !!obj[key]; el.addEventListener('change', () => { obj[key] = el.checked; emit(evt); }); }
function bindNum(id, obj, key, evt, integer = false) {
  const el = $(id); el.value = obj[key];
  el.addEventListener('change', () => { const v = parseFloat(el.value); if (Number.isFinite(v)) { obj[key] = integer ? Math.round(v) : v; emit(evt); } });
}
function bindSel(id, obj, key, evt) { const el = $(id); el.value = obj[key]; el.addEventListener('change', () => { obj[key] = el.value; emit(evt); }); }

function syncContrastInputs() {
  $('#vmin').value = S.contrast.vmin; $('#vmax').value = S.contrast.vmax;
  $('#gamma').value = S.contrast.gamma; $('#gammaVal').textContent = (+S.contrast.gamma).toFixed(2);
}
function syncFilterInputs() {
  const f = S.filters;
  $('#fMinLen').value = f.minLen; $('#fValid').checked = f.validOnly;
  $('#fZMin').value = Number.isFinite(f.zMin) ? f.zMin : ''; $('#fZMax').value = Number.isFinite(f.zMax) ? f.zMax : '';
  $('#fFStart').value = f.fStart; $('#fFEnd').value = f.fEnd; $('#fRec').value = f.recovered;
  $('#fBright').value = f.minBright; $('#fBrightVal').textContent = f.minBright ? `${f.minBright} %` : 'off';
  $('#fBrightWrap').classList.toggle('hidden', !(S.ds && S.ds.has.bright));
  $('#fMotion').value = f.motion || 'all';
  $('#fMotionWrap').classList.toggle('hidden', !(S.ds && S.ds.has.motionClass));
}
function syncDispInputs() { $('#dA').value = S.disp.a; $('#dB').value = S.disp.b; }
function updateDispInfo() {
  if (!S.ds || !(S.disp.on || S.disp.zmap) || !S.ui.drawer) { $('#dispInfo').textContent = ''; return; }
  const D = M.computeDisplacement();
  if (!D.length) { $('#dispInfo').textContent = 'no tracks present in both A and B'; return; }
  const mean = D.reduce((a, d) => a + d.mag, 0) / D.length;
  const mx = Math.max(...D.map((d) => d.mag));
  $('#dispInfo').textContent = `${D.length} tracks · A ${S.disp.a} → B ${S.disp.followB ? S.frame : S.disp.b} · mean |Δr| ${mean.toFixed(2)} µm · max ${mx.toFixed(2)} µm`;
}

function bindUI() {
  // the top bar wraps onto more rows in narrow windows; the layout below follows its height
  const topbar = $('#topbar');
  const setTopbarH = () => document.documentElement.style.setProperty('--topbar-h', `${topbar.offsetHeight}px`);
  new ResizeObserver(setTopbarH).observe(topbar);
  setTopbarH();

  $('#movieSel').addEventListener('change', (e) => loadMovie(e.target.value));
  $('#frameSlider').addEventListener('input', (e) => setFrame(+e.target.value));
  $('#btnPrev').onclick = () => setFrame(S.frame - 1);
  $('#btnNext').onclick = () => setFrame(S.frame + 1);
  $('#btnPlay').onclick = togglePlay;
  $('#fps').addEventListener('change', (e) => { S.fps = clamp(+e.target.value || 10, 1, 60); });
  $('#loop').addEventListener('change', (e) => { S.loop = e.target.checked; });

  // top-bar overlay chips
  bindCheck('#ovBeads', S.ov, 'beads', 'overlay'); bindCheck('#ovTails', S.ov, 'tails', 'overlay');
  bindSel('#ovColor', S.ov, 'colorBy', 'overlay');
  $('#btnDrawer').onclick = () => setDrawer(!S.ui.drawer);
  $('#btnDrawerClose').onclick = () => setDrawer(false);
  $('#btnPanel').onclick = () => setPanel(!S.ui.panel);
  initSplitter();
  $('#badges').addEventListener('click', (e) => { const t = e.target.closest('[data-coord]'); if (t) toggleCoord(t.dataset.coord); });
  $$('.tab').forEach((b) => b.addEventListener('click', () => setTab(b.dataset.tab)));

  // display
  const contrastChanged = debounce(() => { M.clearQueue(); M.redraw(); M.prefetch(); M.backgroundPreload(); }, 250);
  $('#vmin').addEventListener('change', (e) => { S.contrast.vmin = +e.target.value; contrastChanged(); });
  $('#vmax').addEventListener('change', (e) => { S.contrast.vmax = +e.target.value; contrastChanged(); });
  $('#gamma').addEventListener('input', (e) => { S.contrast.gamma = +(+e.target.value).toFixed(2); $('#gammaVal').textContent = S.contrast.gamma.toFixed(2); contrastChanged(); });
  $('#btnAuto').onclick = () => {
    if (!S.info) return;
    S.contrast = { vmin: Math.round(S.info.default_vmin), vmax: Math.round(S.info.default_vmax), gamma: 1 };
    syncContrastInputs(); contrastChanged();
  };
  bindCheck('#ovLobes', S.ov, 'lobes', 'overlay'); bindCheck('#ovLegend', S.ov, 'legend', 'overlay');
  bindCheck('#ovRecDash', S.ov, 'recoveredDashed', 'overlay');
  bindCheck('#ovOnlySel', S.ov, 'onlySel', 'overlay'); bindCheck('#ovDim', S.ov, 'dimUnsel', 'overlay');
  bindNum('#ovTailN', S.ov, 'tailN', 'overlay', true);
  bindCheck('#ovZcAuto', S.ov, 'zcAuto', 'overlay'); bindNum('#ovZcMin', S.ov, 'zcMin', 'overlay'); bindNum('#ovZcMax', S.ov, 'zcMax', 'overlay');
  $('#ovZcAuto').addEventListener('change', (e) => { $('#ovZcMin').disabled = $('#ovZcMax').disabled = e.target.checked; });
  $('#autoRes').addEventListener('change', (e) => { M.autoRes.on = e.target.checked; M.redraw(); M.prefetch(); });
  $('#btnPreload').onclick = () => M.preloadAll();
  bindCheck('#ovRej', S.ov, 'rejected', 'overlay');
  bindCheck('#ovShort', S.ov, 'short', 'overlay');

  // tools / selection
  $$('.tool').forEach((b) => b.addEventListener('click', () => setTool(b.dataset.tool)));
  bindSel('#roiMode', S, 'roiMode', 'noop');
  $('#btnClearSel').onclick = clearSelAndRois;
  $('#btnStar').onclick = toggleStarSel;
  $('#starBar').addEventListener('click', (e) => {
    const b = e.target.closest('.star-chip'); if (!b) return;
    if (b.dataset.id === 'all') setSel(starredIds(), 'replace');
    else emit('pickTrack', { id: Number(b.dataset.id), additive: e.shiftKey || e.ctrlKey || e.metaKey, source: 'star' });
  });
  $('#btnInvert').onclick = invertSel;
  $('#btnClearRoi').onclick = () => { S.rois = []; emit('rois'); };
  $('#btnFit').onclick = () => M.fitView();

  // filters
  const fchg = () => {
    const f = S.filters, num = (id) => { const v = parseFloat($(id).value); return Number.isFinite(v) ? v : null; };
    f.minLen = Math.max(1, Math.round(num('#fMinLen') || 1));
    f.validOnly = $('#fValid').checked;
    f.zMin = num('#fZMin'); f.zMax = num('#fZMax');
    f.fStart = clamp(Math.round(num('#fFStart') || 1), 1, S.nFrames);
    f.fEnd = clamp(Math.round(num('#fFEnd') || S.nFrames), f.fStart, S.nFrames);
    f.recovered = $('#fRec').value;
    f.minBright = +$('#fBright').value || 0;
    f.motion = $('#fMotion').value;
    recompute();
  };
  ['#fMinLen', '#fValid', '#fZMin', '#fZMax', '#fFStart', '#fFEnd', '#fRec', '#fBright', '#fMotion'].forEach((id) => $(id).addEventListener('change', fchg));
  $('#fBright').addEventListener('input', (e) => { $('#fBrightVal').textContent = +e.target.value ? `${e.target.value} %` : 'off'; });
  $('#btnResetFilters').onclick = () => { S.filters = { minLen: 1, zMin: null, zMax: null, validOnly: false, recovered: 'all', fStart: 1, fEnd: S.nFrames, minBright: 0, motion: 'all' }; syncFilterInputs(); recompute(); };

  // displacement
  on('disp', updateDispInfo);
  bindCheck('#dOn', S.disp, 'on', 'disp'); bindCheck('#dZmap', S.disp, 'zmap', 'disp');
  bindNum('#dA', S.disp, 'a', 'disp', true); bindNum('#dB', S.disp, 'b', 'disp', true);
  bindNum('#dScale', S.disp, 'scale', 'disp'); bindCheck('#dOnlySel', S.disp, 'onlySel', 'disp');
  bindSel('#dZmapAt', S.disp, 'zmapAt', 'disp');
  $('#dFollow').addEventListener('change', (e) => { S.disp.followB = e.target.checked; if (e.target.checked) { S.disp.b = S.frame; $('#dB').value = S.frame; } emit('disp'); });
  $('#dSetA').onclick = () => { S.disp.a = S.frame; $('#dA').value = S.frame; emit('disp'); };
  $('#dSetB').onclick = () => { S.disp.b = S.frame; $('#dB').value = S.frame; emit('disp'); };
  on('selection', () => { if (S.disp.onlySel) { emit('disp'); } });
  on('filter', () => emit('disp'));

  // coordinates
  $('#cCorr').addEventListener('change', (e) => changeCoords({ corr: e.target.checked }));
  $('#cRef').addEventListener('change', (e) => changeCoords({ refined: e.target.checked }));
  $('#cXY').addEventListener('change', (e) => changeCoords({ xy: e.target.checked }));
  $('#cZ').addEventListener('change', (e) => changeCoords({ z: e.target.checked }));

  // 3D
  bindSel('#v3Color', S.v3, 'colorBy', 'view3d'); bindSel('#v3Units', S.v3, 'units', 'view3d');
  bindCheck('#v3YDown', S.v3, 'yDown', 'view3d'); bindCheck('#v3OnlySel', S.v3, 'onlySel', 'view3d');
  bindCheck('#v3Dim', S.v3, 'dim', 'view3d');

  // tracking
  const tkSec = document.querySelector('details[data-sec=tracking]');
  tkSec.addEventListener('toggle', () => { if (tkSec.open) loadTrackingDefaults(); });
  $('#tkMethod').addEventListener('change', updateKalmanVisibility);
  $('#tkRun').onclick = runRetrack;
  $('#motionRun').onclick = runMotion;
  $('#tkReset').onclick = resetTracking;

  // export
  $('#exApply').addEventListener('change', (e) => { S.exportApplyFilters = e.target.checked; });
  $('#exSel').onclick = () => {
    if (!S.sel.size) return toast('No tracks selected', 'err');
    doExport('selected', filteredRows(S.ds, S.F, S.sel, S.exportApplyFilters));
  };
  // every row of the server's localization table (also short tracks after re-tracking)
  $('#exAll').onclick = () => { if (S.json) doExport('all', Array.from({ length: S.json.n }, (_, i) => i), false, true); };
  $('#exFilt').onclick = () => {
    const ids = new Set(S.ds.ids.filter((_, k) => S.F.trackOk[k]));
    doExport('filtered', filteredRows(S.ds, S.F, ids, S.exportApplyFilters));
  };
  $('#exStar').onclick = () => {
    const ids = starredIds();
    if (!ids.length) return toast('No starred tracks (select a track and press ☆ in the pill on the movie)', 'err');
    doExport('starred', filteredRows(S.ds, S.F, new Set(ids), S.exportApplyFilters));
  };
  $('#exRoi').onclick = exportRois;
  // this run's exports folder in File Explorer (Export section and the Deformation ⋯ menu)
  for (const b of document.querySelectorAll('#exOpenFolder, .exOpenFolder')) {
    b.onclick = async () => {
      try { const r = await post('/api/open_exports', {}); toast(`Opened ${r.folder}`, 'ok', 4000); }
      catch (e) { toast(`Could not open the export folder: ${e.message}`, 'err', 8000); }
    };
  }
  $('#ex3d').onclick = () => { if (S.ui.tab !== '3d' || !S.ui.panel) setTab('3d'); setTimeout(download3D, 400); };

  // help
  $('#btnHelp').onclick = () => $('#help').classList.toggle('hidden');
  $('#helpClose').onclick = () => $('#help').classList.add('hidden');
  $('#help').addEventListener('click', (e) => { if (e.target.id === 'help') $('#help').classList.add('hidden'); });

  window.addEventListener('keydown', onKey);
  // ☰ › Help: README pages and the video guide, in a new tab
  document.querySelectorAll('[data-open]').forEach((b) => { b.onclick = () => window.open(b.dataset.open, '_blank'); });
  // ☰ › Folders: the movies folder (file picker start) and the results folder, saved as preferences
  document.querySelectorAll('[data-choose]').forEach((b) => {
    b.onclick = async () => {
      b.disabled = true;
      $('#setInfo').textContent = 'A folder dialog is open on this computer (it may be behind this window)…';
      try {
        const r = await post('/api/settings/choose', { which: b.dataset.choose });
        showSettings(r);
        if (r.changed && b.dataset.choose === 'results') refreshRuns();
      } catch (e) { toast(`Could not change the folder: ${e.message}`, 'err', 8000); loadSettings(); }
      finally { b.disabled = false; }
    };
  });
  loadSettings();
  checkVersion();
  setInterval(checkVersion, 6 * 3600 * 1000);
  $('#updBadge').onclick = () => toast(UPDATE_HOW, 'info', 15000);
}
// a newer version on GitHub (installed copies only; the launcher does the update itself)
const UPDATE_HOW = 'To update: when no analysis is running, close the DH-Tracker-2026 window (the black one) and '
  + 'double-click Start DH-Tracker-2026.bat again; answer Y. Your results, calibrations and settings are kept.';
async function checkVersion() {
  let v;
  try { v = await api('/api/version'); } catch (e) { return; }
  $('#updBadge').classList.toggle('hidden', !v.update_available);
  const short = (s) => (s || '').slice(0, 7);
  $('#verInfo').textContent = !v.installed ? ''
    : v.update_available ? `Version ${short(v.installed)}; a newer one (${short(v.latest)}) is on GitHub. ${UPDATE_HOW}`
      : `Version ${short(v.installed)}${v.latest ? ' (up to date)' : ''}.`;
}
const shortPath = (p) => (p && p.length > 40 ? `…${p.slice(-39)}` : p || 'not chosen');
function showSettings(s) {
  for (const [id, p] of [['#setData', s.data_dir], ['#setResults', s.results_dir]]) { $(id).textContent = shortPath(p); $(id).title = p || ''; }
  $('#setInfo').textContent = s.legacy_results
    ? `Earlier results are in ${s.legacy_results}; move those folders into the results folder to see them in the list.`
    : 'Remembered between sessions and across program updates.';
}
async function loadSettings() { try { showSettings(await api('/api/settings')); } catch (e) { /* older server */ } }

function setTool(t) {
  S.tool = t; M.cancelDrawing();
  $$('.tool').forEach((b) => b.classList.toggle('active', b.dataset.tool === t));
  $('#movieCanvas').style.cursor = t === 'pan' ? 'default' : 'crosshair';
}

// ------------------------------------------------------------------ tracking
const TK = ['link_distance_px', 'gap_distance_px', 'max_frame_gap', 'angle_cost_deg', 'kalman_process_noise', 'kalman_measurement_noise', 'kalman_initial_velocity', 'min_track_length'];
let tkDefaults = null;
async function loadTrackingDefaults() {
  if (tkDefaults) return;
  $('#tkInfo').textContent = 'loading tracker defaults from pipeline.py…';
  try {
    tkDefaults = await api('/api/tracking/defaults');
    for (const k of TK) if (k in tkDefaults) $(`#tk_${k}`).value = tkDefaults[k];
    $('#tkMethod').value = tkDefaults.method || 'lap';
    updateKalmanVisibility();
    updateTrackingUi();
  } catch (e) { $('#tkInfo').textContent = `pipeline tracker unavailable: ${e.message}`; }
}
function updateKalmanVisibility() { $$('.kal').forEach((el) => el.classList.toggle('hidden', $('#tkMethod').value !== 'kalman')); }
// ------------------------------------------------------------------ motion analysis
function updateMotionUi(motion) {
  const has = !!(S.ds && S.ds.has.pMoving);
  const hasStage = !!(S.ds && S.ds.has.stage);
  $('#ovColorMotion').hidden = !has; $('#v3ColorMotion').hidden = !has;
  $('#ovColorStage').hidden = !hasStage; $('#v3ColorStage').hidden = !hasStage;
  for (const [st, sel, fallback] of [[S.ov, '#ovColor', 'z'], [S.v3, '#v3Color', 'frame']]) {
    if ((!has && st.colorBy === 'motion') || (!hasStage && st.colorBy === 'stage')) { st.colorBy = fallback; $(sel).value = fallback; }
  }
  const m = motion || {}, s = m.summary, job = m.job || {};
  S.stageNames = s && s.stages ? s.stages.names : null;
  const badge = $('#motionBadge'), info = $('#motionInfo'), run = $('#motionRun');
  badge.classList.toggle('hidden', !has && job.state !== 'running');
  badge.textContent = job.state === 'running' ? 'running…' : has ? 'on' : '';
  run.disabled = job.state === 'running';
  run.textContent = has ? 're-run motion analysis' : 'run motion analysis';
  if (job.state === 'running') info.textContent = `Analysing motion… (${$('#motionStages').value ? 'about a minute' : 'about 10 seconds'})`;
  else if (job.state === 'error') info.textContent = `Motion analysis failed: ${job.message || ''}`;
  else if (m.state === 'stale') info.textContent = 'The motion analysis is older than these localizations; run it again.';
  else if (s) {
    const c = s.classes || {}, sw = s.switching_model, u = s.time_unit || 'frame';
    const over = s.stages
      ? `Stages ${s.stages.names.map((n, i) => `${i + 1}. ${n} (${s.stages.beads_entering[i]} beads)`).join(' → ')}; colour by “stage” to see them. `
      : sw ? `Beads start moving at ${(+sw.p_sm).toFixed(3)} and stop at ${(+sw.p_ms).toFixed(3)} per frame (a movement lasts ~${Math.round(sw.mean_moving_duration_frames)} frames). ` : '';
    info.textContent = `${s.tracks} tracks: ${c.directed || 0} directed, ${c.confined || 0} confined, ${c.brownian || 0} Brownian (α = ${s.alpha}). `
      + over + `Localization errors scaled ×${(s.localization_error_scale_xyz || []).join(' / ')} (x / y / z) from the data. Units: per ${u}.`;
  } else info.textContent = 'Not yet analysed for this movie.';
}
let motionTimer = null;
async function runMotion() {
  if (!S.movie) return;
  const dt = parseFloat($('#motionDt').value);
  try {
    const r = await post(`/api/movie/${encodeURIComponent(S.movie)}/motion`, { frame_interval_ms: dt > 0 ? dt : null, stages: $('#motionStages').value || null });
    updateMotionUi(r);
    const movie = S.movie;
    clearTimeout(motionTimer);
    const task = startTask('motion analysis');   // a separate process: no fraction, a moving bar
    const poll = async () => {
      if (S.movie !== movie) { task.done(); return; }
      const st = await api(`/api/movie/${encodeURIComponent(movie)}/motion`).catch(() => null);
      if (st && st.job && st.job.state === 'running') { updateMotionUi(st); motionTimer = setTimeout(poll, 3000); return; }
      task.done();
      if (st && st.job && st.job.state === 'error') { updateMotionUi(st); toast('Motion analysis failed', 'err'); return; }
      toast('Motion analysis finished', 'ok');
      await loadMovie(movie);   // reload the localizations with the motion columns
    };
    motionTimer = setTimeout(poll, 3000);
  } catch (e) { toast(`Motion analysis: ${e.message}`, 'err'); }
}

function updateTrackingUi() {
  $('#trackBadge').classList.toggle('hidden', !S.tracking);
  $('#tkReset').disabled = !S.tracking;
  if (S.tracking) {
    const s = S.tracking.summary;
    $('#tkInfo').textContent = `${s.method}: ${s.n_tracks} tracks ≥ min length (${s.n_full_length} full-length, median ${s.median_length}) · ${s.n_short_tracks} short tracks (${s.n_short_localizations} loc) set aside · ${s.elapsed_s} s`;
  } else if (tkDefaults) $('#tkInfo').textContent = 'using the pipeline’s track numbers';
  updateBadges();
}
async function runRetrack() {
  if (!S.movie || !S.json) return;
  const body = { method: $('#tkMethod').value };
  for (const k of TK) { const v = parseFloat($(`#tk_${k}`).value); if (Number.isFinite(v)) body[k] = v; }
  $('#tkRun').disabled = true;
  busy(`re-linking with ${body.method}`);
  try {
    const t0 = performance.now();
    const r = await post(`/api/movie/${encodeURIComponent(S.movie)}/retrack`, body);
    S.tracking = { summary: r.summary, params: body, ids: r.track_number, short: r.short };
    const t1 = performance.now();
    rebuildAfterTracking();
    const s = r.summary;
    toast(`Re-tracked (${s.method}): ${s.n_tracks} tracks, ${s.n_full_length} full-length, median ${s.median_length} · server ${s.elapsed_s} s, total ${((performance.now() - t0) / 1000).toFixed(1)} s (client rebuild ${(performance.now() - t1).toFixed(0)} ms)`, 'ok', 8000);
  } catch (e) { toast(`Re-tracking failed: ${e.message}`, 'err', 10000); }
  finally { busy(null); $('#tkRun').disabled = false; }
}
async function resetTracking() {
  if (!S.tracking) return;
  try { await post(`/api/movie/${encodeURIComponent(S.movie)}/retrack`, { method: 'reset' }); } catch (e) { /* server state is advisory */ }
  S.tracking = null;
  rebuildAfterTracking();
  toast('Back to pipeline tracks', 'ok');
}
function rebuildAfterTracking() {
  S.sel = new Set();
  buildMain();
  recompute(false);
  updateTrackingUi();
  emit('data'); emit('selection'); emit('frame');
}

// ------------------------------------------------------------------ export
async function doExport(label, dsRows, useFilters = S.exportApplyFilters, serverRows = false) {
  if (!S.ds || !dsRows.length) return toast('Nothing to export', 'err');
  const rows = serverRows ? dsRows : dsRows.map((r) => origRow(S.ds, r));
  $('#exInfo').textContent = 'exporting…';
  const task = startTask(`export (${label}, ${rows.length.toLocaleString()} rows)`);
  try {
    const body = {
      movie: S.movie, label, rows, coordinates: S.coords,
      tracking: S.tracking ? 'retrack' : 'pipeline',
      filters: useFilters ? S.filters : null, selection: Array.from(S.sel), rois: S.rois,
    };
    const r = await post('/api/export', body);
    $('#exInfo').textContent = `${r.rows} rows · ${r.tracks} tracks → ${r.csv} (+ .mat)`;
    toast(`Exported ${r.rows} rows (${r.tracks} tracks)\n${r.csv}\n${r.mat}`, 'ok', 7000);
    downloadUrl(r.csv_url, r.stem + '.csv');
  } catch (e) { $('#exInfo').textContent = ''; toast(`Export failed: ${e.message}`, 'err', 8000); } finally { task.done(); }
}

async function exportRois() {
  if (!S.rois.length) return toast('No ROIs drawn', 'err');
  const payload = {
    movie: S.movie, pixel_size_um: PX_UM,
    coordinates: S.stab ? 'one-based image pixels of the stabilized (drift-shifted) frames' : 'one-based image pixels (same frame as CSV xMean/yMean)',
    stabilized: S.stab, frame: S.frame, roi_mode: S.roiMode, rois: S.rois, selected_tracks: Array.from(S.sel),
    tracking: S.tracking ? S.tracking.summary : 'pipeline',
  };
  downloadBlob(new Blob([JSON.stringify(payload, null, 1)], { type: 'application/json' }), `${S.movie}_rois.json`);
  try { const r = await post('/api/export_roi', payload); toast(`ROIs saved: ${r.json}`, 'ok'); }
  catch (e) { toast(`ROI save failed: ${e.message}`, 'err'); }
}

// ------------------------------------------------------------------ keyboard
// No global shortcuts (kept deliberately simple): only Enter / Backspace while drawing a
// polygon ROI, and Esc to close the help.
function onKey(e) {
  if (e.key === 'F1') { e.preventDefault(); window.open('/readme', '_blank'); return; }   // the README, anywhere
  if (isModalOpen() && e.key !== 'Escape') return;
  const t = e.target;
  if (t && ((t.tagName === 'INPUT' && t.type !== 'range' && t.type !== 'checkbox') || t.tagName === 'SELECT' || t.tagName === 'TEXTAREA')) return;
  if (M.polyKey(e.key, e)) { e.preventDefault(); return; }
  if (e.key === 'Escape' && !$('#help').classList.contains('hidden')) $('#help').classList.add('hidden');
}

// ------------------------------------------------------------------ runs (result sets)
const escHtml = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const STATE_NOTE = { running: 'running', error: 'failed', cancelled: 'cancelled' };
function runTip(r) {
  if (!r) return 'Result set (analysis run)';
  return `${r.path}\n${r.date}${r.pixel_size_um ? ` · ${r.pixel_size_um} µm/px` : ''}\nmovies: ${r.movies.join(', ') || '—'}`;
}
async function refreshRuns() {
  const { runs, active } = await api('/api/runs');
  S.runs = runs;
  if (!S.run) S.run = active;
  const sel = $('#runSel');
  // rebuilding the list while it is open would close it
  if (document.activeElement !== sel && !runs.some((r) => r.complete)) {
    sel.innerHTML = '<option value="" disabled selected>no analyses yet</option>';   // new installation
  } else if (document.activeElement !== sel) {
    sel.innerHTML = runs.map((r) => {
      const note = !r.complete ? (STATE_NOTE[r.state] || 'no results') : (r.state && r.state !== 'done' ? STATE_NOTE[r.state] : '');
      return `<option value="${escHtml(r.id)}" ${r.complete ? '' : 'disabled'} title="${escHtml(runTip(r))}">${escHtml(r.label)}${note ? ` (${note})` : ''}</option>`;
    }).join('');
    sel.value = S.run;
  }
  sel.title = runTip(runs.find((r) => r.id === S.run));
  return active;
}

// Every request names this page's run (state.withRun), so other tabs or windows showing other runs
// do not affect it; this only picks up runs that finished meanwhile.
let syncing = false;
async function syncRun() {
  if (syncing) return;
  syncing = true;
  try { await refreshRuns(); } catch (e) { console.warn('runs list unavailable', e); } finally { syncing = false; }
}

async function selectRun(id) {
  if (!id || id === S.run) { $('#runSel').value = S.run; return; }
  busy('switching result set');
  try {
    await post('/api/runs/select', { id });   // also the default for pages opened later
  } catch (e) {
    toast(`Cannot open ${id}: ${e.message}`, 'err', 8000);
    $('#runSel').value = S.run;
    busy(null);
    return;
  }
  busy(null);
  S.run = id;
  await loadRun();
}

// (re)load the movie list of the active run, then its first movie
async function loadRun() {
  S.playing = false; updatePlayBtn();
  const { movies, results, run } = await api('/api/movies');
  S.movies = movies; S.run = run;
  S.movie = null;   // no frame requests for the previous run's movie name
  S.tracking = null; S.sel = new Set(); S.rois = [];
  S.filters = { minLen: 1, zMin: null, zMax: null, validOnly: false, recovered: 'all', fStart: 1, fEnd: 1, minBright: 0, motion: 'all' };
  syncFilterInputs();
  M.clearFrames();
  const sel = $('#movieSel');
  sel.innerHTML = movies.map((m) => `<option value="${escHtml(m.name)}" ${m.available ? '' : 'disabled'}>${escHtml(m.name)}${m.localizations ? '' : ' (no CSV)'}</option>`).join('');
  document.title = `DH-Tracker-2026 — ${results.split(/[\\/]/).slice(-1)[0]}`;
  $('#dataInfo').title = results;
  refreshRuns().catch((e) => console.warn('runs list unavailable', e));
  $('#emptyState').classList.add('hidden');
  const first = movies.find((m) => m.available && m.localizations) || movies.find((m) => m.available);
  if (first) { sel.value = first.name; await loadMovie(first.name); }
  else {
    S.movie = null; S.info = null; S.json = null; S.ds = null; S.F = null; S.drift = null; S.rej = null; S.shortDs = null;
    updateCoordControls(); updateTrackingUi(); updateBadges();
    emit('data'); emit('selection'); emit('frame');
    if (!movies.length) {          // e.g. a new installation: nothing analysed yet
      $('#dataInfo').textContent = 'no analyses yet';
      $('#emptyState').classList.remove('hidden');
    } else {
      $('#dataInfo').textContent = 'movie files not found';
      toast(`The movie files of this result set were not found (moved or renamed?):\n${movies.map((m) => m.path).join('\n')}`, 'err', 15000);
    }
  }
}

// ------------------------------------------------------------------ boot
async function boot() {
  console.info(`explorer boot ${new Date().toISOString()}`);
  M.initMovie();
  initPlots();
  initDeform();
  initTable();
  bindUI();
  $('#runSel').addEventListener('change', (e) => selectRun(e.target.value));
  $('#emptyNew').onclick = () => $('#btnNewAnalysis').click();
  // newly finished runs appear when the list is opened; another tab's switch is undone on return
  $('#runSel').addEventListener('pointerdown', () => { if (document.activeElement !== $('#runSel')) syncRun(); });
  window.addEventListener('focus', syncRun);
  document.addEventListener('visibilitychange', () => { if (!document.hidden) syncRun(); });
  setInterval(() => { if (!document.hidden) refreshRuns().catch(() => {}); }, 20000);
  // refresh buttons: make sure the server serves this tab's run, then (if asked) reload the movie's data
  on('syncRun', () => syncRun());
  on('reloadData', async () => {
    await syncRun();
    if (S.movie) await loadMovie(S.movie); else await loadRun();
  });
  initAnalysis({ api, post, openRun: selectRun, refreshRuns });
  try { await loadRun(); } catch (e) { toast(`Server error: ${e.message}`, 'err', 10000); }
}
boot();
window.__S = S; // debugging aid
window.__perf = { movie: M.perf, plots: plotPerf };
