// Sortable, virtualized track table linked to the selection.
// Only the rows in view (+ a small buffer) exist in the DOM; nothing is rendered while the
// Tracks tab is hidden (a dirty flag is kept instead).
import { S } from './state.js';
import { on, emit, fmt, trackColor, debounce } from './util.js';

const COLS = [
  { key: 'id', label: 'track', d: 0 },
  { key: 'len', label: 'len', d: 0, title: 'number of localizations (after filters)' },
  { key: 'first', label: 'first', d: 0 },
  { key: 'last', label: 'last', d: 0 },
  { key: 'meanZ', label: 'z̄', d: 2, title: 'mean z (µm)' },
  { key: 'zRange', label: 'Δz', d: 2, title: 'z range (µm)' },
  { key: 'net', label: 'net', d: 2, title: 'net 3D displacement first→last (µm)' },
  { key: 'maxStep', label: 'step', d: 2, title: 'max step between consecutive localizations (µm)' },
  { key: 'pctRec', label: '%rec', d: 0, title: '% recovered localizations', optional: 'recovered' },
  { key: 'bright', label: 'bright', d: 0, title: 'brightness (%): median weaker-lobe amplitude relative to the tracks within 150 px', optional: 'bright' },
  { key: 'motion', label: 'motion', title: 'motion type of the whole track (aTrack-style likelihood-ratio test): directed, confined or Brownian', optional: 'motionClass',
    render: (s) => (Number.isFinite(s.motion) ? `<span class="mc mc${s.motion}">${['Brownian', 'confined', 'directed'][s.motion]}</span>` : '') },
  { key: 'moving', label: 'moving', d: 0, title: 'frames in the moving state (P > 0.5; ExaTrack-style two-state model)', optional: 'pMoving' },
  { key: 'speed', label: 'speed', d: 3, title: 'mean speed while moving (µm per frame, or µm/s if a frame interval was given)', optional: 'speedWhileMoving' },
];
const RH = 24;                 // row height (px)
let sortKey = 'id', sortDir = 1;
let head, scroller, spacer, body, lastClickedId = null;
let rows = [];                 // sorted stats objects
let dirty = true;
const visible = () => S.ui.panel && S.ui.tab === 'tracks';

export function initTable() {
  head = document.getElementById('tblHead');
  scroller = document.getElementById('tblScroll');
  spacer = document.getElementById('tblSpacer');
  body = document.getElementById('tblBody');
  head.addEventListener('click', (e) => {
    const th = e.target.closest('[data-key]'); if (!th) return;
    const k = th.dataset.key;
    if (sortKey === k) sortDir = -sortDir; else { sortKey = k; sortDir = k === 'id' ? 1 : -1; }
    dirty = true; refreshTable();
  });
  body.addEventListener('click', (e) => {
    const tr = e.target.closest('.tr'); if (!tr) return;
    const id = Number(tr.dataset.id);
    if (e.shiftKey && lastClickedId !== null) {
      const ids = rows.map((r) => r.id);
      const a = ids.indexOf(lastClickedId), b = ids.indexOf(id);
      if (a >= 0 && b >= 0) { emit('setSelection', { ids: ids.slice(Math.min(a, b), Math.max(a, b) + 1), mode: 'add' }); return; }
    }
    lastClickedId = id;
    emit('pickTrack', { id, additive: e.ctrlKey || e.metaKey, source: 'table' });
  });
  body.addEventListener('dblclick', (e) => { const tr = e.target.closest('.tr'); if (tr) emit('gotoTrack', Number(tr.dataset.id)); });
  let raf = 0;
  scroller.addEventListener('scroll', () => { if (!raf) raf = requestAnimationFrame(() => { raf = 0; paint(); }); });
  new ResizeObserver(() => { if (visible()) paint(); }).observe(scroller);
  const mark = debounce(() => { dirty = true; refreshTable(); }, 60);
  on('data', mark); on('filter', mark);
  on('selection', () => { if (visible()) { paint(); scrollToSelection(); } });
  on('panel', refreshTable);
}

function cols() { return COLS.filter((c) => !c.optional || (S.ds && S.ds.has[c.optional])); }
const grid = () => cols().map((c) => (c.key === 'id' ? '1.3fr' : '1fr')).join(' ');

export function refreshTable() {
  if (!visible()) return;
  if (dirty) {
    dirty = false;
    const cs = cols();
    head.style.gridTemplateColumns = grid();
    head.innerHTML = cs.map((c) => `<div data-key="${c.key}" title="${c.title || ''}" class="th ${c.key === sortKey ? (sortDir > 0 ? 'asc' : 'desc') : ''}">${c.label}</div>`).join('');
    rows = [];
    if (S.ds && S.F) for (let k = 0; k < S.ds.ids.length; k++) if (S.F.trackOk[k]) rows.push(S.F.stats[k]);
    rows.sort((a, b) => {
      const x = a[sortKey], y = b[sortKey];
      const fx = Number.isFinite(x), fy = Number.isFinite(y);
      if (!fx || !fy) return fx === fy ? a.id - b.id : fx ? -1 : 1;
      return (x - y) * sortDir || a.id - b.id;
    });
    spacer.style.height = `${rows.length * RH}px`;
    document.getElementById('tableCount').textContent = `${rows.length} tracks`;
  }
  paint();
}

function paint() {
  if (!visible() || !rows) return;
  const cs = cols(), g = grid();
  const top = scroller.scrollTop, h = scroller.clientHeight;
  const i0 = Math.max(0, Math.floor(top / RH) - 5), i1 = Math.min(rows.length, Math.ceil((top + h) / RH) + 5);
  const html = [];
  for (let i = i0; i < i1; i++) {
    const s = rows[i];
    html.push(`<div class="tr${S.sel.has(s.id) ? ' sel' : ''}" data-id="${s.id}" style="top:${i * RH}px;grid-template-columns:${g}">` + cs.map((c) =>
      c.key === 'id' ? `<div><span class="sw" style="background:${trackColor(s.id)}"></span>${s.id}</div>`
        : `<div>${c.render ? c.render(s) : fmt(s[c.key], c.d)}</div>`).join('') + '</div>');
  }
  body.innerHTML = html.join('');
}

function scrollToSelection() {
  if (!S.sel.size) return;
  const i = rows.findIndex((r) => S.sel.has(r.id));
  if (i < 0) return;
  const y = i * RH, top = scroller.scrollTop, h = scroller.clientHeight;
  if (y < top || y + RH > top + h) scroller.scrollTop = Math.max(0, y - h / 3);
}
