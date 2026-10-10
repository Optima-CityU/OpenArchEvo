// Shared helpers: theme, data loading, formatting, colour scales, tooltip, lightbox.
export const PAPER = {
  title: 'Large Language Model-Guided Evolutionary Discovery of Native Neural Architectures for Spiking Sequence Modeling',
  arxiv: 'https://arxiv.org/abs/2609.40258',
  pdf: 'https://arxiv.org/pdf/2609.40258',
  code: 'https://github.com/Optima-CityU/OpenArchEvo',
};

export const reduceMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;

// ---------------------------------------------------------------- theme
const THEME_KEY = 'oae-theme';
function store(k, v) { try { if (v == null) localStorage.removeItem(k); else localStorage.setItem(k, v); } catch (e) { /* ignore */ } }
function load(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }
export function siteTheme() {
  const explicit = document.documentElement.getAttribute('data-site-theme');
  if (explicit) return explicit;
  return 'light';
}
export function initTheme() {
  const saved = load(THEME_KEY);
  if (saved === 'light' || saved === 'dark') document.documentElement.setAttribute('data-site-theme', saved);
  const btn = document.querySelector('[data-theme-toggle]');
  const paint = () => {
    if (!btn) return;
    const dark = siteTheme() === 'dark';
    btn.setAttribute('aria-label', dark ? 'Switch to light theme' : 'Switch to dark theme');
    btn.innerHTML = dark ? ICON.sun : ICON.moon;
  };
  paint();
  btn?.addEventListener('click', () => {
    const next = siteTheme() === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-site-theme', next);
    store(THEME_KEY, next);
    paint();
    window.dispatchEvent(new CustomEvent('oae-theme', { detail: next }));
  });
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener?.('change', () => {
    if (!load(THEME_KEY)) { paint(); window.dispatchEvent(new CustomEvent('oae-theme', { detail: siteTheme() })); }
  });
  // compact section strip (phones): bring the current page's chip into view
  const strip = document.querySelector('.mnav'), onChip = strip?.querySelector('a.on, a[aria-current="page"]');
  if (strip && onChip && strip.scrollWidth > strip.clientWidth) strip.scrollLeft = onChip.offsetLeft - strip.clientWidth / 2 + onChip.offsetWidth / 2;
  const bar = document.querySelector('.topbar');
  if (bar) {
    const onScroll = () => bar.classList.toggle('scrolled', window.scrollY > 8);
    window.addEventListener('scroll', onScroll, { passive: true }); onScroll();
  }
}

export const ICON = {
  moon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5Z"/></svg>',
  sun: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>',
  play: '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M8 5.5v13l11-6.5z"/></svg>',
  pause: '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M7 5h3.5v14H7zM13.5 5H17v14h-3.5z"/></svg>',
  copy: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V6a2 2 0 0 1 2-2h9"/></svg>',
  close: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M6 6l12 12M18 6 6 18"/></svg>',
  arrow: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M5 12h14M13 6l6 6-6 6"/></svg>',
  plus: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M12 5v14M5 12h14"/></svg>',
  minus: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M5 12h14"/></svg>',
  fit: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" aria-hidden="true"><path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/></svg>',
  github: '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 2a10 10 0 0 0-3.2 19.5c.5.1.7-.2.7-.5v-1.7c-2.8.6-3.4-1.3-3.4-1.3-.5-1.2-1.1-1.5-1.1-1.5-.9-.6.1-.6.1-.6 1 .1 1.5 1 1.5 1 .9 1.5 2.4 1.1 2.9.8.1-.7.4-1.1.6-1.3-2.2-.3-4.6-1.1-4.6-5 0-1.1.4-2 1-2.7-.1-.3-.4-1.3.1-2.7 0 0 .8-.3 2.7 1a9.4 9.4 0 0 1 5 0c1.9-1.3 2.7-1 2.7-1 .5 1.4.2 2.4.1 2.7.6.7 1 1.6 1 2.7 0 3.9-2.3 4.7-4.6 5 .4.3.7.9.7 1.9V21c0 .3.2.6.7.5A10 10 0 0 0 12 2Z"/></svg>',
  paper: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round" aria-hidden="true"><path d="M6 3h8l4 4v14H6z"/><path d="M14 3v4h4M9 12h6M9 16h6"/></svg>',
  quote: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" aria-hidden="true"><path d="M8 7 4 12l4 5M16 7l4 5-4 5"/></svg>',
};

export const LOGO = '<svg viewBox="0 0 32 32" aria-hidden="true"><circle cx="16" cy="16" r="15" fill="none" stroke="currentColor" stroke-width="1.6" opacity=".25"/><path d="M5 22c4 0 5-12 9-12s4 8 8 8 4-6 5-6" fill="none" stroke="#b54834" stroke-width="2.2" stroke-linecap="round"/><circle cx="14" cy="10" r="2.4" fill="currentColor"/><circle cx="22" cy="18" r="2.4" fill="#b54834"/></svg>';

// ---------------------------------------------------------------- data
const cache = new Map();
export function getJSON(url) {
  if (!cache.has(url)) cache.set(url, fetch(url).then(r => { if (!r.ok) throw new Error(`${r.status} ${url}`); return r.json(); }));
  return cache.get(url);
}
export function getText(url) {
  const k = 'T:' + url;
  if (!cache.has(k)) cache.set(k, fetch(url).then(r => { if (!r.ok) throw new Error(`${r.status} ${url}`); return r.text(); }));
  return cache.get(k);
}
export function archURL(key) { return `arch.html?arch=${encodeURIComponent(key)}`; }

// ---------------------------------------------------------------- formatting
export const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
export const fmt = (v, d = 2) => (v == null || Number.isNaN(v)) ? 'n/a' : Number(v).toFixed(d);
export const iterLabel = i => i === 0 ? 'Experts' : `Iteration ${i}`;
export const iterShort = i => i === 0 ? 'E' : String(i);
export function displayName(a) { return a.name || `Candidate ${a.key}`; }

// TeX-like markup from the diagram specs (e^{θ+Δ}, W_v, h_{t-1}) → <sup>/<sub> (HTML) or plain text (attributes).
// Single-character scripts only when the next character is not a letter/digit, so code names like res_gate stay intact.
const TEX_RE = /([_^])(?:\{([^{}]*)\}|([A-Za-z0-9α-ωΑ-Ω])(?![A-Za-z0-9]))/g;
const texHTML = s => s.replace(TEX_RE, (_, k, a, b) => { const t = k === '_' ? 'sub' : 'sup'; return `<${t}>${texHTML(a ?? b)}</${t}>`; });
export function tex(s) { return texHTML(esc(s)); }
export function texPlain(s) {
  return String(s ?? '').replace(TEX_RE, (m, k, a, b) => a == null ? m : k === '^' ? `^(${a})` : `_${a}`);
}

// Fire rate (WT2): value + whether it is from the final epoch. Runs whose log stopped reporting a rate early
// (loss went NaN, or the rate reached 0.000) and silent runs are flagged so that sorting puts them last.
export function frInfo(a, task = 'wt2') {
  const v = a.fr?.[task];
  const ep = a.frEp?.[task];
  const note = ep ? `epoch ${ep.at} of ${ep.of}${ep.why === 'nan' ? ', then loss NaN' : ep.why === 'silent' ? ', then silent' : ''}` : '';
  return { v, ep, note, flagged: v == null || v === 0 || !!ep };
}
export function frSortValue(a) { const f = frInfo(a); return f.v == null ? 2e6 : f.flagged ? 1e6 + f.v : f.v; }

// ---------------------------------------------------------------- colour
// Sequential ramp over search iterations (one hue family, light → dark in light mode; dark → light in dark mode).
function hex(h) { const n = parseInt(h.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
function lerp(a, b, t) { return a.map((x, i) => Math.round(x + (b[i] - x) * t)); }
function rampAt(stops, t) {
  const seg = Math.min(stops.length - 2, Math.floor(t * (stops.length - 1)));
  const lt = t * (stops.length - 1) - seg;
  const c = lerp(hex(stops[seg]), hex(stops[seg + 1]), lt);
  return `rgb(${c.join(',')})`;
}
const RAMP_LIGHT = ['#f3c6aa', '#e3916f', '#c4523b', '#8e3448', '#4e2e5e'];
const RAMP_DARK = ['#7c3f37', '#b8563f', '#f0937a', '#f6c3ad', '#fbe9df'];
export function iterColor(i, theme = siteTheme()) {
  if (i === 0) return theme === 'dark' ? '#6d6178' : '#b3a8ba';
  return rampAt(theme === 'dark' ? RAMP_DARK : RAMP_LIGHT, (i - 1) / 9);
}
export function rampCSS(theme = siteTheme()) {
  const st = theme === 'dark' ? RAMP_DARK : RAMP_LIGHT;
  return `linear-gradient(90deg, ${st.join(',')})`;
}

// ---------------------------------------------------------------- tooltip
let tipEl;
export function tip(html, ev) {
  if (!tipEl) { tipEl = document.createElement('div'); tipEl.className = 'tooltip'; tipEl.setAttribute('role', 'status'); document.body.appendChild(tipEl); }
  if (html == null) { tipEl.classList.remove('on'); return; }
  tipEl.innerHTML = html;
  tipEl.classList.add('on');
  let x, y;
  if (ev && 'clientX' in ev && ev.clientX) { x = ev.clientX; y = ev.clientY; }
  else if (ev && ev.target && ev.target.getBoundingClientRect) { const r = ev.target.getBoundingClientRect(); x = r.left + r.width / 2; y = r.top; }
  else { x = window.innerWidth / 2; y = 100; }
  const w = tipEl.offsetWidth, h = tipEl.offsetHeight;
  let left = x + 14, top = y - h - 12;
  if (left + w > window.innerWidth - 8) left = x - w - 14;
  if (left < 8) left = 8;
  if (top < 8) top = y + 16;
  tipEl.style.left = left + 'px'; tipEl.style.top = top + 'px';
}

// ---------------------------------------------------------------- lightbox
export function lightbox({ title, body, foot, paper = false, full = false, onClose }) {
  const dlg = document.createElement('dialog');
  dlg.className = 'lb' + (full ? ' full' : '');
  dlg.innerHTML = `<div class="lb-inner"><div class="lb-head"><h3></h3><span class="lb-tools"></span><button class="iconbtn" data-close aria-label="Close">${ICON.close}</button></div><div class="lb-body${paper ? ' paper' : ''}"></div>${foot ? '<div class="lb-foot"></div>' : ''}</div>`;
  dlg.querySelector('h3').textContent = title || '';
  const b = dlg.querySelector('.lb-body');
  if (typeof body === 'string') b.innerHTML = body; else if (body) b.appendChild(body);
  if (foot) dlg.querySelector('.lb-foot').innerHTML = foot;
  document.body.appendChild(dlg);
  const close = () => { dlg.close(); };
  dlg.addEventListener('close', () => { dlg.remove(); onClose?.(); });
  dlg.addEventListener('click', e => { if (e.target === dlg || e.target.closest('[data-close]')) close(); });
  dlg.showModal();
  return { dlg, body: b, tools: dlg.querySelector('.lb-tools'), close };
}

// ---------------------------------------------------------------- full-screen pan / zoom viewer
// Used for block diagrams, core state-update details and the paper figures, on every device.
// The content is re-sized (not CSS-scaled) on zoom so SVG text stays crisp at any magnification.
// Mouse: wheel zooms at the cursor, drag pans, double-click zooms in. Touch: pinch, drag, double-tap.
// Keyboard: + / − zoom, 0 fits, arrow keys pan, Esc closes.
export function panZoomViewer({ title, content, width, height, bg, foot, onClose }) {
  const vp = document.createElement('div');
  vp.className = 'pz';
  vp.tabIndex = 0;
  vp.setAttribute('aria-label', `${title}. Use plus and minus to zoom, arrow keys to pan, 0 to fit.`);
  const stage = document.createElement('div');
  stage.className = 'pz-stage';
  stage.appendChild(content);
  vp.appendChild(stage);
  const coarse = window.matchMedia('(pointer: coarse)').matches;
  const hint = document.createElement('div');
  hint.className = 'pz-hint';
  const portrait = window.matchMedia('(orientation: portrait)').matches && window.innerWidth < 700;
  hint.textContent = coarse ? `Drag to pan · pinch to zoom${portrait ? ' · rotate for a wider view' : ' · double-tap to zoom in'}` : 'Scroll to zoom · drag to pan · double-click to zoom in';
  vp.appendChild(hint);
  const lb = lightbox({ title, body: vp, foot, full: true, onClose });
  if (bg) lb.body.style.background = bg;
  lb.tools.innerHTML = `<span class="pz-pct" aria-live="polite"></span><button class="iconbtn" data-z="-" aria-label="Zoom out">${ICON.minus}</button><button class="iconbtn" data-z="0" aria-label="Fit to screen">${ICON.fit}</button><button class="iconbtn" data-z="+" aria-label="Zoom in">${ICON.plus}</button>`;
  const pct = lb.tools.querySelector('.pz-pct');
  const st = { s: 1, fit: 1, tx: 0, ty: 0, min: 0.1, max: 6 };
  let raf = 0;
  const paint = () => {
    raf = 0;
    content.style.width = `${width * st.s}px`;
    content.style.height = `${height * st.s}px`;
    stage.style.transform = `translate(${st.tx}px, ${st.ty}px)`;
    pct.textContent = `${Math.round(st.s * 100)}%`;
  };
  const schedule = () => { if (!raf) raf = requestAnimationFrame(paint); };
  const clampPan = () => {
    const W = vp.clientWidth, H = vp.clientHeight, w = width * st.s, h = height * st.s, pad = 40;
    st.tx = w <= W ? (W - w) / 2 : Math.min(pad, Math.max(W - w - pad, st.tx));
    st.ty = h <= H ? (H - h) / 2 : Math.min(pad, Math.max(H - h - pad, st.ty));
  };
  const fit = () => {
    const W = vp.clientWidth, H = vp.clientHeight;
    // fit the width (minus a 24 px margin each side) unless that would overflow the height; clampPan centres it
    const f = Math.min(Math.max(0.05, (W - 48) / width), Math.max(0.05, (H - 48) / height));
    st.min = Math.min(0.5, f * 0.5);
    st.s = st.fit = f;
    clampPan(); schedule();
  };
  // First view: fit to the screen, except when that would make a wide diagram unreadably small (portrait phones):
  // then start at a readable scale on the left edge (the input side) and let the reader pan along the block.
  const initView = () => {
    fit();
    const W = vp.clientWidth, H = vp.clientHeight;
    if (W < 700 && st.fit < 0.42) {
      st.s = Math.max(st.fit, Math.min(0.72, (H / height) * 0.96));
      st.tx = 12; st.ty = (H - height * st.s) / 2;
      clampPan(); schedule();
    }
  };
  const zoomAt = (f, cx, cy) => {
    const r = vp.getBoundingClientRect();
    const px = cx == null ? r.width / 2 : cx - r.left, py = cy == null ? r.height / 2 : cy - r.top;
    const s2 = Math.max(st.min, Math.min(st.max, st.s * f));
    const k = s2 / st.s;
    st.tx = px - (px - st.tx) * k; st.ty = py - (py - st.ty) * k; st.s = s2;
    clampPan(); schedule(); dismissHint();
  };
  let hintGone = false;
  let touched = false;
  const dismissHint = () => { touched = true; if (!hintGone) { hintGone = true; hint.classList.add('gone'); } };
  setTimeout(() => { hintGone = true; hint.classList.add('gone'); }, 3000);
  lb.tools.querySelectorAll('[data-z]').forEach(b => b.addEventListener('click', () => {
    if (b.dataset.z === '0') { fit(); touched = false; } else zoomAt(b.dataset.z === '+' ? 1.4 : 1 / 1.4);
  }));
  vp.addEventListener('wheel', e => {
    e.preventDefault();
    const d = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaY;
    zoomAt(Math.exp(-d * (e.ctrlKey ? 0.01 : 0.0022)), e.clientX, e.clientY);
  }, { passive: false });
  // pointers: one = pan, two = pinch
  const ptrs = new Map();
  let moved = 0, last = null, pinch = null, lastTap = 0;
  vp.addEventListener('pointerdown', e => {
    if (e.button && e.button !== 0) return;
    ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (ptrs.size === 1) { moved = 0; last = { x: e.clientX, y: e.clientY }; }
    if (ptrs.size === 2) { const [a, b] = [...ptrs.values()]; pinch = { d: Math.hypot(a.x - b.x, a.y - b.y), s: st.s }; }
    vp.classList.add('grabbing');
  });
  vp.addEventListener('pointermove', e => {
    if (!ptrs.has(e.pointerId)) return;
    const p = ptrs.get(e.pointerId); p.x = e.clientX; p.y = e.clientY;
    if (ptrs.size === 2 && pinch) {
      const [a, b] = [...ptrs.values()];
      const d = Math.hypot(a.x - b.x, a.y - b.y);
      zoomAt((pinch.s * d / pinch.d) / st.s, (a.x + b.x) / 2, (a.y + b.y) / 2);
      moved = 99; return;
    }
    if (ptrs.size === 1 && last) {
      const dx = e.clientX - last.x, dy = e.clientY - last.y;
      moved += Math.abs(dx) + Math.abs(dy);
      if (moved > 5) {
        if (!vp.hasPointerCapture(e.pointerId)) { try { vp.setPointerCapture(e.pointerId); } catch (_) { /* ignore */ } }
        st.tx += dx; st.ty += dy; clampPan(); schedule(); dismissHint(); tip(null);
      }
      last = { x: e.clientX, y: e.clientY };
    }
  });
  const up = e => {
    ptrs.delete(e.pointerId);
    if (ptrs.size < 2) pinch = null;
    if (ptrs.size === 1) { const [a] = [...ptrs.values()]; last = { x: a.x, y: a.y }; }
    if (!ptrs.size) vp.classList.remove('grabbing');
    // double-tap (touch) zooms in at the tap point
    if (e.pointerType === 'touch' && e.type === 'pointerup' && moved < 6) {
      const now = Date.now();
      if (now - lastTap < 320) { if (st.s > st.fit * 3) fit(); else zoomAt(2, e.clientX, e.clientY); lastTap = 0; }
      else lastTap = now;
    }
  };
  vp.addEventListener('pointerup', up); vp.addEventListener('pointercancel', up);
  // a drag must not count as a click on a diagram node
  vp.addEventListener('click', e => { if (moved > 5) { e.stopPropagation(); e.preventDefault(); } }, true);
  vp.addEventListener('dblclick', e => { if (e.target.closest('.node[data-code],.expand')) return; zoomAt(2, e.clientX, e.clientY); });
  // Keys work anywhere in the dialog (focus starts on its first button), not only when the canvas itself is focused.
  lb.dlg.addEventListener('keydown', e => {
    const k = e.key;
    if (k.startsWith('Arrow') && e.target.closest?.('.node')) return;
    if (k === '+' || k === '=') zoomAt(1.4);
    else if (k === '-' || k === '_') zoomAt(1 / 1.4);
    else if (k === '0') fit();
    else if (k.startsWith('Arrow')) { const s = 60; if (k === 'ArrowLeft') st.tx += s; if (k === 'ArrowRight') st.tx -= s; if (k === 'ArrowUp') st.ty += s; if (k === 'ArrowDown') st.ty -= s; clampPan(); schedule(); }
    else return;
    e.preventDefault();
  });
  const ro = new ResizeObserver(() => { if (touched) { clampPan(); schedule(); } else initView(); });
  ro.observe(vp);
  lb.dlg.addEventListener('close', () => ro.disconnect());
  requestAnimationFrame(initView);
  return { ...lb, fit, zoomAt };
}

export function bindZoomableImages(root = document) {
  root.querySelectorAll('img.zoomable').forEach(img => {
    img.tabIndex = 0;
    img.setAttribute('role', 'button');
    img.setAttribute('aria-label', `Enlarge: ${img.alt}`);
    const open = () => {
      const im = document.createElement('img');
      im.src = img.currentSrc || img.src; im.alt = img.alt; im.draggable = false;
      const w = img.naturalWidth || +img.getAttribute('width') || 2400, h = img.naturalHeight || +img.getAttribute('height') || 800;
      panZoomViewer({ title: img.dataset.title || img.alt, content: im, width: w, height: h, bg: '#fff' });
    };
    img.addEventListener('click', open);
    img.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); } });
  });
}

export function copyText(text, btn) {
  const done = () => { if (!btn) return; const o = btn.innerHTML; btn.innerHTML = 'Copied'; setTimeout(() => (btn.innerHTML = o), 1400); };
  if (navigator.clipboard?.writeText) navigator.clipboard.writeText(text).then(done, () => fallback());
  else fallback();
  function fallback() {
    const ta = document.createElement('textarea'); ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0';
    document.body.appendChild(ta); ta.select();
    try { document.execCommand('copy'); done(); } catch (e) { /* ignore */ }
    ta.remove();
  }
}

// Safe minimal markdown: escape everything first, then add a small set of inline/block constructs.
// Lists survive blank lines between items; indented items nest one level inside the previous item.
export function md(src) {
  const lines = String(src || '').replace(/\r\n?/g, '\n').replace(/\t/g, '    ').split('\n');
  const out = [];
  let list = null, para = [], blank = false;
  const inline = s => esc(s)
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[^*\w])\*([^*\s][^*]*)\*/g, '$1<em>$2</em>');
  const flushP = () => { if (para.length) { out.push(`<p>${inline(para.join(' '))}</p>`); para = []; } };
  const renderList = L => `<${L.t}${L.t === 'ol' && L.start > 1 ? ` start="${L.start}"` : ''}>${L.items.map(it => `<li>${inline(it.text)}${it.sub ? renderList(it.sub) : ''}</li>`).join('')}</${L.t}>`;
  const flushL = () => { if (list) { out.push(renderList(list)); list = null; } };
  for (const raw of lines) {
    const line = raw.trimEnd();
    let m;
    if (!line.trim()) { flushP(); blank = true; continue; }
    const indent = raw.match(/^ */)[0].length;
    const ul = line.match(/^\s*[-*+]\s+(.*)$/), ol = line.match(/^\s*(\d+)[.)]\s+(.*)$/);
    if ((m = line.match(/^\s{0,3}#{1,6}\s+(.*)$/))) { flushP(); flushL(); out.push(`<h4>${inline(m[1])}</h4>`); blank = false; continue; }
    if (ul || ol) {
      flushP();
      const t = ul ? 'ul' : 'ol', text = ul ? ul[1] : ol[2];
      if (list && indent >= 2 && list.items.length && indent > list.indent) {
        const last = list.items[list.items.length - 1];
        if (!last.sub || last.sub.t !== t) last.sub = { t, items: [], start: ol ? +ol[1] : 1 };
        last.sub.items.push({ text });
      } else {
        if (!list || list.t !== t) { flushL(); list = { t, items: [], indent, start: ol ? +ol[1] : 1 }; }
        list.items.push({ text });
      }
      blank = false; continue;
    }
    if (list && indent >= 2) {
      const last = list.items[list.items.length - 1];
      const tgt = last.sub ? last.sub.items[last.sub.items.length - 1] : last;
      tgt.text += ' ' + line.trim(); blank = false; continue;
    }
    if (list && blank) flushL();
    if (list && !blank) { const last = list.items[list.items.length - 1]; last.text += ' ' + line.trim(); continue; }
    para.push(line.trim()); blank = false;
  }
  flushP(); flushL();
  return out.join('');
}
