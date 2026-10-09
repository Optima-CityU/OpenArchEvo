import { PAPER, ICON, LOGO, initTheme, getJSON, getText, esc, fmt, iterColor, iterLabel, displayName, archURL, bindZoomableImages, copyText, lightbox, siteTheme, texPlain, frInfo, tip, rampCSS } from './core.js';
import { TABLE1, ABLATION, BIBTEX } from './paper.js';
import { mountEvolution } from './evolution.js';
import { mountInnerLoop } from './innerloop.js';
import { mountExplorer } from './explorer.js';

// Legacy deep links: index.html?arch=KEY or #arch=KEY → architecture page.
function redirect() {
  const q = new URLSearchParams(location.search).get('arch');
  const h = location.hash.match(/^#arch=(.+)$/) || location.hash.match(/^#((?:i\d\d-[\w-]+))$/);
  const key = q || (h && decodeURIComponent(h[1]));
  if (key) { location.replace(archURL(key)); return true; }
  return false;
}
const REDIRECTING = redirect();
window.addEventListener('hashchange', redirect);

initTheme();
document.querySelector('[data-logo]').insertAdjacentHTML('afterbegin', LOGO);
const codeLink = document.querySelector('[data-code-link]');
codeLink.href = PAPER.code; codeLink.innerHTML = ICON.github;

document.querySelector('[data-cta]').innerHTML = `
  <a class="btn primary" href="${PAPER.pdf}">${ICON.paper} Paper</a>
  <a class="btn" href="${PAPER.arxiv}">arXiv</a>
  <a class="btn" href="${PAPER.code}">${ICON.github} Code</a>
  <a class="btn ghost" href="${archURL('i09-s0557')}">Inspect the best block ${ICON.arrow}</a>`;

document.querySelector('[data-stats]').innerHTML = [
  ['151', '', 'new architectures trained, on top of 20 expert SNNs', 'Each one trained on both proxy tasks (WikiText-2 and ListOps) during the search.'],
  ['0.872', '', 'best search fitness; the SpikingDeltaNet expert scores 0.500', 'Composite fitness (paper App. D.2) of WikiText-2 perplexity and ListOps accuracy.'],
  ['26.4', 'PPL', 'NeuroGate on WikiText-103, vs. 27.5 for the ANN DeltaNet'],
  ['17–51', '×', 'lower estimated arithmetic energy than a dense Transformer', 'Architecture-level estimate for the discovered SNNs, not measured device energy (paper Table 1, App. E).'],
  ['132', 'GPU-days', 'estimated V100 training cost of the whole search'],
].map(([v, u, l, more]) => `<div class="stat"><b>${v}${u ? `<small>${u}</small>` : ''}</b><span>${l}${more ? ` <button class="info" type="button" data-tip="${esc(more)}" aria-label="${esc(more)}">i</button>` : ''}</span></div>`).join('');
document.querySelectorAll('.stat [data-tip]').forEach(b => {
  const show = e => tip(`<span>${esc(b.dataset.tip)}</span>`, e.type === 'focus' || e.type === 'click' ? { target: b } : e);
  b.addEventListener('mouseenter', show); b.addEventListener('focus', show); b.addEventListener('click', show);
  b.addEventListener('mouseleave', () => tip(null)); b.addEventListener('blur', () => tip(null));
});

document.querySelector('[data-bib]').textContent = BIBTEX;
const cb = document.querySelector('[data-copy-bib]');
cb.innerHTML = `${ICON.copy} Copy`;
cb.addEventListener('click', () => copyText(BIBTEX, cb));
document.querySelector('[data-foot-links]').innerHTML = `<a href="${PAPER.arxiv}">arXiv</a> · <a href="${PAPER.code}">GitHub</a> · <a href="#cite">BibTeX</a>`;
bindZoomableImages();

// ---------------------------------------------------------------- tables
(function tables() {
  const t = document.querySelector('[data-table1]');
  const cell = v => v == null ? '—' : (typeof v === 'number' ? v.toFixed(1) : esc(v));
  let h = `<thead><tr><th>Model</th>${TABLE1.cols.map(c => `<th>${esc(c)}</th>`).join('')}</tr></thead><tbody>`;
  for (const g of TABLE1.groups) {
    h += `<tr class="grp"><td colspan="10">${esc(g.name)}</td></tr>`;
    for (const r of g.rows) {
      const bold = TABLE1.bold[r[0]] || [];
      const name = r[10] ? `<a href="${archURL(r[10])}">${esc(r[0])}</a>` : esc(r[0]);
      h += `<tr class="${g.ours ? 'ours' : ''}"><td>${name}</td>${r.slice(1, 10).map((v, i) => `<td class="${bold.includes(i + 1) ? 'best' : ''}">${cell(v)}</td>`).join('')}</tr>`;
    }
  }
  t.innerHTML = h + '</tbody>';
  t.classList.add('t1');
  t.closest('.tablewrap').insertAdjacentHTML('afterend', '<p class="scroll-cue" aria-hidden="true">scroll →</p>');
  document.querySelector('[data-table1-foot]').textContent = TABLE1.foot + ' ' + TABLE1.caption;
  const ab = document.querySelector('[data-ablation]');
  let a = '<thead><tr><th>Variant</th><th>WT2 PPL ↓</th><th>Δ</th></tr></thead><tbody>';
  for (const g of ABLATION) {
    a += `<tr class="grp"><td colspan="3"><a href="${archURL(g.key)}">${esc(g.arch)}</a> · ${esc(g.kind)}</td></tr>`;
    for (const r of g.rows) a += `<tr><td>${esc(r[0])}</td><td>${r[1].toFixed(1)}</td><td>${r[2] ?? '—'}</td></tr>`;
  }
  ab.innerHTML = a + '</tbody>';
  ab.classList.add('abl');
  // right-edge shadow while a table can still scroll sideways
  document.querySelectorAll('.tablewrap').forEach(w => {
    const upd = () => { const more = w.scrollWidth - w.clientWidth; w.classList.toggle('scrolls', more > 2); w.classList.toggle('end', w.scrollLeft >= more - 2); };
    w.addEventListener('scroll', upd, { passive: true }); new ResizeObserver(upd).observe(w); upd();
  });
})();


// ---------------------------------------------------------------- section nav: scroll-spy for the desktop nav and the mobile strip
(function spy() {
  const links = [...document.querySelectorAll('.nav a[href^="#"], .mnav a[href^="#"]')];
  const ids = [...new Set(links.map(a => a.getAttribute('href').slice(1)))];
  // sections in document order (the mobile strip has an extra "Islands" chip, so link order is not page order)
  const secs = ids.map(id => document.getElementById(id)).filter(Boolean)
    .sort((a, b) => (a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING ? -1 : 1));
  const strip = document.querySelector('.mnav');
  const navs = [...document.querySelectorAll('.nav, .mnav')];
  let cur = null;
  const set = passed => {   // passed: ids of the sections above the reading line, in page order
    const key = passed.join(','); if (key === cur) return; cur = key;
    navs.forEach(nav => {
      // each nav marks the last passed section it has a link for (desktop: Islands falls back to Inner loop)
      const own = passed.filter(id => nav.querySelector(`a[href="#${id}"]`)).pop();
      nav.querySelectorAll('a[href^="#"]').forEach(a => { const on = a.getAttribute('href') === '#' + own; a.classList.toggle('on', on); if (on) a.setAttribute('aria-current', 'location'); else a.removeAttribute('aria-current'); });
      if (nav === strip && own && strip.scrollWidth > strip.clientWidth) {
        const a = strip.querySelector(`a[href="#${own}"]`);
        strip.scrollTo({ left: a.offsetLeft - strip.clientWidth / 2 + a.offsetWidth / 2, behavior: 'smooth' });
      }
    });
  };
  const onScroll = () => {
    const atEnd = window.scrollY + window.innerHeight >= document.documentElement.scrollHeight - 4;   // short last section (Cite)
    const y = atEnd ? Infinity : window.scrollY + window.innerHeight * 0.3;
    set(secs.filter(s => s.getBoundingClientRect().top + window.scrollY <= y).map(s => s.id));
  };
  window.addEventListener('scroll', () => requestAnimationFrame(onScroll), { passive: true });
  onScroll();
})();

// Run fn once when el comes near the viewport (lazy sections, thumbnails, data files).
// Anything still pending a few seconds after load is run anyway (in idle time), so the page has no empty
// resting state for readers who jump around, print, or take a full-page screenshot.
const pendingNear = new Set();
function whenNear(el, fn, margin = '600px') {
  if (!el) return;
  if (!('IntersectionObserver' in window)) { fn(); return; }
  let done = false;
  const run = () => { if (done) return; done = true; io.disconnect(); pendingNear.delete(run); fn(); };
  const io = new IntersectionObserver(es => { if (es.some(e => e.isIntersecting)) run(); }, { rootMargin: margin });
  io.observe(el);
  pendingNear.add(run);
}
function flushNearLater() {
  const idle = window.requestIdleCallback || (f => setTimeout(f, 1));
  const go = () => setTimeout(() => idle(() => [...pendingNear].forEach(f => f())), 2500);
  if (document.readyState === 'complete') go(); else window.addEventListener('load', go, { once: true });
}

// ---------------------------------------------------------------- data-driven parts
if (!REDIRECTING) Promise.all([getJSON('data/meta.json'), getJSON('data/architectures.json'), waitD3()]).then(([meta, archs]) => {
  const byKey = new Map(archs.map(a => [a.key, a]));
  const pick = document.querySelector('[data-pick]');
  const itersum = document.querySelector('[data-itersum]');
  const open = key => { location.href = archURL(key); };
  let inner = null;
  const innerRoot = document.querySelector('[data-inner]');
  // The inner-loop chart (and its round file) is mounted only when the reader gets near it.
  const ensureInner = () => {
    if (!inner) {
      inner = mountInnerLoop(innerRoot, { meta, archs, onSelect: key => { autoPick = false; evo.select(key); showPick(key); }, onOpen: open });
      inner.show(10);
    }
    return inner;
  };

  function summarize(t) {
    const cur = archs.filter(a => a.iter === t);
    const upto = archs.filter(a => a.iter <= t);
    const best = upto.reduce((m, a) => (a.fit > (m?.fit ?? -1) ? a : m), null);
    const front = meta.fronts[t].length;
    itersum.innerHTML = `<h4>${t === 0 ? 'Initial archive' : 'After iteration ' + t}</h4>
      <div class="big">${upto.length} <span class="muted" style="font-size:.9rem;font-weight:600">trained</span></div>
      <dl class="kv">
        <dt>${t === 0 ? 'Expert SNNs' : 'New this iteration'}</dt><dd>${cur.length}</dd>
        <dt>Points on the front</dt><dd>${front}</dd>
        <dt>Best fitness so far</dt><dd>${fmt(best?.fit, 3)}</dd>
      </dl>
      <p class="small muted" style="margin:8px 0 0">${best ? `Best: <a href="${archURL(best.key)}">${esc(displayName(best))}</a>` : ''}</p>
      ${t > 0 ? `<button class="btn sm" style="margin-top:12px" data-goinner>Inner loop of iteration ${t} ↓</button>` : ''}`;
    itersum.querySelector('[data-goinner]')?.addEventListener('click', () => { ensureInner().show(t); document.getElementById('inner').scrollIntoView({ behavior: 'smooth' }); });
  }

  // Focus-view thumbnails cropped to the token mixer (built by tools/build_data.py), inlined so they follow the site theme.
  const thumbCache = new Map();
  async function putThumb(box, key) {
    if (!thumbCache.has(key)) thumbCache.set(key, getText(`data/thumbs/${key}.svg`));
    const s = await thumbCache.get(key);
    if (!box.isConnected) return;
    box.innerHTML = s.replace(/<style>[\s\S]*?<\/style>/, '');
    const svg = box.querySelector('svg');
    if (!svg) return;
    svg.removeAttribute('width'); svg.removeAttribute('height');
    svg.setAttribute('data-theme', siteTheme()); svg.setAttribute('aria-hidden', 'true');
  }
  let pickKey = null, autoPick = true;   // autoPick: the panel follows the best-so-far until the reader picks a point
  async function showPick(key) {
    const a = byKey.get(key);
    if (!a) return;
    pickKey = key;
    const color = iterColor(a.iter);
    pick.innerHTML = `<h4>Selected architecture</h4>
      <p class="nm">${esc(displayName(a))}</p>
      <p class="meta"><span style="color:${color}">●</span> ${esc(a.key)} · ${iterLabel(a.iter)}</p>
      <div class="thumb" ${a.diagram ? '' : 'hidden'}${a.summary ? ` title="${esc(texPlain(a.summary))}"` : ''}></div>
      <dl class="kv">
        <dt>WT2 PPL</dt><dd>${fmt(a.wt2, 2)}</dd>
        <dt>ListOps acc.</dt><dd>${fmt(a.listops, 2)}%</dd>
        <dt>Fitness</dt><dd>${fmt(a.fit, 3)}</dd>
        <dt title="WT2 run, last logged epoch">Fire rate (WT2)</dt><dd${frInfo(a).note ? ` title="${esc(frInfo(a).note)}"` : ''}>${fmt(a.fr?.wt2, 3)}${frInfo(a).ep ? `<span class="muted"> · ep ${frInfo(a).ep.at}/${frInfo(a).ep.of}</span>` : ''}</dd>
      </dl>
      <p style="margin-top:14px"><a class="btn sm primary" href="${archURL(a.key)}">Open architecture ${ICON.arrow}</a></p>`;
    if (a.diagram) {
      const box = pick.querySelector('.thumb');
      try { await putThumb(box, key); } catch (e) { box.hidden = true; }
    }
  }
  pick.innerHTML = '<h4>Selected architecture</h4><p class="empty">Hover a dot for its metrics; click to pin it here. Use ← → on the focused chart to step through points.</p>';

  // During playback (or when stepping back) the pinned architecture may not exist yet at the shown iteration:
  // then pin the best architecture found so far instead, so the side panel never describes the future.
  const bestUpTo = t => archs.filter(a => a.iter <= t).reduce((m, a) => (a.fit > (m?.fit ?? -1) ? a : m), null);
  const evo = mountEvolution(document.querySelector('[data-evo]'), {
    archs, meta,
    onIter: t => {
      summarize(t);
      const p = pickKey && byKey.get(pickKey);
      if (autoPick || (p && p.iter > t)) { autoPick = true; const b = bestUpTo(t); if (b && b.key !== pickKey) evo.select(b.key); }
    },
    onSelect: (key, fromChart) => { if (fromChart) autoPick = false; showPick(key); },
    onOpen: open,
  });
  summarize(10);
  // Start with the best architecture pinned so the side panel is never an empty instruction box.
  const bestKey = archs.reduce((m, a) => (a.fit > (m?.fit ?? -1) ? a : m), null)?.key;
  if (bestKey) evo.select(bestKey);
  window.addEventListener('oae-theme', e => document.querySelectorAll('.thumb svg').forEach(s => s.setAttribute('data-theme', e.detail)));
  whenNear(innerRoot, ensureInner);

  // final Pareto front (task plane, after iteration 10)
  const fr = document.querySelector('[data-front]');
  const members = (meta.finalFront || meta.fronts[10]).map(k => byKey.get(k)).filter(Boolean).sort((p, q) => p.wt2 - q.wt2);
  fr.innerHTML = members.map(a => `<a class="fcard" href="${archURL(a.key)}">
      <div class="thumb" data-thumb="${a.key}"></div>
      <span class="f-top"><i class="dot" style="background:${iterColor(a.iter)}"></i>${iterLabel(a.iter)}</span>
      <span class="f-nm">${esc(displayName(a))}</span>
      <span class="f-mx"><span><b>${fmt(a.wt2, 2)}</b>WT2 PPL</span><span><b>${fmt(a.listops, 1)}%</b>ListOps acc.</span></span>
    </a>`).join('');
  fr.querySelectorAll('[data-thumb]').forEach(box => whenNear(box, async () => {
    const a = byKey.get(box.dataset.thumb);
    if (!a?.diagram) { box.hidden = true; return; }
    try { await putThumb(box, a.key); } catch (e) { /* ignore */ }
  }, '400px'));


  mountExplorer({ archs, root: document.getElementById('explorer'), zooCount: meta.counts?.zoo });
  flushNearLater();
}).catch(err => {
  console.error(err);
  document.querySelector('[data-evo]').innerHTML = `<p class="note">Could not load the search data (${esc(err.message)}). Serve the site over HTTP.</p>`;
});

// Hero curve: the paper's search trajectory (Fig. 4). x = number of evaluated architectures in evaluation order
// (the 20 experts first), y = composite fitness, dots coloured by iteration, best-so-far staircase from the first
// expert with a star at every improvement, dashed baseline at 0.500 (SpikingDeltaNet). Points are digitized from the
// paper's vector figure (data/paper_trajectory.json, built by tools/build_data.py).
const starPath = (cx, cy, R, r = R * 0.45) => 'M' + Array.from({ length: 10 }, (_, i) => {
  const t = -Math.PI / 2 + i * Math.PI / 5, k = i % 2 ? r : R;
  return `${(cx + k * Math.cos(t)).toFixed(1)},${(cy + k * Math.sin(t)).toFixed(1)}`;
}).join('L') + 'Z';
function heroTrajectory(T) {
  const fig = document.querySelector('[data-hero-front]');
  if (!fig || !T?.points?.length) return;
  fig.hidden = false;
  const P = T.points.map(([n, it, f, imp]) => ({ n, it, f, imp }));
  const N = Math.max(...P.map(p => p.n));
  const W = Math.round(Math.max(320, Math.min(1180, fig.clientWidth || 1100))), narrow = W < 640;
  const H = narrow ? Math.round(W * 0.74) : 300, L = 34, R = narrow ? 10 : 44, Tp = 24, B = 36, Y = [0.2, 0.9];
  const sx = v => L + (v + 3) / (N + 6) * (W - L - R);
  const sy = v => H - B - (v - Y[0]) / (Y[1] - Y[0]) * (H - Tp - B);
  const theme = siteTheme();
  // iteration boundaries (faint) and dots
  const starts = [...new Set(P.map(p => p.it))].map(it => Math.min(...P.filter(p => p.it === it).map(p => p.n))).filter(n => n > 0);
  const vlines = starts.map(n => `<line x1="${sx(n - 0.5).toFixed(1)}" x2="${sx(n - 0.5).toFixed(1)}" y1="${Tp}" y2="${H - B}" class="hf-it"/>`).join('');
  const dots = P.map(p => `<circle cx="${sx(p.n).toFixed(1)}" cy="${sy(p.f).toFixed(1)}" r="${narrow ? 2.6 : 3.4}" data-t="${p.it}" fill="${iterColor(p.it, theme)}" style="--d:${(0.1 + p.it * 0.11).toFixed(2)}s"/>`).join('');
  // best-so-far staircase, starting at the first expert
  let best = P[0].f, d = `M${sx(P[0].n).toFixed(1)},${sy(best).toFixed(1)}`;
  const imps = [P[0]];
  for (const p of P.slice(1)) if (p.f > best) { d += `H${sx(p.n).toFixed(1)}V${sy(p.f).toFixed(1)}`; best = p.f; imps.push(p); }
  d += `H${sx(N + 2).toFixed(1)}`;
  const stars = imps.map((p, i) => `<path d="${starPath(sx(p.n), sy(p.f), narrow ? 6.2 : 7.5)}" class="hf-star" style="--d:${(1.1 + i * 0.09).toFixed(2)}s"/>`).join('');
  const top = imps[imps.length - 1];
  const yt = [0.3, 0.5, 0.7, 0.9].map(v => `<text x="${L - 6}" y="${(sy(v) + 3.5).toFixed(1)}" class="hf-t" text-anchor="end">${v.toFixed(1)}</text>`).join('');
  const xt = (narrow ? [0, 50, 100, 150] : [0, 25, 50, 75, 100, 125, 150]).map(v => `<text x="${sx(v).toFixed(1)}" y="${H - B + 15}" class="hf-t" text-anchor="middle">${v}</text>`).join('');
  fig.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Search trajectory (paper Fig. 4): composite fitness of all ${P.length} evaluated architectures in evaluation order, experts first. The best-so-far line rises from ${P[0].f.toFixed(2)} for the first expert, passes the 0.500 baseline, and reaches ${top.f.toFixed(3)}.">
      <g class="hf-its">${vlines}</g>
      <line x1="${L}" x2="${W - R}" y1="${sy(T.baseline ?? 0.5).toFixed(1)}" y2="${sy(T.baseline ?? 0.5).toFixed(1)}" class="hf-ref"/>
      <line x1="${L}" x2="${W - R}" y1="${H - B}" y2="${H - B}" class="hf-ax"/>${yt}${xt}
      <g class="hf-dots">${dots}</g>
      <path d="${d}" class="hf-line" pathLength="1"/>
      <g class="hf-stars">${stars}</g>
      <text x="${(sx(top.n) - 10).toFixed(1)}" y="${(sy(top.f) + 4).toFixed(1)}" class="hf-k" text-anchor="end">${top.f.toFixed(3)}</text>
      <text x="${L}" y="${H - 3}" class="hf-t">Number of evaluated architectures →</text>
      <text x="${L - 26}" y="${Tp - 10}" class="hf-t">↑ Composite fitness</text>
    </svg>
    <figcaption><span class="hf-leg"><i style="background:${iterColor(0, theme)}"></i>experts</span><span class="hf-leg hf-ramp">iteration 1<i style="background:${rampCSS(theme)}"></i>10</span><span class="hf-leg"><i class="hf-ln"></i>best so far</span><span class="hf-leg"><i class="hf-ln dash"></i>baseline 0.500</span>
      <span class="hf-cap">Search trajectory (paper Fig. 4). <a href="#evolution">Explore it ↓</a></span></figcaption>`;
  fig.hidden = false;
  if (fig.dataset.themed) return;
  fig.dataset.themed = '1';
  window.addEventListener('oae-theme', e => {
    fig.querySelectorAll('circle[data-t]').forEach(c => c.setAttribute('fill', iterColor(+c.dataset.t, e.detail)));
    const legs = fig.querySelectorAll('.hf-leg i');
    legs[0].style.background = iterColor(0, e.detail); legs[1].style.background = rampCSS(e.detail);
  });
}
getJSON('data/paper_trajectory.json').then(T => {
  heroTrajectory(T);
  const fig = document.querySelector('[data-hero-front]');
  let w = fig.clientWidth, tm = 0;
  new ResizeObserver(() => { if (Math.abs(fig.clientWidth - w) < 8) return; w = fig.clientWidth; clearTimeout(tm); tm = setTimeout(() => { fig.classList.add('static'); heroTrajectory(T); }, 120); }).observe(fig);
}).catch(() => { /* hero curve is optional */ });

function waitD3() {
  return new Promise((res, rej) => {
    if (window.d3) return res();
    let n = 0; const iv = setInterval(() => { if (window.d3) { clearInterval(iv); res(); } else if (++n > 200) { clearInterval(iv); rej(new Error('d3 failed to load')); } }, 50);
  });
}
