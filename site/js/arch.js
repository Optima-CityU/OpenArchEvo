// Architecture detail page: arch.html?arch=<key>  (also #<key>, #arch=<key>)  ·  zoo programs: arch.html?zoo=<key>
import { PAPER, ICON, LOGO, initTheme, getJSON, getText, esc, fmt, iterColor, iterLabel, displayName, archURL, md, tip, lightbox, panZoomViewer, copyText, siteTheme, tex, texPlain, frInfo } from './core.js';

initTheme();
document.querySelector('[data-logo]').insertAdjacentHTML('afterbegin', LOGO);
const cl = document.querySelector('[data-code-link]'); cl.href = PAPER.code; cl.innerHTML = ICON.github;
const root = document.querySelector('[data-root]');

function keyFromURL() {
  const p = new URLSearchParams(location.search);
  if (p.get('zoo')) return { zoo: p.get('zoo') };
  let k = p.get('arch');
  if (!k && location.hash) { const h = decodeURIComponent(location.hash.slice(1)); k = h.startsWith('arch=') ? h.slice(5) : h; }
  return { arch: k };
}

const ARCH_NOTE = {
  wt2: 'WikiText-2 perplexity after the 30-epoch search proxy training. Lower is better.',
  listops: 'ListOps accuracy (%) after 25 training epochs on one third of the ListOps training set (search proxy).',
  fit: 'Composite fitness of WT2 PPL and ListOps accuracy (paper formula, App. D.2). The SpikingDeltaNet expert (58.28 PPL, 53.0%) scores 0.500.',
  pred: 'The surrogate’s prediction for this program at search time, on the search’s own score scale.',
  fr: 'Average firing rate at the last training epoch that logged one (WT2 run; ListOps where a log exists). If a log stopped reporting the rate early, the epoch is given.',
};
const DETAIL_BTN = { core: 'Open core state update', s4: 'Open S4D kernel detail', gsa: 'Open GSA state update' };
const DETAIL_TITLE = { core: 'core state update', s4: 'S4D kernel', gsa: 'GSA state update' };
const detailBtn = d => DETAIL_BTN[d] || `Open ${d} detail`;
const detailTitle = d => DETAIL_TITLE[d] || `${d} detail`;

function info(text) { return `<button class="info" type="button" data-tip="${esc(text)}" aria-label="${esc(text)}">i</button>`; }
function bindInfo(scope) {
  scope.querySelectorAll('[data-tip]').forEach(b => {
    const show = e => tip(`<span>${esc(b.dataset.tip)}</span>`, e.type === 'focus' ? { target: b } : e);
    b.addEventListener('mouseenter', show); b.addEventListener('focus', show);
    b.addEventListener('mouseleave', () => tip(null)); b.addEventListener('blur', () => tip(null));
    b.addEventListener('click', e => { e.preventDefault(); show({ target: b }); });
  });
}

// ------------------------------------------------------------------ code viewer
function waitHljs(ms = 4000) {
  return new Promise(res => { if (window.hljs) return res(window.hljs); const t0 = Date.now(); const iv = setInterval(() => { if (window.hljs || Date.now() - t0 > ms) { clearInterval(iv); res(window.hljs || null); } }, 50); });
}
function splitLines(html) {
  const out = []; const stack = []; let line = '';
  const re = /(<span[^>]*>)|(<\/span>)|([^<]+)/g; let m;
  while ((m = re.exec(html))) {
    if (m[1]) { stack.push(m[1]); line += m[1]; }
    else if (m[2]) { stack.pop(); line += m[2]; }
    else {
      const parts = m[3].split('\n');
      parts.forEach((p, i) => {
        if (i > 0) { line += '</span>'.repeat(stack.length); out.push(line); line = stack.join(''); }
        line += p;
      });
    }
  }
  out.push(line + '</span>'.repeat(stack.length));
  return out;
}
// Leading licence/import block (after an optional module docstring) up to the first top-level class/def.
function headerSpan(src) {
  let a = 0;
  if (/^\s*("""|''')/.test(src[0] || '')) {
    const q = src[0].trim().slice(0, 3);
    const one = src[0].trim().length > 3 && src[0].trim().endsWith(q);
    a = one ? 1 : src.findIndex((l, i) => i > 0 && l.includes(q)) + 1;
  }
  let b = src.findIndex((l, i) => i >= a && /^(class|def|async def)\s|^@/.test(l));
  if (b < 0) return null;
  while (b > a && /^\s*(#.*)?$/.test(src[b - 1]) && !/^#\s*-\*-|copyright/i.test(src[b - 1])) b--;
  return b - a >= 8 ? [a + 1, b] : null;   // 1-based inclusive line range
}
async function renderCode(box, code, { evolved = null } = {}) {
  const hljs = await waitHljs();
  let html;
  try { html = hljs ? hljs.highlight(code, { language: 'python', ignoreIllegals: true }).value : esc(code); } catch (e) { html = esc(code); }
  const lines = splitLines(html);
  const ev = new Set();
  for (const [a, b] of evolved || []) for (let l = a; l <= b; l++) ev.add(l);
  let hdr = headerSpan(code.split('\n'));
  if (hdr && [...ev].some(l => l >= hdr[0] && l <= hdr[1])) hdr = null;
  box.innerHTML = lines.map((l, i) => {
    const n = i + 1;
    const fold = hdr && n === hdr[0] ? `<button type="button" class="fold" data-fold aria-expanded="false"><span class="chev" aria-hidden="true">▸</span> ${hdr[1] - hdr[0] + 1} lines of licence header and imports</button>` : '';
    return `${fold}<span class="ln${ev.has(n) ? ' ev' : ''}${hdr && n >= hdr[0] && n <= hdr[1] ? ' hdr' : ''}" data-l="${n}">${l || ' '}</span>`;
  }).join('');
  if (hdr) {
    box.classList.add('folded');
    box.querySelector('[data-fold]').addEventListener('click', e => {
      const open = box.classList.toggle('folded') === false;
      e.currentTarget.setAttribute('aria-expanded', open);
      e.currentTarget.querySelector('.chev').textContent = open ? '▾' : '▸';
    });
  }
  // open on the evolved part: scroll the code box (not the page) to the first evolved line
  const first = ev.size ? box.querySelector(`.ln[data-l="${Math.min(...ev)}"]`) : null;
  if (first) box.scrollTop = Math.max(0, first.offsetTop - box.offsetTop - 3 * first.offsetHeight);
}
function unfoldFor(box, ranges) {
  if (!box.classList.contains('folded') || !ranges) return;
  if (ranges.some(([a, b]) => { for (let l = a; l <= b; l++) if (box.querySelector(`.ln.hdr[data-l="${l}"]`)) return true; return false; })) box.querySelector('[data-fold]')?.click();
}
function parseRange(s) {
  if (!s) return null;
  const ranges = String(s).split(/[,\s]+/).map(p => p.match(/L?(\d+)(?:\s*-\s*L?(\d+))?/)).filter(Boolean).map(m => [+m[1], +(m[2] || m[1])]);
  return ranges.length ? ranges : null;
}
function highlight(box, ranges, { scroll = true, page = false } = {}) {
  box.querySelectorAll('.ln.hl').forEach(e => e.classList.remove('hl'));
  if (!ranges) return;
  unfoldFor(box, ranges);
  let first = null;
  for (const [a, b] of ranges) for (let l = a; l <= b; l++) { const e = box.querySelector(`.ln[data-l="${l}"]`); if (e) { e.classList.add('hl'); first = first || e; } }
  if (first && scroll) box.scrollTo({ top: Math.max(0, first.offsetTop - box.offsetTop - box.clientHeight * 0.3), behavior: 'smooth' });
  if (first && page) {
    const r = box.getBoundingClientRect();
    if (r.top < 60 || r.top > window.innerHeight * 0.6) box.closest('.codebox').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

// ------------------------------------------------------------------ diagram
function prepSVG(text) {
  const tpl = document.createElement('template');
  tpl.innerHTML = text.trim();
  const svg = tpl.content.querySelector('svg');
  if (!svg) return null;
  svg.removeAttribute('width'); svg.removeAttribute('height');
  svg.querySelectorAll('title').forEach(t => { t.textContent = texPlain(t.textContent); });
  // drop node-level ids that could collide with page ids (markers keep their prefixed ids)
  svg.querySelectorAll('[id]').forEach(e => { if (!/marker|pattern|clipPath|linearGradient|filter/i.test(e.tagName)) e.removeAttribute('id'); });
  return svg;
}

function wireNodes(svg, detail, codeBox, prefix = '') {
  svg.querySelectorAll('.node[data-id]').forEach(n => {
    const id = n.getAttribute('data-id');
    const meta = detail.nodes?.[prefix ? `${prefix}/${id}` : id] || {};
    const code = n.getAttribute('data-code') || meta.code;
    if (!code && !meta.tip) return;
    if (code) n.setAttribute('data-code', code);
    n.querySelector(':scope > title')?.remove();   // the custom tooltip replaces the native one
    n.setAttribute('tabindex', '0');
    n.setAttribute('role', 'button');
    const label = meta.label || n.querySelector('.lbl')?.textContent || id;
    n.setAttribute('aria-label', `${texPlain(label)}${code ? ', source lines ' + code : ''}`);
    const html = `<b>${tex(label)}</b>${meta.tip ? `<span>${tex(meta.tip)}</span><br>` : ''}${code ? `<span class="k">source ${esc(code.replace(/-/g, '–'))}</span>` : ''}<div class="hint">Click to jump to the code</div>`;
    const enter = e => { n.classList.add('lit'); tip(html, e.type === 'focus' ? { target: n } : e); if (code) highlight(codeBox, parseRange(code)); };
    const leave = () => { n.classList.remove('lit'); tip(null); };
    n.addEventListener('mouseenter', enter); n.addEventListener('mousemove', e => tip(html, e)); n.addEventListener('focus', enter);
    n.addEventListener('mouseleave', leave); n.addEventListener('blur', leave);
    const go = () => { tip(null); if (code) { document.querySelector('dialog.lb')?.close(); highlight(codeBox, parseRange(code), { page: true }); } };
    n.addEventListener('click', go);
    n.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(); } });
  });
}

// Union of the token-mixer panel(s): <rect class="panel"> followed by <text class="cap">Token mixer…</text>, inside the translated main group.
function mixerBox(svg) {
  const bx = [];
  svg.querySelectorAll('rect.panel').forEach(r => {
    const cap = r.nextElementSibling;
    if (!cap || !cap.classList.contains('cap') || !/^\s*Token mixer/i.test(cap.textContent)) return;
    const m = (r.parentNode.getAttribute('transform') || '').match(/translate\(\s*([-\d.]+)[ ,]+([-\d.]+)/);
    const ox = m ? +m[1] : 0, oy = m ? +m[2] : 0;
    const x = +r.getAttribute('x') + ox, y = +r.getAttribute('y') + oy;
    bx.push([x, y, x + +r.getAttribute('width'), y + +r.getAttribute('height')]);
  });
  if (!bx.length) return null;
  // evolved blocks outside the mixer (e.g. an evolved output projection) belong in the crop too
  const inner = svg.querySelector('g[transform^="translate"]');
  const t = (inner?.getAttribute('transform') || '').match(/translate\(\s*([-\d.]+)[ ,]+([-\d.]+)/);
  // (only when the mixer itself has no evolved block, so the crop stays a readable token-mixer view otherwise)
  const mix = [Math.min(...bx.map(b => b[0])), Math.min(...bx.map(b => b[1])), Math.max(...bx.map(b => b[2])), Math.max(...bx.map(b => b[3]))];
  const nv = [];
  svg.querySelectorAll('.node.novel .sh').forEach(sh => {
    if (sh.closest('.legend')) return;
    try { const b = sh.getBBox(); const ox = t ? +t[1] : 0, oy = t ? +t[2] : 0; nv.push([b.x + ox, b.y + oy, b.x + ox + b.width, b.y + oy + b.height]); } catch (e) { /* not rendered */ }
  });
  const inside = b => b[0] >= mix[0] && b[2] <= mix[2] && b[1] >= mix[1] && b[3] <= mix[3];
  if (nv.length && !nv.some(inside)) nv.forEach(b => bx.push([b[0] - 12, b[1] - 14, b[2] + 14, b[3] + 12]));
  const p = 12, x0 = Math.min(...bx.map(b => b[0])) - p, y0 = Math.min(...bx.map(b => b[1])) - p;
  return [x0, y0, Math.max(...bx.map(b => b[2])) + p - x0, Math.max(...bx.map(b => b[3])) + p - y0];
}

const PENDING_SVG = `<svg viewBox="0 0 200 90" aria-hidden="true"><g fill="none" stroke="currentColor" stroke-width="1.5" stroke-dasharray="4 4"><rect x="10" y="30" width="44" height="30" rx="7"/><rect x="78" y="18" width="44" height="54" rx="7"/><rect x="146" y="30" width="44" height="30" rx="7"/></g><path d="M54 45h24M122 45h24" stroke="currentColor" stroke-width="1.5"/></svg>`;

// ------------------------------------------------------------------ page
async function main() {
  const sel = keyFromURL();
  if (sel.zoo) return zooPage(sel.zoo);
  const archs = await getJSON('data/architectures.json');
  const order = [...archs].sort((a, b) => (a.iter - b.iter) || (a.sample - b.sample));
  const a = archs.find(x => x.key === sel.arch);
  if (!a) {
    root.innerHTML = `<div class="notfound"><h1>Architecture not found</h1><p class="muted">No architecture with key “${esc(sel.arch || '')}”.</p><p><a class="btn" href="./#explorer">Browse all architectures</a></p></div>`;
    return;
  }
  document.title = `${displayName(a)} · OpenArchEvo`;
  const idx = order.indexOf(a), prev = order[idx - 1], next = order[idx + 1];
  const [detail, code] = await Promise.all([getJSON(`data/arch/${a.key}.json`), getText(`data/code/${a.key}.py`).catch(() => null)]);
  const archByKey = new Map(archs.map(x => [x.key, x]));
  const color = iterColor(a.iter);
  const nameNote = a.nameSrc === 'descriptive' ? 'Descriptive name, written for this site from the code' : a.nameSrc === 'rationale' ? 'Name taken from the design rationale' : a.nameSrc === 'paper' ? 'Name used in the paper' : '';
  const fr = frInfo(a), frL = frInfo(a, 'listops');
  const frTip = ARCH_NOTE.fr + (a.frN?.wt2 ? ` This expert has ${a.frN.wt2} WT2 training logs; the first is shown.` : '');
  const frSub = [fr.v == null ? 'no WT2 log recovered' : `WT2${a.frN?.wt2 ? ` (1 of ${a.frN.wt2} logs)` : ''}${fr.note ? ` · ${fr.note}` : ' · last epoch'}`,
    `ListOps ${fmt(frL.v, 3)}${frL.note ? ` (${frL.note})` : ''}`].join(' · ');

  root.innerHTML = `
  <div class="a-head">
    <div class="crumbs"><a href="./">OpenArchEvo</a><span>/</span><a href="./#explorer">Architectures</a><span>/</span><span>${esc(a.key)}</span>
      <span class="a-nav">${prev ? `<a class="btn sm" href="${archURL(prev.key)}" aria-label="Previous architecture">← ${esc(prev.key)}</a>` : ''}${next ? `<a class="btn sm" href="${archURL(next.key)}" aria-label="Next architecture">${esc(next.key)} →</a>` : ''}</span></div>
    <div class="a-title"><h1${nameNote ? ` title="${esc(nameNote)}"` : ''}>${esc(displayName(a))}</h1><span class="key">${esc(a.key)}</span></div>
    <div class="a-tags">
      <span class="tag" style="background:transparent;border:1px solid ${color};color:var(--ink)"><span style="color:${color}">●</span> ${a.expert ? 'Initial expert' : iterLabel(a.iter)}</span>
      ${a.base ? `<span class="tag plain">state update: ${esc(a.base)} type</span>` : ''}
    </div>
    ${a.summary ? `<p class="a-sum">${tex(a.summary)}</p>` : ''}
  </div>
  <div class="metrics m5">
    <div class="metric"><div class="lb2">WT2 PPL ${info(ARCH_NOTE.wt2)}</div><div class="v">${fmt(a.wt2, 2)}</div><div class="sub">search proxy</div></div>
    <div class="metric"><div class="lb2">ListOps acc. ${info(ARCH_NOTE.listops)}</div><div class="v">${fmt(a.listops, 2)}<small>%</small></div><div class="sub">search proxy</div></div>
    <div class="metric"><div class="lb2">Fitness ${info(ARCH_NOTE.fit)}</div><div class="v">${fmt(a.fit, 3)}</div><div class="sub">SpikingDeltaNet = 0.500</div></div>
    <div class="metric"><div class="lb2">Predicted score ${info(ARCH_NOTE.pred)}</div><div class="v ${a.pred ? '' : 'na'}">${a.pred ? fmt(a.pred.fit, 3) : 'n/a'}</div><div class="sub">${a.pred ? `pred. ${fmt(a.pred.wt2, 1)} PPL · ${fmt(a.pred.listops, 1)}%` : a.expert ? 'initial expert' : 'not recorded'}</div></div>
    <div class="metric"><div class="lb2">Fire rate ${info(frTip)}</div><div class="v ${fr.v == null ? 'na' : ''}">${fmt(fr.v, 3)}</div><div class="sub">${esc(frSub)}</div></div>
  </div>

  <section class="dg panel" aria-label="Block diagram">
    <div class="dg-bar">
      <div class="seg" role="group" aria-label="Diagram theme"><button data-dt="light" aria-pressed="${siteTheme() !== 'dark'}">Day</button><button data-dt="dark" aria-pressed="${siteTheme() === 'dark'}">Night</button></div>
      <label class="switch"><input type="checkbox" data-focus> Evolved mechanisms</label>
      <span class="spacer"></span>
      <span data-details></span>
      <button class="btn sm" data-enlarge>${ICON.fit} Enlarge</button>
    </div>
    <div class="dg-grid">
      <div><div class="dg-wrap" data-wrap><div class="dg-canvas ${siteTheme() === 'dark' ? 'dark' : 'light'}" data-canvas></div></div><div class="dg-hint ${siteTheme() === 'dark' ? 'dark' : 'light'}" data-hint></div></div>
      <div class="dg-side" data-side></div>
    </div>
  </section>

  <div class="a-grid">
    <section class="panel codebox" id="code" aria-label="Source code">
      <div class="codebar"><span class="fn" title="${esc(detail.codeFile ? 'Archive (evaluated) source · sha256 ' + detail.codeFile.sha256 : '')}">${esc(detail.codeFile?.label || a.key + '.py')}</span><button class="btn sm" data-copy>${ICON.copy} Copy</button><a class="btn sm" href="data/code/${esc(a.key)}.py" download>Download</a></div>
      <div class="code" data-code tabindex="0" aria-label="Source code">${code ? '' : '<p class="note" style="padding:16px">Source not available.</p>'}</div>
    </section>
    <div class="panel">
      <div class="box"><h3>Design rationale <span class="tag plain">verbatim · LLM-generated</span></h3>
        <div class="prose" data-rat></div>
        <p class="rat-src">The rationale is the original <code>idea</code> text saved with this program. It states what the LLM intended; the diagram follows what the code does.</p></div>
      <div class="box"><h3>Firing rate during training</h3><div data-fire></div></div>
    </div>
  </div>
  <details class="prov panel" id="provenance">
    <summary><h3>Provenance</h3><span class="muted small">archive record and source hash</span></summary>
    <dl class="kv stack" data-prov></dl>
  </details>`;
  bindInfo(root);

  // ---- rationale
  const ratBox = root.querySelector('[data-rat]');
  const rat = (detail.rationale || '').replace(/^\s*Idea:\s*/i, '');
  ratBox.innerHTML = rat.trim() ? md(rat) : '<p class="muted">No rationale was saved for this program.</p>';
  if (ratBox.scrollHeight > 700) {
    ratBox.classList.add('rat-clamp');
    const more = document.createElement('button'); more.className = 'btn sm'; more.textContent = 'Read the full rationale'; more.style.marginTop = '10px';
    more.addEventListener('click', () => { const open = ratBox.classList.toggle('rat-clamp'); more.textContent = open ? 'Read the full rationale' : 'Collapse'; });
    ratBox.after(more);
  }
  if (rat.trim() && !/[.!?)"'`*:]\s*$/.test(rat.trim())) ratBox.insertAdjacentHTML('beforeend', '<p class="note">The saved rationale ends mid-sentence: it is truncated in the archive and shown verbatim.</p>');

  // ---- fire curves
  const fireBox = root.querySelector('[data-fire]');
  const curves = detail.fireCurves || {};
  const tasks = Object.keys(curves).filter(k => curves[k]?.length);
  if (tasks.length) {
    fireBox.innerHTML = tasks.map(t => {
      const pts = curves[t].filter(p => p[1] != null);
      const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
      const x0 = Math.min(...xs), x1 = Math.max(...xs), y1 = Math.max(...ys, 0.01);
      const P = pts.map(p => `${(4 + (p[0] - x0) / Math.max(1, x1 - x0) * 292).toFixed(1)},${(58 - (p[1] / y1) * 50).toFixed(1)}`).join(' L');
      return `<div style="margin-bottom:10px"><div class="small" style="display:flex;justify-content:space-between"><span>${t === 'wt2' ? `WikiText-2${a.frN?.wt2 ? ` <span class="muted">(first of ${a.frN.wt2} logs)</span>` : ''}` : 'ListOps'}</span><span class="mono muted">last ${fmt(ys[ys.length - 1], 3)} · max ${fmt(y1, 3)}</span></div>
        <svg class="spark" viewBox="0 0 300 64" preserveAspectRatio="none" role="img" aria-label="Firing rate per epoch, ${t}"><line x1="4" x2="296" y1="58" y2="58" stroke="var(--line)"/><path d="M${P}" stroke="${t === 'wt2' ? 'var(--accent)' : 'var(--lav-ink)'}" vector-effect="non-scaling-stroke"/></svg>
        <div class="note">epochs ${x0}–${x1}${(f => f.ep ? ` of ${f.ep.of}: the log stopped reporting the rate after epoch ${f.ep.at}${f.ep.why === 'nan' ? ' (training loss became NaN)' : f.ep.why === 'silent' ? ' (the rate had fallen to 0.000)' : ''}` : '')(frInfo(a, t))}</div></div>`;
    }).join('');
  } else fireBox.innerHTML = '<p class="muted small" style="margin:0">No training log with firing rates was recovered.</p>';

  // ---- provenance
  root.querySelector('[data-prov]').innerHTML = `
    <dt>Archive record</dt><dd>${a.expert ? `expert archive · sample ${esc(a.sample)}` : `${esc(a.group)} · sample ${esc(a.sample)} (${iterLabel(a.iter)})`} · key ${esc(a.key)}</dd>
    ${a.pred?.island != null ? `<dt>Island at proposal</dt><dd>${esc(a.pred.island)}</dd>` : ''}
    <dt>Code shown</dt><dd style="font-family:var(--f-body);font-size:.8rem;text-align:left">Archive version, the exact bytes evaluated in the search; diagram line links refer to this file.${detail.codeFile ? `<br><span class="mono" style="font-size:.74rem">sha256 ${esc(detail.codeFile.sha256)}</span>` : ''}</dd>
    ${detail.zooFile ? `<dt>Zoo file</dt><dd>${esc(detail.zooFile)}</dd>` : ''}`;

  // ---- code
  const codeBox = root.querySelector('[data-code]');
  if (code) {
    // evolved blocks of the diagram (spec nodes marked novel) → terracotta gutter in the code view
    const evolved = Object.values(detail.nodes || {}).filter(n => n.novel && n.code).flatMap(n => parseRange(n.code) || []);
    renderCode(codeBox, code, { evolved });
    root.querySelector('[data-copy]').addEventListener('click', e => copyText(code, e.currentTarget));
  }

  // ---- diagram
  const canvas = root.querySelector('[data-canvas]');
  const side = root.querySelector('[data-side]');
  const hint = root.querySelector('[data-hint]');
  const state = { theme: siteTheme() === 'dark' ? 'dark' : 'light', focus: false };
  const narrow = window.matchMedia('(max-width: 760px)');
  const callouts = Object.entries(detail.callouts || {});
  side.innerHTML = `<h4>Evolved mechanisms</h4>${callouts.length ? `<ol class="callouts">${callouts.map(([n, t]) => {
    const m = t.match(/^([^:]{2,60}):\s*(.*)$/s);
    const cap = x => x.charAt(0).toUpperCase() + x.slice(1);
    return `<li data-n="${esc(n)}" tabindex="0"><span class="n">${esc(n)}</span><span>${m ? `<b>${tex(m[1])}.</b> ${tex(cap(m[2]))}` : tex(t)}</span></li>`;
  }).join('')}</ol>` : `<p class="muted small">${a.expert ? 'Initial expert: a hand-written spiking adaptation of a published architecture.' : a.spec ? (detail.notes?.length ? 'No single block is marked as evolved; the distinguishing details are listed below.' : 'No block is marked as evolved.') : 'Mechanism notes will appear when the diagram spec is ready.'}</p>`}
    ${detail.notes?.length ? `<div class="a-tags" style="margin-top:14px">${detail.notes.map(n => {
      // notes are strings (legacy) or {text, kind: evolved|loss|inherited, novel?}; inherited facts get a neutral chip
      const o = typeof n === 'string' ? { text: n } : n || {};
      return `<span class="tag ${o.kind === 'inherited' ? 'plain' : 'accent'}">${o.novel ? `<span class="tag-n">${esc(o.novel)}</span>` : ''}${tex(o.text)}</span>`;
    }).join('')}</div>` : ''}
`;

  let svgMain = null;
  if (a.diagram) {
    try {
      svgMain = prepSVG(await getText(`data/diagrams/${a.key}.svg`));
    } catch (e) { svgMain = null; }
  }
  const wrap = root.querySelector('[data-wrap]');
  let fullVB = null, mixVB = null;
  if (svgMain) {
    canvas.appendChild(svgMain);
    if (narrow.matches) { state.focus = true; root.querySelector('[data-focus]').checked = true; }
    applyState(svgMain);
    svgMain.setAttribute('role', 'img');
    svgMain.setAttribute('aria-label', `Block diagram of ${displayName(a)}`);
    wireNodes(svgMain, detail, codeBox);
    const vb = svgMain.viewBox.baseVal;
    fullVB = [vb.x, vb.y, vb.width, vb.height];
    mixVB = mixerBox(svgMain);
    // Phones: the canvas shows the token mixer (focus view) at a readable scale with sideways scrolling; a tap opens
    // the full diagram. Wider screens: the whole block at ≥ 0.9× (labels ≈ 11 px), scrolling sideways when it is wider.
    const fitCanvas = () => {
      const small = narrow.matches && mixVB;
      const v = small ? mixVB : fullVB;
      svgMain.setAttribute('viewBox', v.join(' '));
      const k = small ? 0.85 : 0.9;
      svgMain.style.setProperty('--minw', `${Math.round(v[2] * k)}px`);
      svgMain.style.setProperty('--maxw', `${Math.round(v[2] * (small ? 0.85 : 1.15))}px`);
      wrap.classList.toggle('crop', !!small);
      hint.innerHTML = narrow.matches
        ? `<span>${small ? 'Token mixer shown · swipe sideways' : 'Swipe sideways'}</span><span><kbd>tap</kbd> to open the full diagram, then tap a block to jump to its code</span>`
        : `<span><kbd>hover</kbd> a block to light up its source lines</span><span><kbd>click</kbd> to jump to the code</span>${a.details.length ? '<span><kbd>⤢</kbd> on the core opens the state-update math</span>' : ''}<span><kbd>Enlarge</kbd> full screen with zoom and pan</span>`;
      requestAnimationFrame(edges);
    };
    const edges = () => {
      const more = canvas.scrollWidth - canvas.clientWidth;
      wrap.classList.toggle('more-r', more > 4 && canvas.scrollLeft < more - 4);
      wrap.classList.toggle('more-l', canvas.scrollLeft > 4);
    };
    canvas.addEventListener('scroll', edges, { passive: true });
    new ResizeObserver(edges).observe(canvas);
    narrow.addEventListener?.('change', fitCanvas);
    fitCanvas();
    const tapBtn = document.createElement('button');
    tapBtn.className = 'dg-tap'; tapBtn.type = 'button';
    tapBtn.innerHTML = `${ICON.fit}<span>Tap to open full diagram</span>`;
    tapBtn.addEventListener('click', e => { e.stopPropagation(); enlarge(); });
    wrap.appendChild(tapBtn);
    // On phones the in-page diagram is an overview: any tap opens the full-screen viewer, where blocks are big enough to hit.
    canvas.addEventListener('click', e => {
      if (!narrow.matches) return;
      e.preventDefault(); e.stopPropagation(); enlarge();
    }, true);
    svgMain.querySelectorAll('.expand[data-detail]').forEach(ex => {
      const did = ex.getAttribute('data-detail');
      ex.setAttribute('aria-label', detailBtn(did));
      const open = e => { e.stopPropagation(); openDetail(did); };
      ex.addEventListener('click', open);
      ex.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(e); } });
    });
  } else {
    canvas.innerHTML = `<div class="dg-pending">${PENDING_SVG}<div><b>Diagram in progress</b><br><span class="small">The block diagram for this architecture is still being drawn from its source code. The code and metrics below are complete.</span></div></div>`;
    root.querySelector('[data-enlarge]').hidden = true;
    root.querySelectorAll('[data-dt],[data-focus]').forEach(e => (e.disabled = true));
  }
  const detBox = root.querySelector('[data-details]');
  detBox.innerHTML = a.details.map(d => `<button class="btn sm accent" data-open-detail="${esc(d)}">${ICON.plus} ${esc(detailBtn(d))}</button>`).join(' ');
  detBox.querySelectorAll('[data-open-detail]').forEach(b => b.addEventListener('click', () => openDetail(b.dataset.openDetail)));

  function applyState(svg) {
    if (!svg) return;
    svg.setAttribute('data-theme', state.theme);
    svg.classList.toggle('focus', state.focus);
  }
  root.querySelectorAll('[data-dt]').forEach(b => b.addEventListener('click', () => {
    state.theme = b.dataset.dt;
    root.querySelectorAll('[data-dt]').forEach(o => o.setAttribute('aria-pressed', o === b));
    [canvas, hint].forEach(el => { el.classList.toggle('dark', state.theme === 'dark'); el.classList.toggle('light', state.theme === 'light'); });
    applyState(svgMain);
  }));
  root.querySelector('[data-focus]').addEventListener('change', e => { state.focus = e.target.checked; applyState(svgMain); });

  function zoomable(svg, wrapBg, title, prefix) {
    const vb = svg.viewBox.baseVal;
    svg.style.maxWidth = 'none';
    const lb = panZoomViewer({ title, content: svg, width: vb.width || 1600, height: vb.height || 600, bg: wrapBg,
      foot: narrow.matches ? 'Tap a block to jump to its source lines.' : 'Hover a block to see its source lines; click it to jump to the code on the page.' });
    wireNodes(svg, detail, codeBox, prefix);
    return lb;
  }
  async function openDetail(did) {
    let svg;
    try { svg = prepSVG(await getText(`data/diagrams/${a.key}__${did}.svg`)); } catch (e) { svg = null; }
    if (!svg) { lightbox({ title: 'Detail not available yet', body: `<div class="dg-pending">${PENDING_SVG}<p>The detail diagram “${esc(did)}” has not been rendered yet.</p></div>` }); return; }
    applyState(svg);
    const node = svgMain?.querySelector(`.expand[data-detail="${did}"]`)?.closest('.node');
    zoomable(svg, state.theme === 'dark' ? '#1f1828' : '#fffdfa', `${displayName(a)} · ${detailTitle(did)}`, node?.getAttribute('data-id') || '');
  }
  root.querySelector('[data-enlarge]').addEventListener('click', () => enlarge());
  function enlarge() {
    if (!svgMain) return;
    const clone = svgMain.cloneNode(true);
    if (fullVB) clone.setAttribute('viewBox', fullVB.join(' '));
    clone.style.removeProperty('--minw'); clone.style.removeProperty('--maxw');
    applyState(clone);
    const lb = zoomable(clone, state.theme === 'dark' ? '#1f1828' : '#fffdfa', displayName(a), '');
    clone.querySelectorAll('.expand[data-detail]').forEach(ex => ex.addEventListener('click', e => { e.stopPropagation(); lb.close(); openDetail(ex.getAttribute('data-detail')); }));
  }

  // Evolved-mechanism list <-> diagram: hovering or focusing a numbered mechanism lights up the blocks that carry its badge.
  const nodesFor = (svg, n) => svg ? [...svg.querySelectorAll('.node')].filter(nd => [...nd.querySelectorAll('.badge text')].some(t => t.textContent.trim() === n)) : [];
  // A click pins the highlight, so it survives the scroll to the diagram (the list moves away from the pointer).
  let pinned = null;
  const lis = [...side.querySelectorAll('.callouts li[data-n]')];
  lis.forEach(li => {
    const n = li.dataset.n;
    const on = () => { nodesFor(svgMain, n).forEach(nd => nd.classList.add('lit')); li.classList.add('lit'); };
    const off = (force) => { if (pinned === n && force !== true) return; nodesFor(svgMain, n).forEach(nd => nd.classList.remove('lit')); li.classList.remove('lit'); };
    li._off = off;
    li.addEventListener('mouseenter', on); li.addEventListener('mouseleave', off);
    li.addEventListener('focus', on); li.addEventListener('blur', off);
    const show = () => {
      if (pinned === n) { pinned = null; off(true); return; }
      const prev = pinned; pinned = n; lis.forEach(o => { if (o.dataset.n === prev) o._off(true); });
      if (!state.focus) { const cb = root.querySelector('[data-focus]'); cb.checked = true; cb.dispatchEvent(new Event('change')); }
      on(); canvas.scrollIntoView({ behavior: 'smooth', block: 'center' });
    };
    li.addEventListener('click', show);
    li.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); show(); } });
    li.title = 'Show on the diagram';
  });
  if (svgMain) svgMain.querySelectorAll('.node').forEach(nd => {
    const ns = [...nd.querySelectorAll('.badge text')].map(t => t.textContent.trim());
    if (!ns.length) return;
    nd.addEventListener('mouseenter', () => ns.forEach(n => side.querySelector(`li[data-n="${CSS.escape(n)}"]`)?.classList.add('lit')));
    nd.addEventListener('mouseleave', () => ns.forEach(n => side.querySelector(`li[data-n="${CSS.escape(n)}"]`)?.classList.remove('lit')));
  });
}

async function zooPage(key) {
  const zoo = await getJSON('data/zoo.json').catch(() => []);
  const z = zoo.find(x => x.key === key);
  if (!z) { root.innerHTML = `<div class="notfound"><h1>Program not found</h1><p class="muted">No zoo program “${esc(key)}”.</p><p><a class="btn" href="./#explorer">Back to the explorer</a></p></div>`; return; }
  document.title = `${z.name || z.key} · OpenArchEvo zoo`;
  const label = { baseline: 'baseline', asi_arch_discovered: 'ASI-Arch program', fullspike_rewrite: 'fully-spiking LLM rewrite' };
  root.innerHTML = `
  <div class="a-head">
    <div class="crumbs"><a href="./">OpenArchEvo</a><span>/</span><a href="./#explorer">Program zoo</a><span>/</span><span>${esc(z.key)}</span></div>
    <div class="a-title"><h1>${esc(z.name || z.key)}</h1><span class="key">${esc(z.key)}</span></div>
    <div class="a-tags"><span class="tag">${esc(label[z.subset] || z.subset || '')}</span>${z.base ? `<span class="tag plain">${esc(z.base)}</span>` : ''}<span class="tag warn">legacy surrogate-study protocol · not the main-search archive</span></div>
    ${z.parent ? `<p class="a-sum">LLM rewrite of ASI parent <a href="arch.html?zoo=asi-${String(z.parent).padStart(4, '0')}">${esc(z.parent)}</a>.</p>` : ''}
  </div>
  <div class="metrics m4">
    <div class="metric"><div class="lb2">WT2 PPL (legacy)</div><div class="v ${z.wt2 == null ? 'na' : ''}">${fmt(z.wt2, 2)}</div></div>
    <div class="metric"><div class="lb2">ListOps acc. (legacy)</div><div class="v ${z.listops == null ? 'na' : ''}">${fmt(z.listops, 2)}${z.listops == null ? '' : '<small>%</small>'}</div></div>
    <div class="metric"><div class="lb2">Fire rate WT2</div><div class="v ${z.fr?.wt2 == null ? 'na' : ''}">${fmt(z.fr?.wt2, 3)}</div></div>
    <div class="metric"><div class="lb2">Fire rate ListOps</div><div class="v ${z.fr?.listops == null ? 'na' : ''}">${fmt(z.fr?.listops, 3)}</div></div>
  </div>
  <div class="a-grid">
    <section class="panel codebox"><div class="codebar"><span class="fn">zoo/${esc(z.file || z.key + '.py')}</span><button class="btn sm" data-copy>${ICON.copy} Copy</button></div><div class="code" data-code tabindex="0"></div></section>
    <div class="panel">
      ${z.motivation ? `<div class="box"><h3>Motivation <span class="tag plain">verbatim</span></h3><div class="prose">${md(z.motivation)}</div></div>` : ''}
      <div class="box"><h3>Provenance</h3><dl class="kv"><dt>Metrics source</dt><dd>${esc(z.results || 'none')}</dd>${z.notes ? `<dt>Note</dt><dd style="font-family:var(--f-body);text-align:left;font-size:.8rem">${esc(z.notes)}</dd>` : ''}</dl>
      <p class="note" style="margin-top:12px">ASI-Arch programs (and baselines) are ANN token-mixer layers from an earlier surrogate study; their spiking metrics come from a legacy pipeline that is not part of the main search.</p></div>
    </div>
  </div>`;
  const box = root.querySelector('[data-code]');
  if (z.code) {
    const code = await getText(`data/zoo/${z.key}.py`);
    renderCode(box, code);
    root.querySelector('[data-copy]').addEventListener('click', e => copyText(code, e.currentTarget));
  } else box.innerHTML = '<p class="note" style="padding:16px">Source not available.</p>';
}

main().catch(err => { console.error(err); root.innerHTML = `<p class="note" style="padding:80px 0">Could not load this page (${esc(err.message)}).</p>`; });
window.addEventListener('hashchange', () => { if (!new URLSearchParams(location.search).get('arch')) location.reload(); });
