// Representation section (#representation): fingerprint anatomy, SWSP / firing-rate illustration,
// two-architecture fingerprint comparison (real logged data) and the offline surrogate study (paper Fig. 6).
import { getJSON, esc, tip, archURL, displayName, iterLabel, reduceMotion } from './core.js';
import FP from './representation-data.js';

const root = document.getElementById('representation');

// ------------------------------------------------------------------ feature metadata (paper Table 6)
// schema key -> [label, group, description]; group order follows the paper table.
const FEAT = {
  zcp_synflow: ['SynFlow', 'probe', 'Synaptic flow'],
  zcp_gradnorm: ['GradNorm', 'probe', 'Gradient ℓ2 norm'],
  zcp_snip: ['SNIP', 'probe', 'Connection sensitivity'],
  zcp_jacob_cov: ['Jacob<sub>cov</sub>', 'probe', 'Jacobian covariance'],
  zcp_naswot: ['NASWOT', 'probe', 'Activation overlap'],
  zcp_swap: ['SWSP', 'probe', 'Sample-wise spiking patterns'],
  zcp_fire_rate: ['FireRate', 'probe', 'Mean spike activity at initialization'],
  arch_total_params: ['Params', 'module', 'Total trainable parameters'],
  arch_num_spiking_linear: ['SpkLinear', 'module', 'Spiking linear modules'],
  arch_num_conv1d: ['Conv1d', 'module', '1-D convolutions'],
  arch_gate_layer_ratio: ['GateRatio', 'module', 'Fraction of gating layers'],
  arch_depth_proxy: ['Depth', 'struct', 'Shape-changing layers'],
  arch_num_expand: ['Expand', 'struct', 'Dimension-increase steps'],
  arch_num_contract: ['Contract', 'struct', 'Dimension-decrease steps'],
  arch_bottleneck_ratio: ['BneckRatio', 'struct', 'Min dim / model dim'],
  arch_max_expand_ratio: ['MaxExpand', 'struct', 'Largest expansion factor'],
  arch_avg_ffn_expansion: ['FFN<sub>exp</sub>', 'struct', 'Average FFN expansion'],
  arch_ffn_param_ratio: ['FFN<sub>param</sub>', 'struct', 'FFN parameter fraction'],
  arch_ffn_macs_ratio: ['FFN<sub>mac</sub>', 'struct', 'FFN MAC fraction'],
  arch_macs_per_param: ['MAC/Param', 'struct', 'Compute density'],
  arch_largest_subgraph_ratio: ['SubgraphR', 'struct', 'Largest traced subgraph ratio'],
};
const GROUPS = [
  { id: 'probe', name: 'Initialization-time probes', how: 'Forward (and backward) passes of the untrained network on a fixed ListOps minibatch', hl: ['zcp_swap', 'zcp_fire_rate'] },
  { id: 'module', name: 'Module statistics', how: 'Read from the module inventory' },
  { id: 'struct', name: 'Structural statistics', how: 'Read from the execution trace and tensor shapes' },
];
const ORDER = GROUPS.flatMap(g => Object.keys(FEAT).filter(k => FEAT[k][1] === g.id));
const plain = s => s.replace(/<\/?su[bp]>/g, '');

// ------------------------------------------------------------------ helpers
function onVisible(el, fn, margin = '200px') {
  if (!('IntersectionObserver' in window)) { fn(); return; }
  const io = new IntersectionObserver(es => { if (es.some(e => e.isIntersecting)) { io.disconnect(); fn(); } }, { rootMargin: margin });
  io.observe(el);
}
function waitD3() {
  return new Promise((res, rej) => {
    if (window.d3) return res(window.d3);
    let n = 0; const iv = setInterval(() => { if (window.d3) { clearInterval(iv); res(window.d3); } else if (++n > 200) { clearInterval(iv); rej(new Error('d3 failed to load')); } }, 50);
  });
}
function seg(options, current, onPick, label) {
  const el = document.createElement('div');
  el.className = 'seg'; el.setAttribute('role', 'group'); el.setAttribute('aria-label', label);
  el.innerHTML = options.map(([v, t]) => `<button type="button" data-v="${v}" aria-pressed="${v === current}">${t}</button>`).join('');
  el.addEventListener('click', e => {
    const b = e.target.closest('button'); if (!b) return;
    el.querySelectorAll('button').forEach(x => x.setAttribute('aria-pressed', String(x === b)));
    onPick(b.dataset.v);
  });
  return el;
}
const NS = 'http://www.w3.org/2000/svg';
const svgEl = (tag, attrs = {}, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };

// ------------------------------------------------------------------ 1. fingerprint groups
function renderGroups(el) {
  el.innerHTML = GROUPS.map(g => {
    const keys = ORDER.filter(k => FEAT[k][1] === g.id);
    return `<div class="rep-g rep-g-${g.id}">
      <div class="rep-g-top"><span class="rep-g-n">${keys.length}</span><div><h4>${g.name}</h4><p>${g.how}</p></div></div>
      <ul class="rep-chips">${keys.map(k => `<li class="${g.hl?.includes(k) ? 'hl' : ''}" title="${esc(FEAT[k][2])}">${FEAT[k][0]}</li>`).join('')}</ul>
    </div>`;
  }).join('') + '<p class="rep-g-sum"><b>21 numbers</b> per architecture, all measured before any weight training.</p>';
}

// ------------------------------------------------------------------ 2. SWSP / firing-rate illustration
function renderSpikes(fig) {
  const N = 6, S = 4, T = 24;              // neurons, probe samples, time steps
  const CW = 13, CH = 9, GAP = 7, LX = 30, TOP = 4;
  const H = TOP + S * (N * CH) + (S - 1) * GAP + 4, W = LX + T * CW + 2;
  const MODES = {
    diverse: { label: 'Input-sensitive', note: 'Responses differ across inputs: many distinct patterns.' },
    blind: { label: 'Input-blind', note: 'Same spikes for every input: firing rate unchanged, but only two patterns.' },
    silent: { label: 'Near-silent', note: 'Hardly any spikes: low firing rate and few patterns.' },
  };
  let mode = 'diverse';

  const ctrl = document.createElement('div'); ctrl.className = 'rep-spk-ctrl';
  const body = document.createElement('div'); body.className = 'rep-spk-body';
  fig.prepend(ctrl); fig.appendChild(body);
  ctrl.appendChild(seg(Object.entries(MODES).map(([k, m]) => [k, m.label]), mode, v => { mode = v; restart(); }, 'Network behaviour'));
  const playBtn = document.createElement('button');
  playBtn.type = 'button'; playBtn.className = 'btn sm rep-play';
  ctrl.appendChild(playBtn);

  const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, class: 'rep-raster', role: 'img', 'aria-label': 'Spike rasters of six neurons for four probe inputs' });
  body.appendChild(svg);
  const band = svgEl('rect', { class: 'rep-ph', x: LX, y: 0, width: CW, height: H, rx: 3 }, svg);
  const cells = [];   // cells[s][n][t]
  for (let s = 0; s < S; s++) {
    const y0 = TOP + s * (N * CH + GAP);
    const t = svgEl('text', { x: 0, y: y0 + N * CH / 2 + 4, class: 'rep-rl' }, svg); t.textContent = `x${s + 1}`;
    cells.push([]);
    for (let n = 0; n < N; n++) {
      cells[s].push([]);
      for (let k = 0; k < T; k++) cells[s][n].push(svgEl('rect', { x: LX + k * CW + 2, y: y0 + n * CH + 1.5, width: CW - 4, height: CH - 3, rx: 1.5, class: 'c' }, svg));
    }
  }
  const read = document.createElement('div'); read.className = 'rep-read';
  read.innerHTML = `
    <div class="rep-meter"><span class="k">Firing rate</span><b data-fr>0%</b><span class="bar"><i data-frbar></i></span></div>
    <div class="rep-meter"><span class="k">SWSP <span class="muted">distinct patterns</span></span><b data-sw>0</b><span class="of">of 16 possible</span></div>
    <div class="rep-pats" data-pats aria-hidden="true"></div>
    <p class="rep-mnote" data-mnote></p>`;
  body.appendChild(read);
  const patsEl = read.querySelector('[data-pats]');
  const pats = [];
  for (let p = 0; p < 16; p++) {
    const d = document.createElement('span'); d.className = 'pat';
    d.innerHTML = [3, 2, 1, 0].map(b => `<i class="${(p >> b) & 1 ? 'on' : ''}"></i>`).join('');
    patsEl.appendChild(d); pats.push(d);
  }
  const frEl = read.querySelector('[data-fr]'), frBar = read.querySelector('[data-frbar]'), swEl = read.querySelector('[data-sw]'), noteEl = read.querySelector('[data-mnote]');

  // deterministic data per mode
  function rng(seed) { return () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  function gen(m) {
    const r = rng(m === 'diverse' ? 7 : m === 'blind' ? 7 : 11);
    const rate = Array.from({ length: N }, () => 0.2 + r() * 0.25);
    const X = Array.from({ length: S }, () => Array.from({ length: N }, () => new Array(T).fill(0)));
    for (let n = 0; n < N; n++) for (let k = 0; k < T; k++) {
      if (m === 'blind') { const v = r() < rate[n] ? 1 : 0; for (let s = 0; s < S; s++) X[s][n][k] = v; }
      else for (let s = 0; s < S; s++) X[s][n][k] = r() < (m === 'silent' ? 0.02 : rate[n]) ? 1 : 0;
    }
    return X;
  }
  let X = gen(mode), step = 0, timer = null, playing = !reduceMotion(), visible = false;

  function draw(upto) {   // reveal columns [0, upto)
    let spikes = 0; const seen = new Set();
    for (let k = 0; k < T; k++) for (let n = 0; n < N; n++) {
      let code = 0;
      for (let s = 0; s < S; s++) {
        const on = k < upto && X[s][n][k] === 1;
        const c = cells[s][n][k];
        c.setAttribute('class', k < upto ? (on ? 'c on' : 'c seen') : 'c');
        if (k < upto) { spikes += X[s][n][k]; code = (code << 1) | X[s][n][k]; }
      }
      if (k < upto) seen.add(code);
    }
    const total = Math.max(1, upto * N * S), fr = upto ? spikes / total : 0;
    frEl.textContent = `${Math.round(fr * 100)}%`; frBar.style.width = `${Math.min(100, fr * 200)}%`;
    swEl.textContent = String(seen.size);
    pats.forEach((d, p) => d.classList.toggle('lit', seen.has(p)));
    const col = Math.min(upto, T) - 1;
    band.setAttribute('x', LX + Math.max(0, col) * CW);
    band.style.opacity = upto > 0 && upto <= T && playing ? '1' : '0';
  }
  function tick() {
    step++;
    if (step > T + 10) step = 0;           // hold the full raster ~1.4 s, then sweep again
    draw(Math.min(step, T));
  }
  function sync() {
    const run = playing && visible && !document.hidden;
    if (run && !timer) timer = setInterval(tick, 140);
    if (!run && timer) { clearInterval(timer); timer = null; }
    playBtn.textContent = playing ? 'Pause' : 'Play';
    playBtn.setAttribute('aria-pressed', String(!playing));
  }
  function restart() {
    X = gen(mode); noteEl.textContent = MODES[mode].note;
    step = playing ? 0 : T; draw(step); sync();
  }
  playBtn.addEventListener('click', () => { playing = !playing; if (playing && step >= T) step = 0; if (!playing) { step = T; draw(T); } sync(); });
  if ('IntersectionObserver' in window) new IntersectionObserver(es => { visible = es.some(e => e.isIntersecting); sync(); }, { threshold: 0.15 }).observe(fig);
  else visible = true;
  document.addEventListener('visibilitychange', sync);
  noteEl.textContent = MODES[mode].note;
  step = playing ? 0 : T; draw(step); sync();
}

// ------------------------------------------------------------------ 3. two-architecture comparison
const C = FP.scale_c;
const scaled = v => v.map((x, j) => { const [l, u] = FP.bounds[j]; return C * (2 * (x - l) / (u - l) - 1); });
function cosine(a, b) {
  let ab = 0, aa = 0, bb = 0;
  for (let i = 0; i < a.length; i++) { ab += a[i] * b[i]; aa += a[i] * a[i]; bb += b[i] * b[i]; }
  return aa && bb ? ab / Math.sqrt(aa * bb) : 0;
}
const fmtRaw = v => {
  const a = Math.abs(v);
  if (a >= 1e4) return Math.round(v).toLocaleString('en-US');
  if (a >= 100) return v.toFixed(1);
  if (Number.isInteger(v)) return String(v);
  if (a >= 1) return v.toFixed(2);
  return v.toFixed(3);
};
const LIM = 1.7;                                   // display range of the scaled value
const pos = x => 50 + 50 * Math.max(-LIM, Math.min(LIM, x)) / LIM;

function renderCompare(el, archs) {
  const byKey = new Map(archs.map(a => [a.key, a]));
  const usable = archs.filter(a => FP.fp[a.key]);
  const idx = new Map(FP.keys.map((k, j) => [k, j]));
  const PRESETS = [
    ['Very different', 'i00-deltanet', 'i03-s0335'],
    ['Close relatives', 'i00-deltanet', 'i09-s0557'],
    ['Two experts', 'i00-deltanet', 'i00-mamba-s6'],
  ].filter(p => byKey.has(p[1]) && byKey.has(p[2]) && FP.fp[p[1]] && FP.fp[p[2]]);
  let A = PRESETS[0]?.[1] ?? usable[0].key, B = PRESETS[0]?.[2] ?? usable[usable.length - 1].key;

  const optHTML = () => {
    const iters = [...new Set(usable.map(a => a.iter))].sort((x, y) => x - y);
    return iters.map(i => `<optgroup label="${iterLabel(i)}">${usable.filter(a => a.iter === i).map(a => `<option value="${a.key}">${esc(displayName(a))}</option>`).join('')}</optgroup>`).join('');
  };
  el.innerHTML = `
    <div class="rep-cmp-top">
      <label class="rep-pick a"><span class="sw" aria-hidden="true"></span><span class="sr-only">Architecture A</span><select data-a>${optHTML()}</select></label>
      <span class="rep-vs" aria-hidden="true">vs</span>
      <label class="rep-pick b"><span class="sw" aria-hidden="true"></span><span class="sr-only">Architecture B</span><select data-b>${optHTML()}</select></label>
      <div class="rep-presets" role="group" aria-label="Example pairs">${PRESETS.map((p, i) => `<button type="button" class="btn sm" data-p="${i}">${p[0]}</button>`).join('')}</div>
    </div>
    <div class="rep-sim" data-sim aria-live="polite"></div>
    <div class="rep-rows" data-rows></div>
    <p class="note rep-cmp-note">Bars start at the midpoint of each feature&rsquo;s calibration range, which is how the cosine similarity sees them. Values beyond the plotted range end in a notch. Hover a row for the raw values.</p>`;
  const selA = el.querySelector('[data-a]'), selB = el.querySelector('[data-b]'), rowsEl = el.querySelector('[data-rows]'), simEl = el.querySelector('[data-sim]');

  // static rows
  const lo = pos(-C), hi = pos(C);
  rowsEl.innerHTML = GROUPS.map(g => `<div class="rep-rg"><div class="rep-rgh">${g.name}</div>${ORDER.filter(k => FEAT[k][1] === g.id).map(k => `
      <div class="rep-row${g.hl?.includes(k) ? ' hl' : ''}" data-k="${k}" tabindex="0">
        <span class="rep-lab">${FEAT[k][0]}</span>
        <span class="rep-track"><i class="tk" style="left:${lo}%"></i><i class="mid"></i><i class="tk" style="left:${hi}%"></i><i class="bar a"></i><i class="bar b"></i></span>
        <span class="rep-raw"><span class="a"></span><span class="b"></span></span>
      </div>`).join('')}</div>`).join('');
  const rows = [...rowsEl.querySelectorAll('.rep-row')];

  function update() {
    selA.value = A; selB.value = B;
    const va = FP.fp[A], vb = FP.fp[B], sa = scaled(va), sb = scaled(vb);
    rows.forEach(r => {
      const j = idx.get(r.dataset.k);
      [['a', sa[j], va[j]], ['b', sb[j], vb[j]]].forEach(([w, s, raw]) => {
        const bar = r.querySelector(`.bar.${w}`), p = pos(s);
        bar.style.left = `${Math.min(50, p)}%`; bar.style.width = `${Math.abs(p - 50)}%`;
        bar.classList.toggle('clip-r', s > LIM); bar.classList.toggle('clip-l', s < -LIM);
        r.querySelector(`.rep-raw .${w}`).textContent = fmtRaw(raw);
      });
    });
    const s = cosine(sa, sb), a = byKey.get(A), b = byKey.get(B);
    simEl.innerHTML = `<span class="rep-simv"><span class="k">Fingerprint similarity s<sub>fp</sub></span><b>${s.toFixed(2)}</b></span>
      <span class="rep-simbar" aria-hidden="true"><i style="left:${50 + 50 * s}%"></i></span>
      <span class="rep-simt muted">${s > 0.7 ? 'Very similar behaviour.' : s > 0.2 ? 'Partly similar.' : s > -0.2 ? 'Largely unrelated.' : 'Opposite profiles.'}
      <a href="${archURL(A)}">${esc(displayName(a))}</a> · <a href="${archURL(B)}">${esc(displayName(b))}</a></span>`;
    el.querySelectorAll('[data-p]').forEach(btn => { const p = PRESETS[+btn.dataset.p]; btn.setAttribute('aria-pressed', String(p[1] === A && p[2] === B)); });
  }
  selA.addEventListener('change', () => { A = selA.value; update(); });
  selB.addEventListener('change', () => { B = selB.value; update(); });
  el.querySelector('.rep-presets').addEventListener('click', e => { const b = e.target.closest('[data-p]'); if (!b) return; const p = PRESETS[+b.dataset.p]; A = p[1]; B = p[2]; update(); });
  const showTip = (r, ev) => {
    const k = r.dataset.k, j = idx.get(k), [l, u] = FP.bounds[j];
    tip(`<b>${FEAT[k][0]}</b><div class="k">${esc(FEAT[k][2])}</div><table>
      <tr><td>${esc(displayName(byKey.get(A)))}</td><td>${fmtRaw(FP.fp[A][j])}</td></tr>
      <tr><td>${esc(displayName(byKey.get(B)))}</td><td>${fmtRaw(FP.fp[B][j])}</td></tr>
      <tr><td class="k">5th–95th pct.</td><td>${fmtRaw(l)} – ${fmtRaw(u)}</td></tr></table>`, ev);
  };
  rows.forEach(r => {
    r.addEventListener('mousemove', ev => showTip(r, ev));
    r.addEventListener('mouseleave', () => tip(null));
    r.addEventListener('focus', ev => showTip(r, ev));
    r.addEventListener('blur', () => tip(null));
  });
  update();
}

// ------------------------------------------------------------------ 4. surrogate study (paper Fig. 6 a, b)
// Kendall τ, five-fold CV repeated three times; WT2 n = 223, ListOps n = 54 (App. "Offline Surrogate Studies").
// The two values printed on the paper figure are exact (0.46 / 0.65); the others are read from the bar
// geometry of figs/surrogate_composite_aligned.pdf (vector PDF), accurate to about ±0.01.
const SUR = {
  reg: {
    rows: [['SVR'], ['KNN'], ['Random forest'], ['XGBoost'], ['LightGBM'], ['TabPFN-2.5']],
    wt2: [0.161, 0.219, 0.345, 0.399, 0.410, 0.46],
    lo: [0.366, 0.459, 0.565, 0.565, 0.593, 0.65],
    take: 'With the complete fingerprint, <b>TabPFN-2.5 ranks candidates best</b> among the tested regressors on both tasks.',
  },
  feat: {
    rows: [['5 classical proxies', 'probe'], ['+ SWSP, FireRate', 'probe'], ['+ Params', 'module'], ['+ other module stats', 'module'], ['+ dimension flow', 'struct'], ['+ resource allocation', 'struct'], ['+ subgraph (full)', 'struct']],
    wt2: [0.237, 0.309, 0.330, 0.416, 0.427, 0.434, 0.46],
    lo: [0.506, 0.509, 0.565, 0.586, 0.608, 0.577, 0.65],
    take: 'Adding <b>SWSP and firing rate</b> lifts WT2 ranking above the classical proxies, and the <b>complete 21-D fingerprint</b> ranks best on both tasks.',
  },
};
function renderSurrogate(fig, d3) {
  let view = 'reg';
  const head = document.createElement('div'); head.className = 'rep-chart-top';
  head.appendChild(seg([['reg', 'Which regressor?'], ['feat', 'Which features?']], view, v => { view = v; draw(); }, 'Surrogate study'));
  const legend = document.createElement('span'); legend.className = 'note';
  legend.innerHTML = 'Kendall τ between predicted and measured outcome · higher is better';
  head.appendChild(legend);
  const facets = document.createElement('div'); facets.className = 'rep-facets';
  const table = document.createElement('table'); table.className = 'sr-only';
  const cap = fig.querySelector('.rep-take');
  fig.insertBefore(head, cap); fig.insertBefore(facets, cap); fig.appendChild(table);
  const src = document.createElement('p'); src.className = 'note rep-src';
  src.innerHTML = 'Offline studies before the main search (paper Fig. 6a–b, App. surrogate studies): 5-fold CV ×3. Highlighted values as printed in the paper; others read from the figure (±0.01).';
  fig.appendChild(src);

  const tasks = [['wt2', 'WikiText-2', 223], ['lo', 'ListOps', 54]];
  function draw() {
    const D = SUR[view];
    facets.innerHTML = '';
    tasks.forEach(([tk, name, n]) => {
      const box = document.createElement('div'); box.className = 'rep-facet';
      box.innerHTML = `<div class="rep-ft"><b>${name}</b> <span class="muted">${n} architectures</span></div>`;
      facets.appendChild(box);
      const W = Math.max(280, Math.round(box.clientWidth || 300)), rowH = 26, M = { l: view === 'reg' ? 98 : 150, r: 40, t: 4, b: 26 };
      const H = M.t + M.b + D.rows.length * rowH;
      const x = d3.scaleLinear().domain([0, 0.7]).range([M.l, W - M.r]);
      const y = d3.scaleBand().domain(D.rows.map((_, i) => i)).range([M.t, H - M.b]).padding(0.3);
      const svg = d3.select(box).append('svg').attr('viewBox', `0 0 ${W} ${H}`).attr('width', '100%').attr('role', 'img')
        .attr('aria-label', `${name}: Kendall tau by ${view === 'reg' ? 'regressor' : 'fingerprint input'}`);
      svg.append('g').attr('class', 'grid').attr('transform', `translate(0,${H - M.b})`).call(d3.axisBottom(x).ticks(W < 360 ? 4 : 7).tickSize(-(H - M.b - M.t)).tickFormat(''));
      svg.append('g').attr('class', 'axis').attr('transform', `translate(0,${H - M.b})`).call(d3.axisBottom(x).ticks(W < 360 ? 4 : 7).tickSizeOuter(0).tickFormat(d3.format('.1f')));
      const last = D.rows.length - 1;
      const g = svg.append('g').selectAll('g').data(D.rows).join('g').attr('class', (r, i) => `rep-bar${i === last ? ' best' : ''}${r[1] ? ' g-' + r[1] : ''}`);
      g.append('text').attr('class', 'rep-bl').attr('x', M.l - 10).attr('y', (_, i) => y(i) + y.bandwidth() / 2).attr('dy', '0.35em').attr('text-anchor', 'end').text(r => r[0]);
      g.append('path').attr('class', 'm').attr('d', (_, i) => {
        const x0 = x(0), x1 = x(D[tk][i]), yy = y(i), h = y.bandwidth(), rr = Math.min(4, h / 2, x1 - x0);
        return `M${x0},${yy}H${x1 - rr}Q${x1},${yy} ${x1},${yy + rr}V${yy + h - rr}Q${x1},${yy + h} ${x1 - rr},${yy + h}H${x0}Z`;
      });
      g.append('rect').attr('class', 'hit').attr('x', M.l - 6).attr('width', W - M.l).attr('y', (_, i) => y(i) - 3).attr('height', y.bandwidth() + 6).attr('fill', 'transparent')
        .on('mousemove', (ev, r) => { const i = D.rows.indexOf(r); tip(`<b>${esc(r[0])}</b><div class="k">${name}</div>Kendall τ ${i === last ? '' : '≈ '}<span class="mono">${D[tk][i].toFixed(2)}</span>`, ev); })
        .on('mouseleave', () => tip(null));
      svg.append('text').attr('class', 'rep-bv').attr('x', x(D[tk][last]) + 6).attr('y', y(last) + y.bandwidth() / 2).attr('dy', '0.35em').text(D[tk][last].toFixed(2));
    });
    cap.innerHTML = D.take;
    table.innerHTML = `<caption>Kendall τ by ${view === 'reg' ? 'regressor' : 'fingerprint input'}</caption><tr><th>Row</th><th>WikiText-2</th><th>ListOps</th></tr>` +
      D.rows.map((r, i) => `<tr><td>${esc(r[0])}</td><td>${D.wt2[i]}</td><td>${D.lo[i]}</td></tr>`).join('');
  }
  draw();
  let w = facets.clientWidth;
  new ResizeObserver(() => { if (Math.abs(facets.clientWidth - w) > 8) { w = facets.clientWidth; draw(); } }).observe(facets);
}

// ------------------------------------------------------------------ mount
if (root) {
  const groups = root.querySelector('[data-rep-groups]');
  if (groups) renderGroups(groups);
  const spikes = root.querySelector('[data-rep-spikes]');
  if (spikes) renderSpikes(spikes);
  const cmp = root.querySelector('[data-rep-compare]');
  if (cmp) onVisible(cmp, () => getJSON('data/architectures.json').then(a => renderCompare(cmp, a)).catch(e => { cmp.innerHTML = `<p class="note">Could not load architecture data (${esc(e.message)}).</p>`; console.warn(e); }));
  const sur = root.querySelector('[data-rep-surrogate]');
  if (sur) onVisible(sur, () => waitD3().then(d3 => renderSurrogate(sur, d3)).catch(e => console.warn(e)));
}
