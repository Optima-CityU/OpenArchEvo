// Per-iteration inner loop: every accepted candidate's predicted score in accepted order, best-so-far and the trained picks.
import { esc, fmt, tip, getJSON, iterColor, displayName, siteTheme, reduceMotion } from './core.js';

export function mountInnerLoop(root, { meta, archs, onSelect, onOpen }) {
  const d3 = window.d3;
  const byKey = new Map(archs.map(a => [a.key, a]));
  const state = { it: 1, data: null };
  root.innerHTML = `
    <div class="toolbar">
      <div class="iter-pills" role="group" aria-label="Choose iteration"></div>
    </div>
    <div class="il-note"></div>
    <div class="chart" tabindex="0" aria-label="Inner-loop scatter. Arrow keys step through trained candidates."></div>
    <div class="legend"></div>
    <dl class="kv il-stats" style="margin-top:14px"></dl>`;
  const pills = root.querySelector('.iter-pills');
  const chartEl = root.querySelector('.chart');
  const noteEl = root.querySelector('.il-note');
  const legendEl = root.querySelector('.legend');
  const statsEl = root.querySelector('.il-stats');
  meta.rounds.forEach(r => {
    const b = document.createElement('button');
    b.textContent = r.iter; b.dataset.it = r.iter;
    b.setAttribute('aria-label', `Iteration ${r.iter}${r.coverage !== 'full_saved_pool' ? ' (partial pool)' : ''}`);
    if (r.coverage !== 'full_saved_pool') b.classList.add('partial');
    b.addEventListener('click', () => show(r.iter));
    pills.appendChild(b);
  });

  const svg = d3.select(chartEl).append('svg');
  let W = 760, H = 380, pts = [], delaunay, xy = [], evalPts = [];

  async function show(it) {
    state.it = it;
    pills.querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed', +b.dataset.it === it));
    const rm = meta.rounds[it - 1];
    noteEl.innerHTML = '<p class="note">Loading…</p>';
    const R = await getJSON(`data/rounds/${rm.id}.json`);
    if (state.it !== it) return;
    const c = R.cols;
    pts = c.s.map((s, i) => ({ s, f: c.f[i], w: c.w[i], l: c.l[i], isl: c.i[i], b: c.b[i], e: c.e[i] || null, k: c.k[i], n: c.n[i] }));
    state.data = R;
    const partial = R.coverage !== 'full_saved_pool';
    noteEl.innerHTML = partial
      ? `<div class="callout"><span><b>Incomplete pool.</b> Only ${R.n} of the roughly 1,080 accepted candidates of iteration ${it} were saved, so no best-so-far line is drawn.</span></div>`
      : '';
    statsEl.innerHTML = `
      <dt>Candidates shown</dt><dd>${R.n.toLocaleString()}${partial ? ' (partial pool)' : ''}</dd>
      <dt>Trained on both proxy tasks</dt><dd>${R.evaluated}</dd>
      <dt>Highest predicted score</dt><dd>${fmt(R.best_pred, 3)}</dd>`;
    draw();
  }

  function draw() {
    const theme = siteTheme();
    W = Math.max(300, chartEl.clientWidth || 760);
    H = Math.round(Math.min(420, Math.max(280, W * (W < 560 ? 0.8 : 0.42))));
    const M = { l: W < 560 ? 40 : 52, r: 14, t: 30, b: 42 };
    svg.attr('viewBox', `0 0 ${W} ${H}`).attr('width', W).attr('height', H);
    svg.selectAll('*').remove();
    const data = pts.filter(p => p.f != null);
    const x = d3.scaleLinear().domain([0, Math.max(1080, d3.max(data, p => p.s))]).nice().range([M.l, W - M.r]);
    const y = d3.scaleLinear().domain(d3.extent(data, p => p.f)).nice().range([H - M.b, M.t]);
    svg.append('g').attr('class', 'grid').attr('transform', `translate(${M.l},0)`).call(d3.axisLeft(y).ticks(5).tickSize(-(W - M.l - M.r)).tickFormat(''));
    svg.append('g').attr('class', 'axis').attr('transform', `translate(0,${H - M.b})`).call(d3.axisBottom(x).ticks(W < 560 ? 4 : 8).tickSizeOuter(0));
    svg.append('g').attr('class', 'axis').attr('transform', `translate(${M.l},0)`).call(d3.axisLeft(y).ticks(5).tickSizeOuter(0));
    svg.append('text').attr('class', 'axis-title').attr('x', M.l).attr('y', H - 6)
      .text(W < 560 ? 'Accepted order' : 'Accepted order (sample number)');
    svg.append('text').attr('class', 'axis-title').attr('transform', `translate(${M.l - 28},${M.t - 16})`).attr('text-anchor', 'start').text('↑ Surrogate-predicted score');
    const col = iterColor(state.it, theme);
    const g = svg.append('g');
    // all accepted
    {
      g.selectAll('circle.il-dot').data(data.filter(p => !p.e)).join('circle').attr('class', 'il-dot')
        .attr('cx', p => x(p.s)).attr('cy', p => y(p.f)).attr('r', W < 560 ? 1.6 : 2)
        .attr('fill', theme === 'dark' ? '#8f82a0' : '#9a8daa').attr('opacity', data.length > 200 ? .45 : .8);
      // best-so-far (complete pools only)
      const best = data.filter(p => p.b != null);
      if (best.length) {
        const line = d3.line().x(p => x(p.s)).y(p => y(p.b)).curve(d3.curveStepAfter);
        const path = g.append('path').attr('class', 'best-line').attr('d', line(best));
        // Animate only when the iteration (or view) changes, never on a resize/theme redraw.
        const key = String(state.it);
        if (!reduceMotion() && state.animKey !== key) { const L = path.node().getTotalLength(); path.attr('stroke-dasharray', `${L} ${L}`).attr('stroke-dashoffset', L).transition().duration(900).attr('stroke-dashoffset', 0).on('end', () => path.attr('stroke-dasharray', null)); }
        state.animKey = key;
      }
    }
    const X = p => x(p.s);
    // trained
    evalPts = data.filter(p => p.e);
    g.selectAll('circle.il-sel').data(evalPts).join('circle').attr('class', 'il-sel')
      .attr('cx', X).attr('cy', p => y(p.f)).attr('r', W < 560 ? 5 : 6).attr('fill', col);
    xy = data.map(p => [X(p), y(p.f)]);
    delaunay = d3.Delaunay.from(xy);
    pts._data = data;
    legendEl.innerHTML = `<span><i class="sw" style="background:#9a8daa"></i>Candidate (surrogate prediction)</span>
      ${data.some(p => p.b != null) ? '<span><i class="ln"></i>Best predicted so far</span>' : ''}
      <span><i class="sw" style="background:${col}"></i>Trained (click for the real result)</span>`;
    const ring = svg.append('circle').attr('class', 'hover-ring').attr('r', 0);
    svg.on('pointermove', ev => {
      const p = near(ev);
      if (!p) { ring.attr('r', 0); tip(null); svg.style('cursor', null); return; }
      const i = data.indexOf(p);
      ring.attr('cx', xy[i][0]).attr('cy', xy[i][1]).attr('r', p.e ? 10 : 6);
      svg.style('cursor', p.e ? 'pointer' : 'default');
      tip(tipHTML(p), ev);
    }).on('pointerleave', () => { ring.attr('r', 0); tip(null); })
      .on('click', ev => { const p = near(ev); if (p?.e) onSelect?.(p.e); })
      .on('dblclick', ev => { const p = near(ev); if (p?.e) onOpen?.(p.e); });
    function near(ev) {
      const [mx, my] = d3.pointer(ev, svg.node());
      const i = delaunay.find(mx, my);
      if (i < 0) return null;
      // prefer trained points within a generous radius
      let best = null, bd = 18;
      for (const p of evalPts) { const j = data.indexOf(p); const d = Math.hypot(xy[j][0] - mx, xy[j][1] - my); if (d < bd) { bd = d; best = p; } }
      if (best) return best;
      return Math.hypot(xy[i][0] - mx, xy[i][1] - my) < 10 ? data[i] : null;
    }
  }

  function tipHTML(p) {
    const a = p.e ? byKey.get(p.e) : null;
    let h = `<b>${a ? esc(displayName(a)) : 'Accepted candidate'}</b><span class="k">sample ${p.s} · island ${p.isl ?? 'n/a'}</span>
      <table><tr><td class="k">Predicted score</td><td>${fmt(p.f, 3)}</td></tr>
      <tr><td class="k">Predicted WT2 PPL</td><td>${fmt(p.w, 2)}</td></tr>
      <tr><td class="k">Predicted ListOps</td><td>${fmt(p.l, 2)}%</td></tr>
`;
    if (a) h += `<tr><td colspan="2" class="k" style="padding-top:6px">Real training result</td></tr>
      <tr><td class="k">WT2 PPL</td><td>${fmt(a.wt2, 2)}</td></tr>
      <tr><td class="k">ListOps acc.</td><td>${fmt(a.listops, 2)}%</td></tr>
      <tr><td class="k">Fitness</td><td>${fmt(a.fit, 3)}</td></tr>`;
    h += '</table>';
    if (a) h += '<div class="hint">Click to inspect · double-click to open</div>';
    return h;
  }

  chartEl.addEventListener('keydown', e => {
    if (!['ArrowLeft', 'ArrowRight', 'Enter'].includes(e.key) || !evalPts.length) return;
    e.preventDefault();
    const data = pts._data;
    state.kb = state.kb == null ? 0 : (state.kb + (e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0) + evalPts.length) % evalPts.length;
    const p = evalPts[state.kb];
    if (e.key === 'Enter') { onOpen?.(p.e); return; }
    const j = data.indexOf(p); const r = chartEl.querySelector('svg').getBoundingClientRect();
    tip(tipHTML(p), { clientX: r.left + xy[j][0] * r.width / W, clientY: r.top + xy[j][1] * r.height / H });
    onSelect?.(p.e);
  });
  chartEl.addEventListener('blur', () => tip(null));

  new ResizeObserver(() => { if (state.data && Math.abs(chartEl.clientWidth - W) > 4) draw(); }).observe(chartEl);
  window.addEventListener('oae-theme', () => state.data && draw());
  return { show, get it() { return state.it; } };
}
