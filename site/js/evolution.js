// Main evolution view: 171 trained architectures across 10 outer iterations.
// Plane view: WT2 PPL (log, lower better) vs ListOps proxy accuracy, cumulative Pareto front per iteration.
// Fitness view: composite fitness by iteration with best-so-far.
import { esc, fmt, tip, iterColor, iterLabel, iterShort, displayName, reduceMotion, siteTheme, ICON } from './core.js';

const PPL_CLIP = 200;

export function mountEvolution(root, { archs, meta, onSelect, onIter, onOpen }) {
  const d3 = window.d3;
  const state = { t: 10, view: 'plane', sel: null, playing: false, frontOnly: false, zoom: true };
  const ZOOM = { x: [56.8, 80], y: [30, 56] };
  const byKey = new Map(archs.map(a => [a.key, a]));
  const fronts = meta.fronts.map(f => new Set(f));

  root.innerHTML = `
    <div class="toolbar">
      <div class="seg" role="tablist" aria-label="Chart view">
        <button role="tab" aria-selected="true" data-view="plane">Task plane</button>
        <button role="tab" aria-selected="false" data-view="fitness">Fitness by iteration</button>
      </div>
      <div class="seg" role="group" aria-label="Plane extent" data-zoomseg><button data-zoom="1" aria-pressed="true">Frontier</button><button data-zoom="0" aria-pressed="false">Full range</button></div>
      <label class="switch"><input type="checkbox" data-frontonly> Front only</label>
      <span class="spacer"></span>
    </div>
    <div class="chart" tabindex="0" aria-label="Scatter plot of trained architectures. Use left and right arrow keys to move between points, Enter to open the selected architecture."></div>
    <div class="scrub">
      <button class="iconbtn play" aria-label="Play iterations">${ICON.play}</button>
      <div class="ticks" role="group" aria-label="Search iteration"><div class="fill"></div></div>
    </div>
    <div class="legend"></div>`;
  const chartEl = root.querySelector('.chart');
  const ticksEl = root.querySelector('.ticks');
  const playBtn = root.querySelector('.play');
  const legendEl = root.querySelector('.legend');

  for (let i = 0; i <= 10; i++) {
    const b = document.createElement('button');
    b.className = 'tick'; b.dataset.t = i;
    b.innerHTML = `<i>${iterShort(i)}</i><span>${i === 0 ? 'experts' : 'iter ' + i}</span>`;
    b.setAttribute('aria-label', i === 0 ? 'Show initial experts only' : `Show archive after iteration ${i}`);
    b.addEventListener('click', () => { stop(); setIter(i); });
    ticksEl.appendChild(b);
  }

  // ---------------------------------------------------------------- svg skeleton
  const svg = d3.select(chartEl).append('svg').attr('role', 'img');
  const clip = svg.append('defs').append('clipPath').attr('id', 'evo-clip').append('rect');
  const gGrid = svg.append('g').attr('class', 'grid');
  const gAxX = svg.append('g').attr('class', 'axis');
  const gAxY = svg.append('g').attr('class', 'axis');
  const gDeco = svg.append('g');
  const gFront = svg.append('g').attr('clip-path', 'url(#evo-clip)');
  const gPts = svg.append('g');
  const gLbl = svg.append('g');
  const ring = svg.append('circle').attr('class', 'hover-ring').attr('r', 0);
  const titleX = svg.append('text').attr('class', 'axis-title');
  const titleY = svg.append('text').attr('class', 'axis-title');

  let W = 760, H = 480, M = { l: 52, r: 18, t: 14, b: 46 }, x, y, xy = new Map(), delaunay, ordered = [];
  const isNarrow = () => W < 560;

  function jitterFitness() {
    // deterministic spread inside each iteration column (sorted by fitness, alternating offsets)
    const out = new Map();
    for (let i = 0; i <= 10; i++) {
      const col = archs.filter(a => a.iter === i).sort((a, b) => a.fit - b.fit);
      col.forEach((a, j) => {
        const k = ((j * 7) % col.length) / Math.max(1, col.length - 1);
        out.set(a.key, (k - 0.5) * 0.62);
      });
    }
    return out;
  }
  const jit = jitterFitness();

  function layout() {
    W = Math.max(300, chartEl.clientWidth || 760);
    H = Math.round(Math.min(560, Math.max(320, W * (W < 560 ? 0.95 : 0.66))));
    M = { l: W < 560 ? 42 : 54, r: W < 560 ? 34 : 46, t: 34, b: 46 };
    svg.attr('viewBox', `0 0 ${W} ${H}`).attr('width', W).attr('height', H);
    if (state.view === 'plane') {
      if (state.zoom) {
        x = d3.scaleLog().domain(ZOOM.x).range([M.l, W - M.r - 26]);
        y = d3.scaleLinear().domain(ZOOM.y).range([H - M.b, M.t]);
      } else {
        x = d3.scaleLog().domain([56, PPL_CLIP]).range([M.l, W - M.r - 26]);
        const ys = archs.map(a => a.listops);
        y = d3.scaleLinear().domain([Math.floor(d3.min(ys) - 2), Math.ceil(d3.max(ys) + 2)]).range([H - M.b, M.t]).nice();
      }
    } else {
      x = d3.scaleLinear().domain([-0.6, 10.6]).range([M.l, W - M.r]);
      y = d3.scaleLinear().domain([0.2, 0.92]).range([H - M.b, M.t]);
    }
    clip.attr('x', M.l).attr('y', M.t).attr('width', W - M.r - 26 - M.l).attr('height', H - M.b - M.t);
    xy.clear();
    for (const a of archs) {
      if (state.view === 'plane') {
        const over = !state.zoom && a.wt2 > PPL_CLIP;
        xy.set(a.key, [over ? W - M.r - 8 + ((a.listops * 7) % 9) - 4 : x(a.wt2), y(a.listops)]);
      } else {
        xy.set(a.key, [x(a.iter + jit.get(a.key)), y(a.fit)]);
      }
    }
  }

  function drawAxes() {
    const dur = reduceMotion() ? 0 : 450;
    gAxX.attr('transform', `translate(0,${H - M.b})`);
    gAxY.attr('transform', `translate(${M.l},0)`);
    gDeco.selectAll('*').remove();
    if (state.view === 'plane') {
      const tv = state.zoom ? (isNarrow() ? [57, 60, 65, 70, 80] : [57, 58, 59, 60, 62, 65, 70, 75, 80]) : (isNarrow() ? [60, 80, 100, 150] : [56, 60, 65, 70, 80, 90, 100, 120, 150]);
      gAxX.transition().duration(dur).call(d3.axisBottom(x).tickValues(tv).tickFormat(d3.format('d')).tickSizeOuter(0));
      gAxY.transition().duration(dur).call(d3.axisLeft(y).ticks(6).tickSizeOuter(0));
      gGrid.selectAll('*').remove();
      gGrid.append('g').attr('transform', `translate(0,${H - M.b})`).call(d3.axisBottom(x).tickValues(state.zoom ? [58, 60, 65, 70, 80] : [60, 70, 80, 100, 150, 200]).tickSize(-(H - M.b - M.t)).tickFormat(''));
      gGrid.append('g').attr('transform', `translate(${M.l},0)`).call(d3.axisLeft(y).ticks(6).tickSize(-(W - M.l - M.r)).tickFormat(''));
      titleX.attr('x', M.l).attr('y', H - 8).attr('text-anchor', 'start').text('WT2 PPL (log scale)  ← better');
      titleY.attr('transform', `translate(${M.l - 30},${M.t - 18})`).attr('text-anchor', 'start').text('↑ ListOps acc. (%) · higher is better');
      // overflow band (full range) / outside-view count (frontier)
      const bx = W - M.r - 16;
      if (!state.zoom) {
        gDeco.append('rect').attr('x', bx - 4).attr('y', M.t).attr('width', 24).attr('height', H - M.b - M.t).attr('rx', 6).attr('fill', 'var(--bg-2)');
        gDeco.append('text').attr('class', 'anchor-lbl over-count').attr('x', bx + 8).attr('y', H - M.b + 18).attr('text-anchor', 'middle').text('>200');
      } else {
        gDeco.append('text').attr('class', 'anchor-lbl out-count').attr('x', W - M.r - 26).attr('y', isNarrow() ? M.t + 12 : H - M.b - 8).attr('text-anchor', 'end');
      }
      // Fitness reference (0.500): the SpikingDeltaNet expert, 58.28 PPL / 53.0 % (paper values, see build_data.py).
      gDeco.append('line').attr('class', 'anchor-line').attr('x1', x(58.28)).attr('x2', x(58.28)).attr('y1', M.t).attr('y2', H - M.b);
      gDeco.append('line').attr('class', 'anchor-line').attr('x1', M.l).attr('x2', bx - 8).attr('y1', y(53)).attr('y2', y(53));
      gDeco.append('text').attr('class', 'anchor-lbl halo').attr('x', x(58.28) + 5).attr('y', H - M.b - 6).text(isNarrow() ? 'SpikingDeltaNet' : 'SpikingDeltaNet (fitness 0.500)');
    } else {
      gAxX.transition().duration(dur).call(d3.axisBottom(x).tickValues(d3.range(0, 11)).tickFormat(i => i === 0 ? 'E' : i).tickSizeOuter(0));
      gAxY.transition().duration(dur).call(d3.axisLeft(y).ticks(7).tickSizeOuter(0));
      gGrid.selectAll('*').remove();
      gGrid.append('g').attr('transform', `translate(${M.l},0)`).call(d3.axisLeft(y).ticks(7).tickSize(-(W - M.l - M.r)).tickFormat(''));
      titleX.attr('x', M.l).attr('y', H - 8).attr('text-anchor', 'start').text('Search iteration (E = initial experts)');
      titleY.attr('transform', `translate(${M.l - 30},${M.t - 18})`).attr('text-anchor', 'start').text('↑ Fitness · higher is better');
      gDeco.append('line').attr('class', 'anchor-line').attr('x1', M.l).attr('x2', W - M.r).attr('y1', y(0.5)).attr('y2', y(0.5));
      // label at the left end, above the line, in the gap left of the experts' column
      gDeco.append('text').attr('class', 'anchor-lbl halo').attr('x', M.l + 4).attr('y', y(0.5) - 6).attr('text-anchor', 'start').text(isNarrow() ? '0.500' : '0.500 · SpikingDeltaNet');
    }
  }

  function frontPath(t) {
    const pts = [...fronts[t]].map(k => byKey.get(k)).filter(a => a && a.wt2 <= PPL_CLIP).sort((a, b) => a.wt2 - b.wt2);
    if (!pts.length) return { line: '', area: '' };
    const P = pts.map(a => [x(a.wt2), y(a.listops)]);
    const right = W - M.r - 26, bottom = H - M.b;
    let d = `M${P[0][0]},${P[0][1]}`;
    for (let i = 1; i < P.length; i++) d += `L${P[i][0]},${P[i - 1][1]}L${P[i][0]},${P[i][1]}`;
    d += `L${right},${P[P.length - 1][1]}`;
    return { line: d, area: `M${P[0][0]},${bottom}L` + d.slice(1) + `L${right},${bottom}Z` };
  }

  function bestPath(t) {
    // best-so-far fitness (including experts) by iteration, step after
    let best = -1, pts = [], stars = [];
    for (let i = 0; i <= t; i++) {
      const col = archs.filter(a => a.iter === i);
      const m = d3.max(col, a => a.fit);
      if (m > best) { if (i > 0) stars.push(col.find(a => a.fit === m)); best = m; }
      pts.push([i, best]);
    }
    let d = '';
    pts.forEach(([i, b], j) => {
      const X0 = x(i - 0.45), X1 = x(i + 0.45);
      d += (j ? 'L' : 'M') + `${X0},${y(b)}L${X1},${y(b)}`;
    });
    return { d, stars };
  }

  function drawFront() {
    const dur = reduceMotion() ? 0 : 500;
    gFront.selectAll('*').remove();
    if (state.view === 'plane') {
      const cur = frontPath(state.t);
      if (state.t > 0) gFront.append('path').attr('class', 'front-prev').attr('d', frontPath(state.t - 1).line);
      gFront.append('path').attr('class', 'front-area').attr('d', cur.area);
      const p = gFront.append('path').attr('class', 'front-line').attr('d', cur.line);
      if (dur) { const L = p.node().getTotalLength(); p.attr('stroke-dasharray', `${L} ${L}`).attr('stroke-dashoffset', L).transition().duration(dur).attr('stroke-dashoffset', 0).on('end', () => p.attr('stroke-dasharray', null)); }
    } else {
      const { d } = bestPath(state.t);
      gFront.append('path').attr('class', 'best-line').attr('d', d);
    }
  }

  function inView(a) {
    if (state.view !== 'plane' || !state.zoom) return true;
    return a.wt2 >= ZOOM.x[0] && a.wt2 <= ZOOM.x[1] && a.listops >= ZOOM.y[0] && a.listops <= ZOOM.y[1];
  }
  function visible(a) {
    if (a.iter > state.t) return false;
    if (!inView(a)) return false;
    if (state.frontOnly && state.view === 'plane') return fronts[state.t].has(a.key);
    return true;
  }

  function drawPoints(animate = true) {
    const dur = animate && !reduceMotion() ? 520 : 0;
    const theme = siteTheme();
    const r0 = isNarrow() ? 4 : 5;
    const sel = gPts.selectAll('circle.pt').data(archs, a => a.key);
    sel.enter().append('circle').attr('class', 'pt')
      .attr('cx', a => xy.get(a.key)[0]).attr('cy', a => xy.get(a.key)[1]).attr('r', 0)
      .merge(sel)
      .attr('fill', a => iterColor(a.iter, theme))
      .classed('cur', a => a.iter === state.t && state.t > 0)
      .classed('front', a => state.view === 'plane' && fronts[state.t].has(a.key))
      .classed('sel', a => a.key === state.sel)
      .classed('hide', a => !visible(a))
      .transition().duration(dur)
      .attr('cx', a => xy.get(a.key)[0]).attr('cy', a => xy.get(a.key)[1])
      .attr('r', a => !visible(a) ? 0 : (a.key === state.sel ? r0 + 2.5 : (a.iter === state.t && state.t > 0 ? r0 + 0.5 : r0)));
    gPts.selectAll('circle.pt').filter(a => a.key === state.sel).raise();
    // label: the selected point. Greedy anchor placement: each label tries a ring of
    // candidate positions around its point and takes the cheapest one that stays inside the plot, never overlaps an
    // already placed label and covers as few dots as possible; a leader line is drawn when the label sits away from its dot.
    gLbl.selectAll('*').remove();
    const want = archs.filter(a => visible(a) && a.key === state.sel)
      .sort((p, q) => (q.key === state.sel) - (p.key === state.sel) || (xy.get(p.key)[1] - xy.get(q.key)[1]));
    const obst = archs.filter(visible).map(a => xy.get(a.key));
    const box = { x0: M.l + 2, x1: W - M.r - (state.view === 'plane' ? 28 : 2), y0: M.t + 2, y1: H - M.b - 4 };
    const placed = [];
    // fitness view: the improvement stars (drawn above their dots) and the anchor label are fixed obstacles
    const starKeys = state.view === 'fitness' ? bestPath(state.t).stars.map(a => a.key) : [];
    for (const k of starKeys) { const [sx, sy] = xy.get(k); placed.push({ x: sx - 7, y: sy - 21, w: 14, h: 15 }); }
    if (state.view === 'fitness') placed.push({ x: M.l, y: y(0.5) - 18, w: isNarrow() ? 40 : 150, h: 14 });
    const hit = (r, q) => r.x < q.x + q.w && q.x < r.x + r.w && r.y < q.y + q.h && q.y < r.y + r.h;
    const ANG = [-25, 25, -55, 55, 0, -90, 90, 205, 155, 180, 235, 125];
    const RAD = isNarrow() ? [9, 24, 42, 64, 90] : [10, 28, 50, 76, 108, 140];
    for (const a of want) {
      const [px, py] = xy.get(a.key);
      const t = gLbl.append('text').attr('class', 'elite-lbl' + (a.key === state.sel ? ' sel' : '')).text(displayName(a));
      const w = (t.node().getComputedTextLength?.() || 70) + 4, h = 14;
      let best = null;
      for (const R of RAD) for (const deg of ANG) {
        const th = deg * Math.PI / 180, ax = px + R * Math.cos(th), ay = py + R * Math.sin(th);
        const right = Math.cos(th) >= -0.01;
        const r = { x: right ? ax : ax - w, y: ay - h / 2, w, h };
        if (r.x < box.x0 || r.x + r.w > box.x1 || r.y < box.y0 || r.y + r.h > box.y1) continue;
        if (placed.some(q => hit(r, q))) continue;
        let cover = 0;
        for (const [ox, oy] of obst) if (ox > r.x - 4 && ox < r.x + r.w + 4 && oy > r.y - 4 && oy < r.y + r.h + 4) cover++;
        const cost = R * 0.55 + cover * 40 + (deg > 90 || deg < -90 ? 6 : 0);
        if (!best || cost < best.cost) best = { cost, r, ax, ay, right, R };
      }
      if (!best) { t.remove(); continue; }
      placed.push(best.r);
      t.attr('x', best.right ? best.r.x + 2 : best.r.x + best.r.w - 2).attr('y', best.r.y + h - 3.5).attr('text-anchor', best.right ? 'start' : 'end');
      if (best.R > 12) {
        const lx = best.right ? best.r.x : best.r.x + best.r.w, ly = best.r.y + h / 2;
        const d = Math.hypot(lx - px, ly - py), k = (isNarrow() ? 5 : 6) / d;
        gLbl.insert('path', 'text').attr('class', 'leader').attr('d', `M${px + (lx - px) * k},${py + (ly - py) * k}L${lx},${ly}`);
      }
    }
    if (state.view === 'plane' && !state.zoom) {
      // clipped (> PPL_CLIP) runs are drawn in the band, never dropped: show how many are there
      const over = archs.filter(a => a.iter <= state.t && a.wt2 > PPL_CLIP).length;
      gDeco.select('.over-count').text(over ? `>200 (${over})` : '>200');
    }
    if (state.view === 'plane' && state.zoom) {
      const out = archs.filter(a => a.iter <= state.t && !inView(a)).length;
      gDeco.select('.out-count').text(out ? isNarrow() ? `+${out} outside view` : `${out} more outside this view → Full range` : '');
    }
    // stars for best-so-far improvements (fitness view)
    if (state.view === 'fitness') {
      const { stars } = bestPath(state.t);
      for (const a of stars) {
        const [px, py] = xy.get(a.key);
        gLbl.append('path').attr('d', d3.symbol(d3.symbolStar, 70)()).attr('transform', `translate(${px},${py - 13})`).attr('fill', 'var(--accent)').attr('pointer-events', 'none');
      }
    }
    // hit-testing structure
    ordered = archs.filter(visible);
    delaunay = d3.Delaunay.from(ordered, a => xy.get(a.key)[0], a => xy.get(a.key)[1]);
    svg.attr('aria-label', `${ordered.length} trained architectures shown up to ${iterLabel(state.t).toLowerCase()}.`);
  }

  function render(animate = true) { layout(); drawAxes(); drawFront(); drawPoints(animate); updateTicks(); updateLegend(); }

  function updateTicks() {
    ticksEl.querySelectorAll('.tick').forEach(b => {
      const i = +b.dataset.t;
      b.classList.toggle('on', i === state.t);
      b.classList.toggle('done', i < state.t);
      b.setAttribute('aria-pressed', i === state.t);
    });
    ticksEl.querySelector('.fill').style.width = `${(state.t / 10) * 91}%`;
  }
  function updateLegend() {
    const theme = siteTheme();
    const stops = [1, 3, 5, 7, 10].map(i => iterColor(i, theme)).join(',');
    legendEl.innerHTML = `<span><i class="sw" style="background:${iterColor(0, theme)}"></i>Initial experts</span>
      <span>Iteration 1<i class="ramp" style="background:linear-gradient(90deg,${stops})"></i>10</span>
      ${state.view === 'plane' ? '<span><i class="ln"></i>Cumulative Pareto front</span><span><i class="sw ring"></i>On the front</span><span><i class="ln dash"></i>Front one iteration earlier</span>' : '<span><i class="ln"></i>Best-so-far fitness</span><span>★ improvement</span>'}
      <span><i class="sw cur"></i>Trained in the shown iteration</span>`;
  }

  // ---------------------------------------------------------------- interaction
  function nearest(ev) {
    const [mx, my] = d3.pointer(ev, svg.node());
    const sx = W / svg.node().getBoundingClientRect().width;
    const i = delaunay?.find(mx, my);
    if (i == null || i < 0) return null;
    const a = ordered[i];
    const [px, py] = xy.get(a.key);
    return Math.hypot(px - mx, py - my) < 26 * sx ? a : null;
  }
  function tipHTML(a) {
    return `<b>${esc(displayName(a))}</b><span class="k">${esc(a.key)} · ${iterLabel(a.iter)}</span>
      <table><tr><td class="k">WT2 PPL</td><td>${fmt(a.wt2, 2)}${a.wt2 > PPL_CLIP ? ' (off-scale, drawn in the &gt;200 band)' : ''}</td></tr>
      <tr><td class="k">ListOps acc.</td><td>${fmt(a.listops, 2)}%</td></tr>
      <tr><td class="k">Fitness</td><td>${fmt(a.fit, 3)}</td></tr></table>
      <div class="hint">Click to inspect · double-click to open</div>`;
  }
  svg.on('pointermove', ev => {
    const a = nearest(ev);
    if (!a) { ring.attr('r', 0); tip(null); svg.style('cursor', null); return; }
    const [px, py] = xy.get(a.key);
    ring.attr('cx', px).attr('cy', py).attr('r', (isNarrow() ? 4 : 5) + 5).raise();
    svg.style('cursor', 'pointer');
    tip(tipHTML(a), ev);
  });
  svg.on('pointerleave', () => { ring.attr('r', 0); tip(null); });
  svg.on('click', ev => { const a = nearest(ev); if (a) select(a.key, true); });
  svg.on('dblclick', ev => { const a = nearest(ev); if (a) onOpen?.(a.key); });
  chartEl.addEventListener('keydown', e => {
    if (!['ArrowLeft', 'ArrowRight', 'Enter', 'Home', 'End'].includes(e.key)) return;
    e.preventDefault();
    const list = [...ordered].sort((p, q) => state.view === 'plane' ? p.wt2 - q.wt2 : (p.iter - q.iter) || (p.fit - q.fit));
    let i = list.findIndex(a => a.key === state.sel);
    if (e.key === 'Enter') { if (state.sel) onOpen?.(state.sel); return; }
    if (e.key === 'Home') i = 0; else if (e.key === 'End') i = list.length - 1;
    else i = i < 0 ? 0 : Math.max(0, Math.min(list.length - 1, i + (e.key === 'ArrowRight' ? 1 : -1)));
    const a = list[i];
    if (a) { select(a.key, true); const [px, py] = xy.get(a.key); const r = svg.node().getBoundingClientRect(); tip(tipHTML(a), { clientX: r.left + px * r.width / W, clientY: r.top + py * r.height / H }); }
  });
  chartEl.addEventListener('blur', () => tip(null));

  root.querySelectorAll('[data-view]').forEach(b => b.addEventListener('click', () => {
    state.view = b.dataset.view;
    root.querySelectorAll('[data-view]').forEach(o => o.setAttribute('aria-selected', o === b));
    root.querySelector('[data-zoomseg]').style.display = state.view === 'plane' ? '' : 'none';
    render(true);
  }));
  root.querySelectorAll('[data-zoom]').forEach(b => b.addEventListener('click', () => {
    state.zoom = b.dataset.zoom === '1';
    root.querySelectorAll('[data-zoom]').forEach(o => o.setAttribute('aria-pressed', o === b));
    render(true);
  }));
  root.querySelector('[data-frontonly]').addEventListener('change', e => { state.frontOnly = e.target.checked; drawPoints(true); });

  let timer = null;
  function stop() { state.playing = false; clearTimeout(timer); playBtn.innerHTML = ICON.play; playBtn.setAttribute('aria-label', 'Play iterations'); }
  function step() {
    if (!state.playing) return;
    if (state.t >= 10) { stop(); return; }
    setIter(state.t + 1);
    timer = setTimeout(step, reduceMotion() ? 1800 : 1300);
  }
  playBtn.addEventListener('click', () => {
    if (state.playing) { stop(); return; }
    state.playing = true; playBtn.innerHTML = ICON.pause; playBtn.setAttribute('aria-label', 'Pause');
    if (state.t >= 10) { setIter(0); timer = setTimeout(step, 900); } else step();
  });

  function setIter(t) {
    state.t = t;
    drawFront(); drawPoints(true); updateTicks();
    onIter?.(t);
  }
  function select(key, fromChart) {
    state.sel = key;
    const a = byKey.get(key);
    if (a && a.iter > state.t) { state.t = a.iter; drawFront(); updateTicks(); onIter?.(state.t); }
    if (a && !inView(a)) { state.zoom = false; root.querySelectorAll('[data-zoom]').forEach(o => o.setAttribute('aria-pressed', o.dataset.zoom === '0')); render(false); }
    drawPoints(false);
    onSelect?.(key, fromChart);
  }

  new ResizeObserver(() => { const w = chartEl.clientWidth; if (Math.abs(w - W) > 4) render(false); }).observe(chartEl);
  window.addEventListener('oae-theme', () => { drawPoints(false); updateLegend(); });
  render(false);
  return { setIter: t => { stop(); setIter(t); }, select, get t() { return state.t; }, playIntro() { stop(); setIter(0); state.playing = true; playBtn.innerHTML = ICON.pause; timer = setTimeout(step, 900); } };
}
