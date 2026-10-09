import { esc, fmt, iterColor, iterLabel, displayName, archURL, getJSON, frSortValue, frInfo, tex } from './core.js';

function frCell(a) {
  const f = frInfo(a);
  return `<span${f.note ? ` title="Fire rate from ${esc(f.note)}"` : ''}><em>fire</em>${fmt(f.v, 3)}${f.ep ? `<em style="margin:0 0 0 3px">ep ${f.ep.at}</em>` : ''}</span>`;
}

export function mountExplorer({ archs, root, zooCount }) {
  const q = root.querySelector('[data-q]'), fi = root.querySelector('[data-fi]'), ff = root.querySelector('[data-ff]'), fs = root.querySelector('[data-fs]');
  const list = root.querySelector('[data-list]'), count = root.querySelector('[data-count]'), more = root.querySelector('[data-more]');
  fi.innerHTML = '<option value="">All iterations</option><option value="0">Experts</option>' + Array.from({ length: 10 }, (_, i) => `<option value="${i + 1}">Iteration ${i + 1}</option>`).join('');
  const fams = [...new Set(archs.map(a => a.base).filter(Boolean))].sort();
  ff.innerHTML = '<option value="">All state-update types</option>' + fams.map(f => `<option>${esc(f)}</option>`).join('');
  let showAll = false;

  function render() {
    const s = q.value.trim().toLowerCase();
    let rows = archs.filter(a => {
      if (fi.value !== '' && a.iter !== +fi.value) return false;
      if (ff.value && a.base !== ff.value) return false;
      if (s && !`${displayName(a)} ${a.key} ${a.base || ''} ${a.summary || ''} ${a.slug || ''}`.toLowerCase().includes(s)) return false;
      return true;
    });
    const k = fs.value;
        rows.sort((x, y) => k === 'fit' ? y.fit - x.fit : k === 'wt2' ? x.wt2 - y.wt2 : k === 'listops' ? y.listops - x.listops : k === 'fr' ? frSortValue(x) - frSortValue(y) : (x.iter - y.iter) || (x.sample - y.sample));
    count.textContent = `${rows.length} of ${archs.length}`;
    const lim = showAll || s || fi.value || ff.value ? rows.length : 24;
    more.hidden = rows.length <= lim;
    more.textContent = `Show all ${rows.length}`;
    list.innerHTML = rows.slice(0, lim).map(a => `
      <a class="card" href="${archURL(a.key)}">
        <span class="top"><i class="dot" style="background:${iterColor(a.iter)}"></i>${esc(a.key)}${a.expert ? '<span class="tag plain">expert</span>' : ''}</span>
        <span class="nm">${esc(displayName(a))}</span>
        <span class="sum">${a.summary ? tex(a.summary) : esc(iterLabel(a.iter))}</span>
        <span class="mx"><span><em>fit</em>${fmt(a.fit, 3)}</span><span><em>WT2</em>${fmt(a.wt2, 1)}</span><span><em>ListOps</em>${fmt(a.listops, 1)}</span>${k === 'fr' ? frCell(a) : ''}</span>
      </a>`).join('') || '<p class="muted">No architecture matches.</p>';
  }
  [q, fi, ff, fs].forEach(e => e.addEventListener('input', render));
  more.addEventListener('click', () => { showAll = true; render(); });
  render();

  // zoo: the 100 KB table is fetched only when the reader opens it
  const det = root.querySelector('[data-zoo]');
  if (zooCount === 0) return;
  det.hidden = false;
  if (zooCount) root.querySelector('[data-zoo-count]').textContent = `${zooCount} programs`;
  let loaded = false;
  det.addEventListener('toggle', () => {
    if (!det.open || loaded) return;
    loaded = true;
    const t = root.querySelector('[data-zoo-table]');
    t.innerHTML = '<tbody><tr><td class="muted">Loading…</td></tr></tbody>';
    getJSON('data/zoo.json').then(zoo => {
      const label = { baseline: 'baseline', asi_arch_discovered: 'ASI-Arch', fullspike_rewrite: 'fully-spiking rewrite' };
      root.querySelector('[data-zoo-count]').textContent = `${zoo.length} programs`;
      t.innerHTML = `<thead><tr><th>Program</th><th>Set</th><th>Family</th><th>WT2 PPL</th><th>ListOps</th><th>Fire rate WT2</th><th>Fire rate ListOps</th></tr></thead><tbody>` +
        zoo.map(z => `<tr><td><a href="arch.html?zoo=${encodeURIComponent(z.key)}">${esc(z.name || z.key)}</a> <span class="muted mono small">${esc(z.key)}</span></td><td>${esc(label[z.subset] || z.subset || '')}</td><td style="text-align:left">${esc(z.base || '')}</td><td>${fmt(z.wt2, 2)}</td><td>${fmt(z.listops, 2)}</td><td>${fmt(z.fr?.wt2, 3)}</td><td>${fmt(z.fr?.listops, 3)}</td></tr>`).join('') + '</tbody>';
    }).catch(() => { t.innerHTML = '<tbody><tr><td class="muted">Could not load the zoo table.</td></tr></tbody>'; });
  });
}
