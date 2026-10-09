// Gallery: every trained architecture as a block-diagram thumbnail. Filters and sort live in the URL so a view can be shared.
import { PAPER, ICON, LOGO, initTheme, getJSON, esc, fmt, iterColor, iterLabel, displayName, archURL, siteTheme, frSortValue, frInfo } from './core.js';

initTheme();
document.querySelector('[data-logo]').insertAdjacentHTML('afterbegin', LOGO);
const cl = document.querySelector('[data-code-link]'); cl.href = PAPER.code; cl.innerHTML = ICON.github;

const grid = document.querySelector('[data-grid]');
const gi = document.querySelector('[data-gi]'), gs = document.querySelector('[data-gs]'), gc = document.querySelector('[data-gc]');
const thumb = key => `data/thumbs/${key}${siteTheme() === 'dark' ? '.d' : ''}.svg`;

getJSON('data/architectures.json').then(archs => {
  gi.innerHTML = '<option value="">All iterations</option><option value="0">Experts</option>' + Array.from({ length: 10 }, (_, i) => `<option value="${i + 1}">Iteration ${i + 1}</option>`).join('');
  const q = new URLSearchParams(location.search);
  if (q.has('iter')) gi.value = q.get('iter');
  if (q.has('sort')) gs.value = q.get('sort');

  function render() {
    let rows = archs.filter(a => gi.value === '' || a.iter === +gi.value);
    const k = gs.value;
    rows.sort((x, y) => k === 'fit' ? y.fit - x.fit : k === 'wt2' ? x.wt2 - y.wt2 : k === 'listops' ? y.listops - x.listops : k === 'fr' ? frSortValue(x) - frSortValue(y) : (x.iter - y.iter) || (x.sample - y.sample));
    gc.textContent = `${rows.length} of ${archs.length}`;
    const metric = a => k === 'wt2' ? `<b>${fmt(a.wt2, 1)}</b> PPL` : k === 'listops' ? `<b>${fmt(a.listops, 1)}</b>% ListOps` : k === 'fr' ? (f => `<b>${fmt(f.v, 3)}</b> fire rate${f.ep ? ` · epoch ${f.ep.at}` : ''}`)(frInfo(a)) : `<b>${fmt(a.fit, 3)}</b> fitness`;
    grid.innerHTML = rows.map((a, i) => `
      <a class="g-card" href="${archURL(a.key)}">
        <span class="g-thumb">${a.diagram ? `<img src="${thumb(a.key)}" data-key="${esc(a.key)}" alt="" loading="lazy" decoding="async" ${i < 6 ? 'fetchpriority="high"' : ''}>` : '<span class="muted small">diagram in progress</span>'}</span>
        <span class="g-body">
          <span class="g-top"><i class="dot" style="background:${iterColor(a.iter)}"></i>${esc(a.key)}${a.expert ? '<span class="tag plain">expert</span>' : ''}</span>
          <span class="g-nm">${esc(displayName(a))}</span>
          <span class="g-mx"><span>${metric(a)}</span></span>
        </span>
      </a>`).join('') || '<p class="muted">No architecture matches these filters.</p>';
    const p = new URLSearchParams();
    if (gi.value) p.set('iter', gi.value);
    if (gs.value !== 'fit') p.set('sort', gs.value);
    history.replaceState(null, '', p.toString() ? `?${p}` : location.pathname);
  }
  [gi, gs].forEach(e => e.addEventListener('change', render));
  window.addEventListener('oae-theme', () => grid.querySelectorAll('img[data-key]').forEach(im => { im.src = thumb(im.dataset.key); }));
  render();
}).catch(err => { console.error(err); grid.innerHTML = `<p class="note">Could not load the architecture list (${esc(err.message)}). Serve the site over HTTP.</p>`; });
