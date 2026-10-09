# OpenArchEvo site — design notes

## Reference pages and what we took from them

Live fetching was restricted during this build. Only the ASI-Arch and Nerfies pages could be fetched as raw HTML. The other notes come from the previous research round (Demo_Code/docs/DESIGN_RESEARCH.md) and from general knowledge of these pages.

| Reference | What makes it work | What we took |
|---|---|---|
| ASI-Arch gallery (gair-nlp.github.io/ASI-Arch) | One data-first entry point (a sortable list plus a relationship tree). Generic grey UI with many panels and card shadows. Without its data it renders "0 networks", so it depends entirely on runtime data. | A sortable explorer of every architecture. We avoided its look: no stacked dashboards and no tree that implies verified parentage. |
| Nerfies template | Centered title, an author line, one row of pill buttons (Paper / arXiv / Code / Data), then the abstract, a figure and BibTeX. Readers know where everything is. | The hero order (title → authors → link buttons) and BibTeX at the end with a copy button. |
| distill.pub articles | A narrow text column with wide figures that break out of it. Captions sit right next to the evidence. Interactive figures have a clear default state and few controls. Muted axes and one accent colour. | A text measure of about 66ch, figures and charts at full width, notes directly under each chart, and one accent colour (terracotta) for "the thing to look at" (front, best-so-far, evolved). |
| AlphaEvolve blog | The method is told as a loop diagram, then results as a few large numbers, then examples. Lots of whitespace. | The story order: pipeline figure → two loops as two short numbered lists → a key-number strip → one strong interactive centrepiece. |

## Principles used

- **One centrepiece.** The evolution chart is the largest and most interactive element. Everything else is quieter: tables, cards and a schematic.
- **Fewer boxes.** Sections are separated by whitespace and hairlines. Panels are used only where controls and a chart belong together.
- **Honest labels next to the data.** Every chart has a caption that states its provenance: reported WT2 PPL with the split unresolved, ListOps proxy, the paper-formula fitness, and the predicted score on the historical mapping. Paper results (Table 1/2) carry a "full-scale retraining · not search-proxy numbers" badge. The island animation carries "Illustrative schematic · not a replay".
- **Colour by job.** The iteration is ordinal, so a single sequential ramp (peach → terracotta → plum in light mode, reversed in dark mode), with experts in grey. The terracotta accent is reserved for the front, best-so-far and evolved mechanisms. The lavender and plum family is for spikes and secondary encodings. Iteration is also visible from the scrubber position and the ring on the current iteration, so colour is never the only cue.
- **Palette.** Warm white #fffdfa, terracotta #b54834, lavender #eee6f7, plum #30243b / #241d30, peach #f0b79d. The neutrals are derived with a slight plum bias. The site dark theme uses the attribute `data-site-theme` on `<html>`, deliberately different from archviz's `data-theme`, so block diagrams keep their own Day/Night toggle; the toggle starts from the site theme (see DESIGN_REVIEW.md).
- **Type.** Manrope 800 for display, DM Sans for body text, JetBrains Mono for code and numbers in tables.
- **Accessibility.** Keyboard: the charts are focusable, ←/→ steps through points and Enter opens the architecture page. Diagram nodes are buttons, and `⤢` expand buttons respond to Enter and Space. Focus rings are visible, `prefers-reduced-motion` disables transitions, and `prefers-color-scheme` sets the default theme.
- **Performance.** No build step. d3 7.9.0 and highlight.js 11.9.0 come from cdnjs. Data is lazy: `architectures.json` (~110 KB) on load, each round file (≤ 60 KB) when its iteration is opened, and per-architecture rationale, code and diagram files only on the detail page.

## Structure

- `index.html` contains: hero (links, key numbers) → method (pipeline figure, two loops, three-view representation) → evolution view (task plane with frontier and full range, fitness by iteration, scrubber and play) → inner loop per iteration (accepted order, or shortlist score × novelty) → Algorithm-2 schematic → elites, paper Table 1 and ablations → explorer and ASI-Arch zoo → BibTeX.
- `arch.html?arch=<key>`: the architecture page. `#<key>` and `#arch=<key>` also work, and `index.html?arch=` and `index.html#arch=` redirect to it. Zoo programs use `arch.html?zoo=<key>`.
- `js/`: `core.js` (theme, data, colour, tooltip, lightbox, safe markdown), `evolution.js`, `innerloop.js`, `islands.js`, `explorer.js`, `home.js`, `arch.js`, `paper.js` (paper tables transcribed from the LaTeX).
- `css/site.css` is hand-written. `css/archviz.css` is generated from `archviz/render.py`'s `CSS` by the data script.

## Rebuild

```sh
python3 site/tools/build_data.py            # re-assemble site/data from all sources (idempotent)
python3 site/tools/build_data.py --figures  # also re-render paper figures (pdftoppm + Pillow)
python3 -m http.server -d site 8000         # preview
```

The script wipes and rebuilds `site/data/`. It picks up any archviz spec or SVG and the zoo index that exist at run time. Source locations can be overridden with `OAE_WORK`, `OAE_PKG`, `OAE_ARCHVIZ`, `OAE_ZOO_JSON`, `OAE_ZOO_DIR` and `OAE_PAPER`.
