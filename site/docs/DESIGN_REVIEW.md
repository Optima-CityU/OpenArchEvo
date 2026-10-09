# OpenArchEvo site: design review

Scope: the presentation layer only (structure, CSS, layout, typography, interaction, copy tone). Data semantics and `tools/build_data.py` are out of scope.
Method: the site was served locally and captured with Playwright (Chromium) at 1440 px and 400 px in light and dark: home page, `arch.html?arch=i09-s0557`, `arch.html?arch=i00-comba`, the inner-loop view for iterations 3 and 5, and the core-detail overlay. The ASI-Arch page could not be fetched live in this session (the fetch permission was not granted), so the comparison relies on the earlier capture summarised in `DESIGN_NOTES.md`.

## Baseline verdict

The draft was solid in its parts and generic as a whole. Palette, type pairing and the honesty of the captions were already good. It felt "ugly" for these reasons: there was no focal point, every section had the same weight and the same box, and the centrepiece chart was hard to read. ASI-Arch leads with its data. Our draft leads with two columns of prose.

## Findings, ranked by impact

| # | Area | Problem | Fix |
|---|------|---------|-----|
| 1 | Hierarchy / hero | The 3-line, 60-character paper title is the h1. There are two text columns, a "Below: ..." paragraph that restates the nav, 5 CTA buttons on two rows, and the brand name repeated as a kicker. The page has no single message. | Use a one-line claim as the h1, one sentence of lede, and one row of 4 actions. Keep the full paper title as a quiet citation line. Add a soft background glow. Put the evolution chart directly under the hero so the first scroll reaches real data. |
| 2 | Centrepiece chart | The Pareto "area" tinted almost the whole plot pink, because the dominated region is most of the plane. The newest iterations (dark plum, plus a radius bump and an ink ring) read as black blobs. The front points themselves were not marked. The rotated y title collided with ticks. The scrubber repeated each label twice ("1" plus "iter 1"). | Remove the tint and ring the front points in terracotta. Use a ramp that ends in violet-plum, not ink. Mark the current iteration with a thin halo instead of a size jump. Set the y title horizontally above the axis, use a calmer grid and a slimmer scrubber. |
| 3 | Empty states | The "Selected architecture" panel started empty, so a third of the centrepiece was a paragraph of instructions. | Pre-select the best architecture so the panel shows a diagram, metrics and an "Open" button from first paint. |
| 4 | Box fatigue | Panel, figure frame, equation box, 3 cards, table frame, 24 explorer cards and a cite box all had the same 1 px border and 14 px radius. Section rhythm was uniform. | Use tinted full-bleed bands for the two interactive sections, show the explorer as a hairline list instead of a card grid, use frameless equations, and keep boxes only where controls and chart belong together. |
| 5 | Architecture page | The diagram (the reason to visit) was held to 1100 px and drawn at about 45 % scale, so its labels were about 6 px. The metric strip was as loud as the title. The code column was shorter than the rationale column and scrolled away. Long provenance strings overflowed: `i00-comba` had 55 px of horizontal page scroll at 1440 and 305 px at 400. In a dark site the Day diagram was a glaring white slab. | Let the diagram break out to 1360 px. Make the metrics a quiet inline strip. Make the code panel sticky beside the rationale. Stack the provenance rows and wrap them. Make the diagram theme follow the site theme (the Day/Night toggle still works). |
| 6 | Affordance | Clicking a dot, expanding the core and linking code to the diagram were explained only in prose hints. "⤢ Core state update" looked like any other small button. | Use an accent "Open core state update" button. The cursor and hover ring already exist. Add a short hint row with key icons, and a "swipe to pan" cue on narrow diagrams. |
| 7 | Elite showcase | The diagram thumbnails were about 330 px wide, so they were illegible decoration. | Show each elite as a full-width row (diagram about 640 px plus text) so the evolved blocks are readable. |
| 8 | Inner loop | The ten pills wrapped to two rows at 400 px. The "pool incomplete" callout was a heavy pink slab. The stats were key/value rows stretched across 1100 px, which forced long eye travel. | Fit the pills in one row, make the callout quiet with a rule on the left, and show the stats as a four-up strip. |
| 9 | Typography | The scale was reasonable, but the h1 was 3.6 rem across 3 lines and eyebrows competed with the h2s. The ListOps/WT2 numbers in cards mixed mono and sans. | Set the h1 to about 3.4 rem on 2 lines with tighter tracking, and keep tabular numbers in the body font. |
| 10 | Dark mode | Mostly good. Issues: white figure frames (acceptable for the paper raster figures), white diagram thumbnails, and the light-only `#fff` frame. | Theme the diagram thumbnails and keep the paper figures on a soft "paper" card. |
| 11 | Mobile | No horizontal page scroll except the `i00-comba` overflow above. The nav is hidden, which is fine. The inner-loop y-axis title overlapped the ticks. The stats grid left an orphan. | Covered by the fixes in rows 2, 5 and 8. |
| 12 | Accessibility | Focus rings exist, charts are keyboard-steppable and reduced motion is honoured. Gaps: the scrubber uses `aria-pressed` on 11 buttons with no group label text, and the colour-only iteration encoding (mitigated by the ring and scrubber). | Keep the existing support. The `prefers-reduced-motion` guard now also covers the hero glow and card hover lift. |
| 13 | Performance / first paint | Fine: no build, deferred d3, lazy round files. The three Google font families (12 weights) are the heaviest blocking request. The pipeline raster loads lazily. | Trim the font weights to those actually used. |

## What we keep from ASI-Arch, and where we go further

ASI-Arch's strength is one data-first entry point. We now open on the evolution plane as well. We go further in three ways: every point links to a page with a real block diagram that is wired to its source lines, the provenance captions are honest, and there is no dashboard clutter.

## Remaining weaknesses (after the implementation rounds)

Three Playwright rounds were run after the changes (1440 px and 400 px, light and dark, arch pages, inner loop for iterations 3 and 5, core overlay, and an interaction smoke test: play, view and zoom toggles, novelty mode, theme switch, search, focus toggle, Night, Enlarge, node-to-code jump). There were no console errors and no horizontal page scroll at either width.

What is still not as good as it could be:

- **Elite labels in the frontier view** still sit close together at the top-left, because those four points are genuinely clustered. On narrow screens only the selected point is labelled. A proper fix is a force- or anchor-based label layout with leader lines into empty space.
- **Algorithm 2 schematic.** Step 1 leaves a lot of empty stage. The schematic would benefit from a redesign: a smaller stage, or the stage below the story on wide screens.
- **Diagram text on phones.** The block diagram is about 680 px wide and pans inside its container. The labels are small at 400 px, so "Enlarge" (now full-screen on phones) is the real reading mode. The `hover` hint is meaningless on touch devices.
- **The inner-loop best-so-far line re-animates** whenever the chart redraws on a width change (for example, a scrollbar appearing). This is harmless but visible. Animate only on the first draw or on an iteration change.
- **The paper raster figures** (pipeline, three-view) stay on a white "paper" card in dark mode. They are bitmaps, so they cannot be themed without SVG sources.
- **The caveat paragraph under the evolution chart** is long and dense. Its content is owned by the data-accuracy review, so it was only restyled, not shortened. A collapsible "About these numbers" would read better.
- **There is no mobile section nav.** The top nav is hidden below 860 px.

