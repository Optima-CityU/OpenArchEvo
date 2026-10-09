#!/usr/bin/env python3
"""Assemble the static website data under site/data/ from the OpenArchEvo sources.

Re-runnable: every run rebuilds site/data/ from scratch, so diagrams/specs that appear
later (archviz) and the zoo index (zoo_data.json) are picked up automatically.

    python3 site/tools/build_data.py            # data only
    python3 site/tools/build_data.py --figures  # also re-render paper figures (needs pdftoppm + Pillow)

Paths can be overridden with environment variables (see SRC below). Nothing here retrains,
rescores or invents values: every number is copied from a logged/derived source field and
keeps the provenance label of that field.
"""
import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

SITE = Path(__file__).resolve().parents[1]
# Private research workspace (not in this repo): curated delivery, archviz specs, paper source.
WORK = Path(os.environ.get("OAE_WORK", Path(__file__).resolve().parents[4]))
SRC = {
    "pkg": Path(os.environ.get("OAE_PKG", WORK / "Source_Code/OpenArchEvo")),
    "archviz": Path(os.environ.get("OAE_ARCHVIZ", WORK / "archviz")),
    "zoo_json": Path(os.environ.get("OAE_ZOO_JSON", WORK / "repo/zoo_data.json")),
    "zoo_dir": Path(os.environ.get("OAE_ZOO_DIR", WORK / "repo/OpenArchEvo/zoo")),
    "paper": Path(os.environ.get("OAE_PAPER", WORK / "arXiv_Submission/arXiv_NAS4SNN_source")),
    # paper Fig. 4 (search trajectory), digitized from the vector PDF: drawn as the hero curve
    "fig4": Path(os.environ.get("OAE_FIG4", WORK / "Demo_Code/site/data/figure4.json")),
}
OUT = SITE / "data"
BASE_ALIAS = {"Mamba2/SSD": "Mamba2 (SSD)", "S4D (SS4D)": "S4D", "S4D (SS4D) + GSA": "GSA + S4D"}
# Paper overrides. The paper is the primary source: where the archive disagrees with it, the paper value is used
# everywhere on the site (explorer, gallery, architecture pages, evolution chart, best-so-far, fronts) and fitness
# is recomputed from it.
#   i00-deltanet: the paper's SpikingDeltaNet reference is 58.28 WT2 PPL / 53.0% ListOps, i.e. fitness 0.500
#                 (App. D.2, Fig. 4); the archive stored 57.53 (a test-split value).
#   i07-s0884:    LoopMem, 57.21 WT2 PPL as in the paper's search records (same as the archive; pinned explicitly).
PAPER_OVERRIDES = {"i00-deltanet": {"wt2": 58.28}, "i07-s0884": {"wt2": 57.21}}
# Summaries that must follow the paper wording (Sec. 4.3, Fig. 6 caption, App. F).
SUMMARY_OVERRIDES = {
    "i07-s0884": "Feeds a normalized summary of its recurrent state back to the forget gate (state-norm feedback) "
                 "and adds penalties on the mean spike activity of its token-mixing and channel-mixing branches "
                 "to the training loss.",
}

warnings = []


def warn(msg):
    warnings.append(msg)
    print("WARN", msg, file=sys.stderr)


def jl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def dump(obj, path, pretty=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if pretty:
            json.dump(obj, f, ensure_ascii=False, indent=1)
        else:
            json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))


def r4(x, n=4):
    if x is None:
        return None
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return round(float(x), n)


def fitness(p, a):
    """Paper Appendix D.2 fitness (copy of tools/fitness.py)."""
    rw = (58.28 - p) / (58.28 * (0.02 if p <= 58.28 else 0.25))
    rl = (a - 53.0) / (53.0 * (0.02 if a >= 53.0 else 0.70))

    def phi(r):
        z = (3.0 if r >= 0 else math.log(3.0)) * max(-1.0, min(1.0, r))
        return 1.0 / (1.0 + math.exp(-z))
    return (phi(rw) + phi(rl)) / 2


def range_ok(code, n_lines):
    """True when every 'L12-L15, L20' range lies inside a file of n_lines lines."""
    rs = re.findall(r"L?(\d+)(?:\s*-\s*L?(\d+))?", str(code))
    return bool(rs) and all(1 <= int(a) <= int(b or a) <= n_lines for a, b in rs)


def pretty_slug(s):
    s = re.sub(r"^(spiking_?fla_?|SpikingFLABlock_?)", "", s, flags=re.I)
    s = s.replace("_", " ").strip()
    return s[:1].upper() + s[1:] if s else s


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true", help="re-render paper figures into site/assets")
    args = ap.parse_args()

    pkg = SRC["pkg"]
    runs = pkg / "runs/main-cluster"
    if OUT.exists():
        shutil.rmtree(OUT)
    for d in ["code", "arch", "rounds", "diagrams", "thumbs", "zoo"]:
        (OUT / d).mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- base tables
    index = json.load(open(SRC["archviz"] / "arch_index.json"))
    by_aid = {}
    for row in index:
        if row["architecture_id"] in by_aid:
            warn(f"duplicate architecture_id in arch_index: {row['key']}")
        by_aid[row["architecture_id"]] = row["key"]
    web = json.load(open(pkg / "website/architectures.json"))["records"]
    web_by_aid = {r["architecture_id"]: r for r in web}
    rec_to_key = {r["record_id"]: by_aid.get(r["architecture_id"]) for r in web}
    missing = [r["record_id"] for r in web if r["architecture_id"] not in by_aid]
    if missing:
        warn(f"{len(missing)} archive records without arch_index key")

    zoo_rows = {}
    if SRC["zoo_json"].exists():
        z = json.load(open(SRC["zoo_json"]))
        zoo_rows = {r["key"]: r for r in z.get("rows", [])}
    else:
        warn("zoo_data.json missing: fire rates parsed from training logs, no base family, no ASI zoo")

    # Fire rates straight from the training logs: used as fallback when zoo_data lacks a value, and to
    # cross-check zoo_data (its value must equal the last logged epoch of one of the logs) and count how many
    # distinct logs exist (some experts have 2-3 WT2 logs with different last-epoch fire rates).
    fire_fallback, fire_logs, log_best = {}, defaultdict(dict), defaultdict(dict)
    obs = list(jl(runs / "training_log_observations.jsonl"))
    lines = defaultdict(list)
    for d in jl(runs / "training_epoch_lines.jsonl"):
        lines[d["evaluation_log_id"]].append(d)
    for o in obs:
        ls = sorted(lines.get(o["evaluation_id"], []), key=lambda d: d.get("line_number") or 0)
        frs = [float(m.group(1)) for d in ls
               for m in [re.search(r"Fire Rate:\s*([0-9.]+)", d["logged_line"])] if m]
        bests = [float(m.group(1)) for d in ls for m in [re.search(r"Best Val PPL:\s*([0-9.]+)", d["logged_line"])] if m]
        if bests:
            log_best[(o["architecture_id"], o["task"])][o["evaluation_id"]] = bests[-1]
        if frs:
            fire_logs[(o["architecture_id"], o["task"])][o["evaluation_id"]] = frs[-1]
            fire_fallback.setdefault((o["architecture_id"], o["task"]), (frs[-1], frs))

    # ---------------------------------------------------------------- specs / diagrams
    sys.path.insert(0, str(SRC["archviz"]))
    try:
        from render import CSS  # noqa
        (SITE / "css/archviz.css").write_text("/* generated from archviz/render.py CSS by tools/build_data.py */\n" + CSS)
    except Exception as e:  # pragma: no cover
        warn(f"could not import archviz CSS: {e}")
    specs_dir, svg_dir = SRC["archviz"] / "specs", SRC["archviz"] / "out"

    def load_spec(key):
        p = specs_dir / f"{key}.json"
        if not p.exists():
            return None
        try:
            return json.load(open(p))
        except Exception as e:
            warn(f"bad spec {key}: {e}")
            return None

    style_re = re.compile(r"<style>.*?</style>", re.S)

    # pipeline ids such as "(clu8 · sample 557)" in the diagram's meta line are not shown on the site (the page
    # header already gives the iteration; the raw record id is in the page's Provenance section)
    clu_re = re.compile(r"\s*\(clu\d+\s*·\s*sample\s*-?\d+\)")

    def copy_svg(src, dst):
        s = src.read_text(encoding="utf-8")
        s = style_re.sub("", s, count=1)
        s = clu_re.sub("", s)
        dst.write_text(s, encoding="utf-8")

    import_re = re.compile(r"@import\s+url\([^)]*\)[^;]*;")

    def mixer_box(s):
        """Union of the token-mixer panel(s) of a rendered diagram, in SVG user units (None when not found).
        Panels are <rect class="panel"> followed by their <text class="cap">Token mixer…</text>, inside the
        translated main group."""
        tr = re.search(r'<g transform="translate\(([-\d.]+),\s*([-\d.]+)\)">', s)
        if not tr:
            return None
        ox, oy = float(tr.group(1)), float(tr.group(2))
        boxes = []
        for m in re.finditer(r'<rect class="panel[^"]*" x="([-\d.]+)" y="([-\d.]+)" width="([\d.]+)" height="([\d.]+)"[^>]*/>\s*'
                             r'<text class="cap"[^>]*>\s*Token mixer', s):
            x, y, w, h = map(float, m.groups())
            boxes.append((x + ox, y + oy, x + ox + w, y + oy + h))
        if not boxes:
            return None
        # The crop is the token mixer. Only when the mixer has no evolved (novel) block are the evolved blocks outside
        # it (e.g. an evolved output projection) added, so the thumbnail still shows what changed.
        mix = (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))
        nv = []
        body = s[tr.end():s.find('<g class="legend"') if '<g class="legend"' in s else len(s)]
        for m in re.finditer(r'<g class="node[^"]*\bnovel\b[^"]*"', body):
            sh = re.search(r'<(rect|circle) class="sh" ([^>]*)/>', body[m.end():m.end() + 1500])
            if not sh:
                continue
            at = dict(re.findall(r'([a-z]+)="([-\d.]+)"', sh.group(2)))
            if sh.group(1) == "rect":
                x, y, w, h = (float(at[k]) for k in ("x", "y", "width", "height"))
            else:
                r = float(at["r"]); x, y, w, h = float(at["cx"]) - r, float(at["cy"]) - r, 2 * r, 2 * r
            nv.append((x + ox - 12, y + oy - 14, x + ox + w + 14, y + oy + h + 12))
        inside = lambda b: b[0] + 12 >= mix[0] and b[2] - 14 <= mix[2] and b[1] + 14 >= mix[1] and b[3] - 12 <= mix[3]
        if nv and not any(inside(b) for b in nv):
            boxes += nv
        return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))

    def write_thumbs(src, key):
        """Self-contained thumbnails (<img>-safe: keeps the embedded <style>, drops the font @import that an
        SVG-as-image could not fetch anyway). They show the "evolved mechanisms" (focus) view cropped to the token
        mixer, so evolved blocks read as terracotta shapes and every thumbnail gets its own silhouette.
        One light and one dark variant; pages swap them with the site theme."""
        s = src.read_text(encoding="utf-8")
        s = import_re.sub("", s, count=1)
        s = clu_re.sub("", s)
        s = re.sub(r">\s+<", "><", s)
        s = re.sub(r'\s(?:tabindex|role)="[^"]*"', "", s)
        s = re.sub(r"<title>.*?</title>", "", s, flags=re.S)
        s = re.sub(r'<g class="expand".*?</g>', "", s, flags=re.S)        # no "details" pills in a thumbnail
        if mixer_box(s):  # parts outside the crop: legend, title, meta line, chips
            s = re.sub(r'<g class="legend">.*?</g></svg>', "</svg>", s, flags=re.S)
            s = re.sub(r'<text class="ttl".*?</text>', "", s, flags=re.S)
            # a chip may hold a nested <g class="badge">…</g> (numbered evolved note): consume it whole
            s = re.sub(r'<g class="chip[^"]*">(?:<g class="badge">.*?</g>|(?!</g>).)*?</g>', "", s, flags=re.S)
        s = re.sub(r"<style>(.*?)</style>", lambda m: "<style>" + re.sub(r"\s*\n\s*", "", m.group(1)) + "</style>", s, count=1, flags=re.S)
        s = re.sub(r'(<svg[^>]*?class=")av(")', r"\1av focus\2", s, count=1)
        m = re.search(r'<svg[^>]*viewBox="0 0 ([\d.]+) ([\d.]+)"', s)
        box = mixer_box(s)
        if m and box:
            pad = 10
            x0, y0, x1, y1 = box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad
            vb = f"{x0:g} {y0:g} {x1 - x0:g} {y1 - y0:g}"
        else:  # fallback: whole block without the title band and the legend
            lg = re.search(r'<g class="legend">.*?\by="([\d.]+)"', s, re.S)
            w, top = float(m.group(1)) if m else 1000.0, 72.0
            bot = float(lg.group(1)) - 10 if lg else (float(m.group(2)) if m else 400.0)
            vb, x1, x0, y1, y0 = f"0 {top:g} {w:g} {bot - top:g}", w, 0, bot, top
            if not box:
                warn(f"{key}: no token-mixer panel found; thumbnail uses the whole block")
        s = re.sub(r'(<svg[^>]*?)viewBox="[^"]*"', rf'\1viewBox="{vb}"', s, count=1)
        s = re.sub(r'(<svg[^>]*?)width="[^"]*"', rf'\1width="{x1 - x0:g}"', s, count=1)
        s = re.sub(r'(<svg[^>]*?)height="[^"]*"', rf'\1height="{y1 - y0:g}"', s, count=1)
        # the full-size background rects would now sit off-crop with their rounded corners: stretch them to the crop
        s = re.sub(r'<rect class="bg" width="100%" height="100%" rx="14"/>',
                   f'<rect class="bg" x="{x0:g}" y="{y0:g}" width="{x1 - x0:g}" height="{y1 - y0:g}"/>', s, count=1)
        (OUT / f"thumbs/{key}.svg").write_text(s, encoding="utf-8")
        dark = re.sub(r'(<svg[^>]*?)data-theme="light"', r'\1data-theme="dark"', s, count=1)
        if dark == s:
            dark = s.replace("<svg ", '<svg data-theme="dark" ', 1)
        (OUT / f"thumbs/{key}.d.svg").write_text(dark, encoding="utf-8")

    # ---------------------------------------------------------------- rounds
    snaps = {d["iteration_id"]: d for d in jl(runs / "selection_snapshots.jsonl")}
    coverage = {d["iteration_id"]: d for d in json.load(open(runs / "coverage.json"))}

    proposal = {}      # proposal_id -> compact record (all rounds)
    eval_prop = {}     # key -> proposal record
    rounds_meta = []
    for it in range(10):
        clu = f"clu{it}"
        R = json.load(open(pkg / f"website/rounds/{clu}.json"))
        recs = sorted(R["records"], key=lambda r: r["sample_num"])
        snap = snaps.get(clu) or {}
        groups = snap.get("groups") or {}
        nov = snap.get("derived_novelty_including_self")
        sl, nv = {}, {}
        order = []
        for gname in ("top", "dissim", "selected24"):
            for pid in groups.get(gname, []) or []:
                sl.setdefault(pid, {"top": 1, "dissim": 2, "selected24": 3}[gname])
                order.append(pid)
        if nov and len(nov) == len(order):
            nv = {pid: nov[i] for i, pid in enumerate(order)}
        for a in snap.get("logged_selection_annotations") or []:
            if isinstance(a.get("novelty"), (int, float)):
                nv[a["proposal_id"]] = a["novelty"]
        cols = {k: [] for k in ("s", "f", "w", "l", "i", "b", "e", "k", "n")}
        n_eval = 0
        for r in recs:
            pm = r.get("predicted_metrics") or {}
            link = r.get("evaluated_archive_link")
            key = None
            if link:
                key = by_aid.get(link.get("evaluated_architecture_id")) or rec_to_key.get(link.get("evaluated_archive_record_id"))
                if not key:
                    warn(f"{clu} sample {r['sample_num']}: evaluated link without key")
            pid = r["proposal_id"]
            cols["s"].append(r["sample_num"])
            cols["f"].append(r4(pm.get("fitness_logged")))
            cols["w"].append(r4(pm.get("wikitext2_ppl"), 3))
            cols["l"].append(r4(pm.get("listops_accuracy_pct"), 3))
            cols["i"].append(r.get("island_id_logged"))
            cols["b"].append(r4(r.get("best_so_far_predicted_fitness_derived")))
            cols["e"].append(key or 0)
            cols["k"].append(sl.get(pid, 0))
            cols["n"].append(r4(nv.get(pid)))
            proposal[pid] = {"clu": clu, "s": r["sample_num"], "f": pm.get("fitness_logged"), "key": key,
                             "pm": pm, "isl": r.get("island_id_logged"), "parents": r.get("parent_ids_ordered_logged") or []}
            if key:
                n_eval += 1
                eval_prop[key] = proposal[pid]
        cov = R.get("history_coverage")
        cv = coverage.get(clu, {})
        kinds = {k for k in cols["k"] if k}
        meta = {
            "id": clu, "iter": it + 1, "coverage": cov, "n": len(recs), "evaluated": n_eval,
            "shortlist": sum(1 for x in cols["k"] if x), "novelty": any(x is not None for x in cols["n"]),
            # 'top_dissim' = saved 30 top + 30 dissimilar lists; 'selected24' = only the logged selection set (clu3)
            "shortlist_kind": ("selected24" if kinds == {3} else "top_dissim" if kinds else None),
            # derived = 1 - mean(saved similarity row, incl. self); logged = novelty column of the selection log
            "novelty_kind": ("logged" if snap.get("logged_selection_annotations") else
                             "derived_incl_self" if nov else None),
            "accepted_note": cv.get("accepted_count_from_note"),
            "best_pred": r4(max((x for x in cols["f"] if x is not None), default=None)),
            "curve_x": R.get("curve_x"), "curve_y": R.get("curve_y"),
        }
        rounds_meta.append(meta)
        dump({**meta, "cols": cols}, OUT / f"rounds/{clu}.json")

    # ---------------------------------------------------------------- architectures
    archs = []
    n_diag = 0
    for row in index:
        key = row["key"]
        w = web_by_aid.get(row["architecture_id"]) or {}
        zr = zoo_rows.get(key, {})
        spec = load_spec(key)
        meas = {m["task"]: m for m in w.get("measurements", [])}
        wt2 = meas.get("wikitext2", {}).get("value", row.get("wt2"))
        lo = meas.get("listops", {}).get("value", row.get("listops"))
        fit = (w.get("fitness_derived_from_reported_metrics") or {}).get("value", row.get("fitness_paper"))
        ov = PAPER_OVERRIDES.get(key)
        if ov:
            wt2, lo = ov.get("wt2", wt2), ov.get("listops", lo)
            fit = fitness(wt2, lo)
        elif wt2 is not None and lo is not None and fit is not None and abs(fitness(wt2, lo) - fit) > 1e-6:
            warn(f"{key}: fitness mismatch {fit} vs formula {fitness(wt2, lo)}")
        # fire rates
        fr = {"wt2": zr.get("fire_rate_wt2"), "listops": zr.get("fire_rate_listops")}
        fr_n = {}  # task -> number of training logs when they disagree on the last-epoch fire rate
        fr_ep = {}  # task -> {"at": epoch of the shown value, "of": last epoch of the log, "why": "nan"|"silent"}
        curves = {}
        for task, short in (("wikitext2", "wt2"), ("listops", "listops")):
            lg = ((zr.get("fire_rate_logs") or {}).get(task) or {})
            pe = lg.get("per_epoch") or []
            if pe:
                curves[short] = [[e.get("epoch"), e.get("fire_rate")] for e in pe]
            if fr[short] is None and (row["architecture_id"], task) in fire_fallback:
                v, frs = fire_fallback[(row["architecture_id"], task)]
                fr[short] = v
                curves[short] = [[i + 1, x] for i, x in enumerate(frs)]
            logs = fire_logs.get((row["architecture_id"], task), {})
            if fr[short] is not None and logs and not any(abs(fr[short] - v) < 5e-4 for v in logs.values()):
                warn(f"{key}: {task} fire rate {fr[short]} matches no training log last epoch {sorted(logs.values())}")
            if len(set(logs.values())) > 1:
                fr_n[short] = len(logs)
            # Which epoch the shown value comes from. Some logs stop printing "Fire Rate:" early (loss went NaN, or
            # the rate hit 0.000 and the field was dropped); then the value is from an earlier epoch than the last one.
            vals = [(e.get("epoch"), e.get("fire_rate")) for e in pe if e.get("fire_rate") is not None]
            if fr[short] is not None and vals and pe:
                at, last = vals[-1][0], max(e.get("epoch") or 0 for e in pe)
                if at is not None and at < last:
                    after = [d for d in lines.get(lg.get("log_id") or "", []) if (d.get("epoch") or 0) > at]
                    why = ("nan" if any(re.search(r"Loss:\s*nan", d["logged_line"], re.I) for d in after)
                           else "silent" if fr[short] == 0 else None)
                    fr_ep[short] = {"at": at, "of": last, **({"why": why} if why else {})}
        # multi-log experts: do any of the recovered WT2 logs reach the reported PPL? (none do for the known cases)
        if fr_n.get("wt2") and wt2 is not None:
            bests = log_best.get((row["architecture_id"], "wikitext2"), {})
            if bests and not any(abs(b - wt2) < 0.05 for b in bests.values()):
                fr_ep["wt2NoMatch"] = True
        # name
        zname = (zr.get("name") or "").strip()
        if spec and spec.get("name"):
            name, name_src = spec["name"], spec.get("name_source") or "spec"
        elif row.get("public_name"):
            name, name_src = row["public_name"], "public"
        elif zname:
            name, name_src = pretty_slug(zname), "rationale"
        else:
            name, name_src = None, None
        prop = eval_prop.get(key)
        diag = (svg_dir / f"{key}.svg").exists()
        details = []
        if diag:
            n_diag += 1
            copy_svg(svg_dir / f"{key}.svg", OUT / f"diagrams/{key}.svg")
            write_thumbs(svg_dir / f"{key}.svg", key)
            for p in sorted(svg_dir.glob(f"{key}__*.svg")):
                did = p.stem.split("__", 1)[1]
                # theme/focus render variants ({key}__dark__focus, {key}__core__dark__focus, …) are not detail
                # views: the page switches theme and focus with CSS classes on the one SVG, so skip them.
                if "__" in did or did in ("dark", "focus"):
                    continue
                details.append(did)
                copy_svg(p, OUT / f"diagrams/{p.name}")
        a = {
            "key": key, "name": name, "nameSrc": name_src, "slug": zname or None,
            "iter": row["iteration"], "group": row["group"], "sample": row["sample_num"],
            "expert": row["iteration"] == 0,
            "base": BASE_ALIAS.get(b0 := ((spec or {}).get("base") or zr.get("base_family")), b0),
            "summary": SUMMARY_OVERRIDES.get(key) or (spec or {}).get("summary"),
            "wt2": wt2, "listops": lo, "fit": r4(fit, 5),
            "score": r4((w.get("archive_logged_score") or {}).get("value")),
            "fr": fr,
            **({"frN": fr_n} if fr_n else {}),
            **({"frEp": fr_ep} if fr_ep else {}),
            "pred": ({"fit": r4(prop["f"]), "wt2": r4(prop["pm"].get("wikitext2_ppl"), 2),
                      "listops": r4(prop["pm"].get("listops_accuracy_pct"), 2), "island": prop["isl"]} if prop else None),
            "diagram": diag, "details": details,
            "spec": bool(spec),
        }
        archs.append(a)
        # per-architecture lazy file
        rat = None
        rp = pkg / w.get("rationale", {}).get("path", "") if w.get("rationale") else None
        if rp and rp.is_file():
            rat = rp.read_text(encoding="utf-8", errors="replace")
        # The displayed file is always the arch_index `source` (= the exact archive bytes whose hash is the
        # architecture_id). Diagram specs take their `code` line ranges from this same file (SPEC_GUIDE), so
        # ranges stay valid; any range beyond the file is dropped below instead of highlighting wrong lines.
        code_src = WORK / row["source"]
        code_info = None
        if code_src.exists():
            raw = code_src.read_bytes()
            shutil.copyfile(code_src, OUT / f"code/{key}.py")
            sha = hashlib.sha256(raw).hexdigest()
            if f"arch-sha256-{sha}" != row["architecture_id"]:
                warn(f"{key}: displayed source hash {sha[:12]} != archive architecture_id")
            zf = zr.get("file")
            zp = SRC["zoo_dir"] / zf if zf else None
            same_zoo = bool(zp and zp.is_file() and hashlib.sha256(zp.read_bytes()).hexdigest() == sha)
            code_info = {"label": ("zoo/" + zf) if same_zoo else str(Path(row["source"]).name),
                         "sha256": sha, "lines": raw.decode("utf-8", "replace").count("\n") + 1,
                         "version": "archive",
                         "named": [{"name": v["name"], "path": (v.get("source_path") or "")} for v in w.get("named_source_variants") or []],
                         "namedNote": zr.get("notes") if zr.get("elite_file") else None}
        else:
            warn(f"{key}: missing source {code_src}")
        detail = {
            "key": key,
            "rationale": rat,
            "callouts": (spec or {}).get("callouts") or {},
            "notes": (spec or {}).get("notes") or [],
            "meta": (spec or {}).get("meta"),
            "nodes": {n["id"]: {"label": n.get("label"), "code": n.get("code"), "tip": n.get("tip"),
                                **({"novel": n["novel"]} if n.get("novel") else {})}
                      for n in (spec or {}).get("nodes", []) if n.get("code") or n.get("tip")},
            "fireCurves": curves,
            "named": [{"name": v["name"], "identity": v.get("identity_status")} for v in w.get("named_source_variants") or []],
            "zooFile": zr.get("file"), "zooNotes": zr.get("notes") or None,
            "codeFile": code_info,
        }
        # detail-graph node code ranges too
        for n in (spec or {}).get("nodes", []):
            for dn in ((n.get("detail") or {}).get("nodes") or []):
                if dn.get("code") or dn.get("tip"):
                    detail["nodes"].setdefault(f"{n['id']}/{dn['id']}", {"label": dn.get("label"), "code": dn.get("code"), "tip": dn.get("tip")})
        # validate line ranges against the displayed file (spec + data-code attributes baked into the SVGs)
        if code_info:
            nl = code_info["lines"]
            for nid, nd in detail["nodes"].items():
                if nd.get("code") and not range_ok(nd["code"], nl):
                    warn(f"{key}: node {nid} code range {nd['code']} outside displayed file ({nl} lines); dropped")
                    nd["code"] = None
            bad_svg = []
            for p in [OUT / f"diagrams/{key}.svg"] + [OUT / f"diagrams/{key}__{d}.svg" for d in details]:
                if p.exists():
                    bad_svg += [c for c in re.findall(r'data-code="([^"]+)"', p.read_text(encoding="utf-8")) if not range_ok(c, nl)]
            if bad_svg:
                warn(f"{key}: SVG data-code ranges outside displayed file: {bad_svg[:5]}")
                detail["badSvgRanges"] = bad_svg
        dump(detail, OUT / f"arch/{key}.json")

    # ---------------------------------------------------------------- zoo (ASI-Arch etc.)
    zoo = []
    for k, r in zoo_rows.items():
        if r.get("collection") != "asi_arch":
            continue
        src = SRC["zoo_dir"] / (r.get("file") or "")
        has_code = src.is_file()
        if has_code:
            shutil.copyfile(src, OUT / f"zoo/{k}.py")
        zoo.append({"key": k, "name": r.get("name"), "id": r.get("sample"), "subset": r.get("subset"),
                    "base": r.get("base_family"), "wt2": r4(r.get("wt2_ppl_reported"), 2), "listops": r4(r.get("listops_acc"), 2),
                    "fr": {"wt2": r4(r.get("fire_rate_wt2"), 3), "listops": r4(r.get("fire_rate_listops"), 3)},
                    "parent": r.get("parent_id"), "motivation": r.get("motivation_text"),
                    "results": r.get("has_csv_results") or None, "notes": r.get("notes") or None,
                    "file": r.get("file"), "code": has_code})
    zoo.sort(key=lambda x: (x["subset"] != "baseline", x["subset"], x["id"] or 0))
    dump(zoo, OUT / "zoo.json")

    # ---------------------------------------------------------------- per-iteration fronts (measured)
    def front(points):
        pts = sorted(points, key=lambda p: (p["wt2"], -p["listops"]))
        out, best = [], -1
        for p in pts:
            if p["listops"] > best:
                out.append(p["key"])
                best = p["listops"]
        return out
    fronts = []
    for t in range(11):
        pts = [a for a in archs if a["iter"] <= t and a["wt2"] is not None and a["listops"] is not None]
        fronts.append(front(pts))

    meta = {
        "built": __import__("datetime").date.today().isoformat(),
        "counts": {"architectures": len(archs), "experts": sum(a["expert"] for a in archs),
                   "discovered": sum(not a["expert"] for a in archs), "diagrams": n_diag,
                   "specs": sum(a["spec"] for a in archs), "zoo": len(zoo)},
        "rounds": rounds_meta,
        "fronts": fronts,
        "finalFront": fronts[10],
        "warnings": warnings,
    }
    dump(meta, OUT / "meta.json")
    dump(archs, OUT / "architectures.json")

    # ---------------------------------------------------------------- paper Fig. 4 trajectory (hero curve)
    if SRC["fig4"].exists():
        f4 = json.load(open(SRC["fig4"]))
        pts = sorted(f4["points"], key=lambda p: p["evaluationOrder"])
        dump({"source": "Paper Fig. 4 (search trajectory), marker positions digitized from the vector PDF",
              "baseline": f4.get("baselineFitness", 0.5),
              # [evaluation order (0 = first expert), iteration (0 = experts), composite fitness, best-so-far improvement]
              "points": [[p["evaluationOrder"], p["iteration"], round(p["fitness"], 4), 1 if p.get("bestSoFarImprovement") else 0]
                         for p in pts]}, OUT / "paper_trajectory.json")
    else:
        warn(f"paper Fig. 4 data missing ({SRC['fig4']}): no hero trajectory")

    if args.figures:
        render_figures()

    size = sum(p.stat().st_size for p in SITE.rglob("*") if p.is_file())
    print(f"built {len(archs)} architectures ({n_diag} diagrams), {len(zoo)} zoo programs; site size {size/1e6:.1f} MB; {len(warnings)} warnings")


def render_figures():
    from PIL import Image, ImageChops
    figs = SRC["paper"] / "figs"
    tmp = SITE / "assets/_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    for name, out in [("pipeline", "pipeline"), ("representation-v2", "representation"), ("motivation_v3", "motivation"), ("archs", "elites-archs")]:
        subprocess.run(["pdftoppm", "-png", "-r", "220", "-singlefile", str(figs / f"{name}.pdf"), str(tmp / out)], check=True)
        im = Image.open(tmp / f"{out}.png").convert("RGB")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bbox = ImageChops.difference(im, bg).getbbox()
        if bbox:
            pad = 16
            im = im.crop((max(0, bbox[0] - pad), max(0, bbox[1] - pad), min(im.width, bbox[2] + pad), min(im.height, bbox[3] + pad)))
        if im.width > 2400:
            im = im.resize((2400, round(im.height * 2400 / im.width)), Image.LANCZOS)
        im.save(SITE / f"assets/{out}.webp", "WEBP", quality=88, method=6)
    shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
