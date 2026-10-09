"""One outer iteration (Algorithm 1 of the paper), up to the training hand-off.

    1. refresh the surrogate on D_{t-1}
    2. roots P_t0  <- NSGA-II on [y, N(a; D_{t-1})], |P_t0| = |D_0|
    3. pool  P_t   <- inner loop (Algorithm 2)
    4. P_f         <- P_top U P_dissim, without near-duplicates (of D_{t-1} or of each other)
    5. batch Q_t   <- NSGA-II on [y_hat, N(a; P_f)], |Q_t| <= min(K, B - |D_{t-1} \\ D_0|)
    6. export Q_t to round_XX/to_train/            -> real training happens outside
    7. (ingest_results) D_t <- D_{t-1} U Q_t with measured metrics
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from openarchevo.candidate import Archive, Candidate, write_jsonl
from openarchevo.components import Components
from openarchevo.config import SearchConfig
from openarchevo.inner_loop import InnerLoop
from openarchevo.outer_loop.handoff import write_training_batch
from openarchevo.outer_loop.selection import (drop_near_duplicates, most_dissimilar, screen_against, select_batch,
                                              select_roots, top_candidates)
from openarchevo.outer_loop.workspace import RunDir

log = logging.getLogger("openarchevo.outer_loop")


@dataclass
class RoundResult:
    round: int
    pool: list[Candidate]
    shortlist: list[Candidate]
    batch: list[Candidate]
    inner_stats: dict[str, int]


def refresh_surrogate(archive: Archive, comps: Components) -> dict:
    """Fit the surrogate on every archived fingerprint and its measured metrics."""
    features = [c.fingerprint for c in archive]
    targets = {m: [c.measured[m] for c in archive] for m in comps.required_metrics}
    comps.surrogate.fit(features, targets)
    return {"n": len(features), "ids": [c.id for c in archive], "targets": comps.required_metrics}


def run_round(cfg: SearchConfig, run: RunDir, comps: Components, t: int) -> RoundResult:
    """Run steps 1-6 of outer iteration ``t`` and write all artefacts under ``round_XX/``."""
    archive = Archive.load(run.archive)
    if not len(archive):
        raise RuntimeError(f"Empty archive at {run.archive}; initialise it with scripts/init_archive.py.")
    out = run.round_dir(t)
    out.mkdir(parents=True, exist_ok=True)
    oc, sim = cfg.outer_loop, comps.similarity
    warm_start = oc.warm_start.enabled and t == 1

    # 1. surrogate refresh
    context = refresh_surrogate(archive, comps)
    (out / "surrogate_context.json").write_text(json.dumps(context, indent=1))

    # 2. roots
    n_roots = len(archive.initial()) or len(archive)
    roots = select_roots(list(archive), sim, n_roots)
    write_jsonl(out / "roots.jsonl", roots)
    log.info("Round %d: %d roots from an archive of %d", t, len(roots), len(archive))

    # 3. inner loop
    inner = InnerLoop(cfg.inner_loop, comps.generator, comps.embedder, comps.surrogate, sim, comps.fitness,
                      round_index=t, seed=cfg.seed + t, pool_path=out / "pool.jsonl",
                      warm_start_pool=oc.warm_start.pool_size if warm_start else None)
    result = inner.run(roots)

    # 4. shortlist: performance + dissimilarity, then archive screening
    top = top_candidates(result.pool, oc.shortlist_top, oc.best_per_island, oc.stepping_stones)
    dissim = most_dissimilar(result.pool, top, sim, oc.shortlist_dissim)
    for group, cands in (("top", top), ("dissim", dissim)):
        for c in cands:
            c.meta["group"] = group
    shortlist = drop_near_duplicates(screen_against(top + dissim, list(archive), sim), sim)
    write_jsonl(out / "shortlist.jsonl", shortlist)

    # 5. training batch within the remaining budget
    k = max(0, min(oc.batch_size, oc.budget - archive.new_evaluations()))
    batch = select_batch(shortlist, sim, k)
    log.info("Round %d: pool %d -> shortlist %d -> batch %d (budget left %d)", t, len(result.pool),
             len(shortlist), len(batch), oc.budget - archive.new_evaluations())

    # 6. hand-off
    write_training_batch(run, t, batch)
    sim.save(keep_ids=archive.ids() | {c.id for c in batch})
    return RoundResult(t, result.pool, shortlist, batch, result.stats)
