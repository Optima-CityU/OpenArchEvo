#!/usr/bin/env python
"""Run only the inner loop (no shortlist, no training batch).

Refreshes the surrogate on the archive, selects roots and evolves programs until
``inner_loop.pool_size`` candidates are accepted; the pool is written to ``--out``.
Useful for inspecting prompts, acceptance statistics and island dynamics.

    python scripts/run_inner_loop.py --run-dir runs/demo --dry-run --out runs/demo/inner_pool.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openarchevo.candidate import Archive  # noqa: E402
from openarchevo.cli import search_parser, setup  # noqa: E402
from openarchevo.inner_loop import InnerLoop  # noqa: E402
from openarchevo.outer_loop import refresh_surrogate, select_roots  # noqa: E402


def main() -> None:
    p = search_parser(__doc__)
    p.add_argument("--out", type=Path, required=True, help="JSON-lines file for the accepted pool.")
    p.add_argument("--round-index", type=int, default=0, help="Label used in candidate ids.")
    args = p.parse_args()
    cfg, run, comps = setup(args)
    archive = Archive.load(run.archive)
    refresh_surrogate(archive, comps)
    roots = select_roots(list(archive), comps.similarity, len(archive.initial()) or len(archive))
    inner = InnerLoop(cfg.inner_loop, comps.generator, comps.embedder, comps.surrogate, comps.similarity,
                      comps.fitness, round_index=args.round_index, seed=cfg.seed, pool_path=args.out)
    try:
        result = inner.run(roots)
    except RuntimeError as exc:  # persistent LLM or surrogate errors
        sys.exit(str(exc))
    comps.similarity.save(keep_ids=archive.ids())
    best = max(result.pool, key=lambda c: c.predicted_fitness, default=None)
    print(json.dumps({"accepted": len(result.pool), "stats": result.stats, "island_sizes": inner.islands.sizes(),
                      "best": None if best is None else {"id": best.id, "predicted_fitness": best.predicted_fitness}},
                     indent=1))


if __name__ == "__main__":
    main()
