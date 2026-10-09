#!/usr/bin/env python
"""Create the initial archive D_0 from expert programs and their measured metrics.

``--metrics`` is a CSV with an ``id`` column (the program file stem) and one
column per metric of the fitness mode (``wikitext2_ppl``, ``listops_acc``).
Fingerprints are computed by the running surrogate service. Design rationales
are read from ``<rationales>/<id>.txt`` and embedded; a program without one
gets an empty rationale view (a warning is logged), which weakens three-view
novelty for it.

    python scripts/init_archive.py --run-dir runs/main --programs zoo/experts \
        --metrics examples/expert_metrics.csv --rationales zoo/experts/rationales
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openarchevo.cli import search_parser, setup, update_front  # noqa: E402
from openarchevo.outer_loop import init_archive, read_metrics_csv  # noqa: E402


def main() -> None:
    p = search_parser(__doc__)
    p.add_argument("--programs", type=Path, required=True, help="Directory of expert *.py programs.")
    p.add_argument("--metrics", type=Path, required=True, help="CSV: id column + one column per metric.")
    p.add_argument("--rationales", type=Path, help="Directory of <id>.txt design rationales (recommended).")
    args = p.parse_args()
    _, run, comps = setup(args)
    if run.archive.exists():
        sys.exit(f"{run.archive} already exists.")
    if args.rationales is None:
        print("WARNING: no --rationales given; every initial program gets an empty rationale view.", file=sys.stderr)
    archive = init_archive(sorted(args.programs.glob("*.py")), read_metrics_csv(args.metrics), comps.surrogate,
                           comps.embedder, comps.fitness, args.rationales)
    run.root.mkdir(parents=True, exist_ok=True)
    archive.save(run.archive)
    update_front(run, comps)
    print(f"Initial archive with {len(archive)} programs written to {run.archive}")


if __name__ == "__main__":
    main()
