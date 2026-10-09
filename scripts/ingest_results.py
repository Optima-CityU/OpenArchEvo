#!/usr/bin/env python
"""Merge measured training results of a round into the archive.

Reads ``round_XX/to_train/manifest.jsonl`` and ``round_XX/results/<id>.<task>.json``
(written by ``training/train_*.py``), computes the measured fitness, appends the
programs to ``archive.jsonl`` and refreshes ``front.jsonl``.

    python scripts/ingest_results.py --run-dir runs/main            # the pending round
    python scripts/ingest_results.py --run-dir runs/main --round 3 --allow-missing
    python scripts/ingest_results.py --run-dir runs/demo --dry-run --fake-train   # rehearse offline
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openarchevo.cli import search_parser, setup, summary, update_front  # noqa: E402
from openarchevo.outer_loop import ingest_results  # noqa: E402


def main() -> None:
    p = search_parser(__doc__)
    p.add_argument("--round", type=int, help="Round to ingest (default: the pending one).")
    p.add_argument("--allow-missing", action="store_true", help="Skip programs without complete results.")
    p.add_argument("--fake-train", action="store_true",
                   help="With --dry-run: write fake result files first, to rehearse the hand-off offline.")
    args = p.parse_args()
    if args.fake_train and not args.dry_run:
        p.error("--fake-train needs --dry-run")
    _, run, comps = setup(args, preflight=False)  # ingesting needs no LLM or surrogate
    t = args.round if args.round is not None else run.pending_round()
    if t is None:
        sys.exit("No round is waiting for results.")
    if run.ingested_marker(t).exists():
        sys.exit(f"Round {t} has already been ingested.")
    if args.fake_train:
        from openarchevo.outer_loop.dry_run import fake_training

        fake_training(run, t, comps.surrogate)
    try:
        added = ingest_results(run, t, comps.fitness, comps.required_metrics, args.allow_missing)
    except RuntimeError as exc:
        sys.exit(str(exc))
    front = update_front(run, comps)
    print(f"Round {t}: added {len(added)} programs; front has {len(front)} members.\n{summary(run)}")


if __name__ == "__main__":
    main()
