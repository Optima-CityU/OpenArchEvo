#!/usr/bin/env python
"""Run outer iterations up to the training hand-off.

Manual mode (default; how the original study was run): one round is executed and
the selected programs are written to ``<run-dir>/round_XX/to_train/``. Train them
(``training/``), put the result files in ``round_XX/results/`` and run
``scripts/ingest_results.py``; then call this script again for the next round.

Automatic mode (``--auto``): after each round the ``outer_loop.train_commands``
from the config are executed (from the repository root) for every selected
program, results are ingested and the next round starts, until ``--rounds``
rounds or the budget is used up.

    python scripts/run_round.py --run-dir runs/main
    python scripts/run_round.py --run-dir runs/demo --dry-run --auto      # offline demo
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openarchevo.candidate import Archive  # noqa: E402
from openarchevo.cli import search_parser, setup, summary, update_front  # noqa: E402
from openarchevo.outer_loop import ingest_results, run_round, run_training_commands  # noqa: E402

log = logging.getLogger("openarchevo")


def main() -> None:
    p = search_parser(__doc__)
    p.add_argument("--auto", action="store_true", help="Train (train_commands, or fake in --dry-run) and ingest automatically.")
    p.add_argument("--rounds", type=int, help="Rounds to run in --auto mode (default: outer_loop.rounds).")
    args = p.parse_args()
    cfg, run, comps = setup(args)
    if not run.archive.exists():
        sys.exit(f"No archive at {run.archive}. Create one with scripts/init_archive.py (or use --dry-run).")

    pending = run.pending_round()
    if pending is not None:
        sys.exit(f"Round {pending} is waiting for training results in {run.results_dir(pending)}; "
                 f"run scripts/ingest_results.py --run-dir {run.root} first.")

    last = cfg.outer_loop.rounds if args.rounds is None else run.next_round() + args.rounds - 1
    while True:
        t = run.next_round()
        if t > min(last, cfg.outer_loop.rounds):
            log.info("Reached the configured number of rounds.")
            break
        if Archive.load(run.archive).new_evaluations() >= cfg.outer_loop.budget:
            log.info("Evaluation budget B=%d is used up.", cfg.outer_loop.budget)
            break
        try:
            result = run_round(cfg, run, comps, t)
        except RuntimeError as exc:  # e.g. persistent LLM or surrogate errors in the inner loop
            sys.exit(f"Round {t} failed: {exc}")
        if not args.auto:
            print(f"\nRound {t}: {len(result.batch)} programs written to {run.batch_dir(t)}\n"
                  f"Train each on every proxy task so that {run.results_dir(t)}/<id>.<task>.json exists, e.g.\n"
                  f"  python training/train_wikitext2.py --program {run.batch_dir(t)}/<id>.py "
                  f"--result_json {run.results_dir(t)}/<id>.wikitext2.json\n"
                  f"then run: python scripts/ingest_results.py --run-dir {run.root}")
            return
        if args.dry_run:
            from openarchevo.outer_loop.dry_run import fake_training

            fake_training(run, t, comps.surrogate)
        else:
            run_training_commands(run, t, cfg.outer_loop.train_commands)
        try:
            ingest_results(run, t, comps.fitness, comps.required_metrics)
        except RuntimeError as exc:  # some training command failed or wrote no metrics
            sys.exit(f"{exc}\nSee the training errors above. After fixing them, train the missing programs "
                     f"and run scripts/ingest_results.py --run-dir {run.root}.")
        update_front(run, comps)
        log.info("Round %d done. %s", t, summary(run))
    print(summary(run))


if __name__ == "__main__":
    main()
