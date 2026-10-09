"""Small helpers shared by the scripts in ``scripts/``."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from openarchevo.candidate import Archive, write_jsonl
from openarchevo.components import Components, build_components, check_endpoints
from openarchevo.config import SearchConfig, load_search_config
from openarchevo.outer_loop.selection import archive_front
from openarchevo.outer_loop.workspace import RunDir

REPO_ROOT = Path(__file__).resolve().parents[1]


def search_parser(doc: str) -> argparse.ArgumentParser:
    """Parser with the shared options; ``doc`` is the script's docstring (first line = description)."""
    lines = doc.strip().splitlines()
    p = argparse.ArgumentParser(description=lines[0], epilog="\n".join(lines[1:]).strip() or None,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "search.yaml")
    p.add_argument("--run-dir", type=Path, required=True, help="Directory holding archive.jsonl and round_XX/.")
    p.add_argument("--dry-run", action="store_true",
                   help="Fake LLM, embedder, surrogate and training; CPU only, no API keys.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def setup(args: argparse.Namespace, preflight: bool = True) -> tuple[SearchConfig, RunDir, Components]:
    """Load the config, build the components and check the run directory.

    ``preflight``: in a real run, check that the surrogate service and the
    embedding endpoint answer before any work starts.
    """
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_search_config(args.config, dry_run=args.dry_run)
    run = RunDir(args.run_dir)
    if run.archive.exists() and run.dry_run_marker.exists() != args.dry_run:
        kind = "a dry-run (fake)" if run.dry_run_marker.exists() else "a real"
        raise SystemExit(f"{run.root} holds {kind} run; use {'--dry-run' if not args.dry_run else 'no --dry-run'} "
                         "or another --run-dir. Fake and real results are never mixed in one archive.")
    try:
        comps = build_components(cfg, run.similarity_cache, dry_run=args.dry_run)
    except (EnvironmentError, ImportError) as exc:  # missing API variables or optional packages
        raise SystemExit(f"{exc}\n(Use --dry-run for an offline run.)") from exc
    if preflight and not args.dry_run:
        try:
            check_endpoints(comps)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
    if args.dry_run and not run.archive.exists():
        from openarchevo.outer_loop.dry_run import bootstrap_toy_archive

        bootstrap_toy_archive(run, comps)
        update_front(run, comps)
        logging.getLogger("openarchevo").info("Dry run: created a toy archive at %s", run.archive)
    return cfg, run, comps


def update_front(run: RunDir, comps: Components) -> list:
    """Recompute ``ND_D(D)`` and write it to ``front.jsonl``."""
    archive = Archive.load(run.archive)
    front = archive_front(list(archive), comps.similarity)
    write_jsonl(run.front, sorted(front, key=lambda c: c.fitness or 0.0, reverse=True))
    comps.similarity.save(keep_ids=archive.ids())
    return front


def summary(run: RunDir) -> str:
    archive = Archive.load(run.archive)
    best = max(archive, key=lambda c: c.fitness or float("-inf"))
    return (f"archive: {len(archive)} programs ({archive.new_evaluations()} new) | "
            f"best fitness {best.fitness:.3f} ({best.id}, {best.measured})")
