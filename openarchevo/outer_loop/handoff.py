"""The hand-off between search and real training.

In the original study, search and training ran on different machines, so each
outer iteration ended by exporting the selected programs and resumed once their
measured results were collected. This module makes that boundary explicit:

1. :func:`write_training_batch` exports ``Q_t`` to ``round_XX/to_train/``.
2. Each program is trained on every proxy task (``training/``), and each run
   writes ``round_XX/results/<id>.<task>.json`` with a ``metrics`` dict.
3. :func:`ingest_results` merges the measured metrics into the archive.

:func:`run_training_commands` is an optional hook that executes step 2 locally
from shell templates (``outer_loop.train_commands`` in ``configs/search.yaml``).
"""

from __future__ import annotations

import json
import logging
import math
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Callable, Mapping, Sequence

from openarchevo.candidate import Archive, Candidate, read_jsonl, write_jsonl
from openarchevo.outer_loop.workspace import RunDir

log = logging.getLogger("openarchevo.handoff")

#: Repository root; ``train_commands`` run from here, so relative script paths resolve.
REPO_ROOT = Path(__file__).resolve().parents[2]


def write_training_batch(run: RunDir, t: int, batch: Sequence[Candidate]) -> Path:
    """Write one ``<id>.py`` per selected program plus ``manifest.jsonl``."""
    out = run.batch_dir(t)
    out.mkdir(parents=True, exist_ok=True)
    run.results_dir(t).mkdir(parents=True, exist_ok=True)
    for cand in batch:
        (out / f"{cand.id}.py").write_text(cand.code, encoding="utf-8")
    write_jsonl(run.manifest(t), batch)
    return out


def collect_metrics(results_dir: Path, cand_id: str) -> dict[str, float]:
    """Merge the ``metrics`` dicts of all ``<id>.*.json`` result files."""
    metrics: dict[str, float] = {}
    for path in sorted(results_dir.glob(f"{cand_id}.*.json")):
        metrics.update(json.loads(path.read_text()).get("metrics", {}))
    return metrics


def ingest_results(run: RunDir, t: int, fitness: Callable[[Mapping[str, float]], float],
                   required: Sequence[str], allow_missing: bool = False) -> list[Candidate]:
    """Add the trained programs of round ``t`` to the archive.

    Raises if a program lacks a required metric, unless ``allow_missing`` is set,
    in which case that program is skipped (it does not enter the archive).
    """
    archive = Archive.load(run.archive)
    added, missing = [], []
    for cand in read_jsonl(run.manifest(t)):
        metrics = collect_metrics(run.results_dir(t), cand.id)
        if any(metrics.get(m) is None or not math.isfinite(float(metrics[m])) for m in required):
            missing.append(cand.id)
            continue
        cand.measured = {k: float(v) for k, v in metrics.items()}
        cand.fitness = fitness(cand.measured)
        added.append(cand)
    if missing and not allow_missing:
        raise RuntimeError(f"Round {t}: missing results for {missing} in {run.results_dir(t)} "
                           f"(need {list(required)}). Re-run training or pass --allow-missing.")
    for cand in added:
        archive.add(cand)
    archive.save(run.archive)
    run.ingested_marker(t).write_text(json.dumps({"added": [c.id for c in added], "skipped": missing}) + "\n")
    log.info("Round %d: ingested %d programs (%d skipped); archive size %d", t, len(added), len(missing), len(archive))
    return added


def run_training_commands(run: RunDir, t: int, templates: Sequence[str], cwd: Path = REPO_ROOT) -> None:
    """Optional automatic execution: run every template for every program of round ``t``.

    Commands run in ``cwd`` (the repository root). Placeholders: ``{python}``,
    ``{repo}`` (repository root), ``{program}``, ``{id}``, ``{results_dir}``; paths
    are absolute. Failures are logged; :func:`ingest_results` then reports the
    missing results.
    """
    if not templates:
        raise ValueError("outer_loop.train_commands is empty; nothing to run.")
    failed = 0
    for cand in read_jsonl(run.manifest(t)):
        values = {"python": shlex.quote(sys.executable), "repo": shlex.quote(str(REPO_ROOT)),
                  "program": shlex.quote(str((run.batch_dir(t) / f"{cand.id}.py").resolve())),
                  "id": cand.id, "results_dir": shlex.quote(str(run.results_dir(t).resolve()))}
        for template in templates:
            cmd = template.format(**values)
            log.info("Running: %s", cmd)
            if subprocess.run(cmd, shell=True, cwd=cwd).returncode != 0:
                failed += 1
                log.error("Training command failed for %s: %s", cand.id, cmd)
    if failed:
        log.error("Round %d: %d training command(s) failed.", t, failed)
