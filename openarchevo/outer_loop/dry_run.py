"""Offline helpers for ``--dry-run``: a toy initial archive and fake training results."""

from __future__ import annotations

import json
from pathlib import Path

from openarchevo.candidate import read_jsonl
from openarchevo.components import Components
from openarchevo.outer_loop.archive_init import init_archive
from openarchevo.outer_loop.workspace import RunDir
from openarchevo.surrogate.fake import FakeSurrogate

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "blocks"


def bootstrap_toy_archive(run: RunDir, comps: Components, programs_dir: Path = EXAMPLES) -> None:
    """``D_0`` from the example blocks, with fake 'measured' metrics."""
    assert isinstance(comps.surrogate, FakeSurrogate), "bootstrap_toy_archive needs the fake surrogate"
    programs = sorted(programs_dir.glob("*.py"))
    metrics = {p.stem: comps.surrogate.train(p.read_text(), p.stem) for p in programs}
    archive = init_archive(programs, metrics, comps.surrogate, comps.embedder, comps.fitness)
    for cand in archive:  # give the toy seeds a rationale so every view is populated
        cand.rationale = (cand.code.split('"""')[1].strip() if '"""' in cand.code else cand.id)
        cand.embedding = comps.embedder.embed(cand.rationale)
    run.root.mkdir(parents=True, exist_ok=True)
    archive.save(run.archive)
    run.dry_run_marker.write_text("This run directory uses fake components (--dry-run).\n")


def fake_training(run: RunDir, t: int, surrogate: FakeSurrogate) -> None:
    """Write result files as the real training scripts would: here one per metric
    of the fitness mode, ``results/<id>.<metric>.json``."""
    run.results_dir(t).mkdir(parents=True, exist_ok=True)
    for cand in read_jsonl(run.manifest(t)):
        for metric, value in surrogate.train(cand.code, cand.id).items():
            path = run.results_dir(t) / f"{cand.id}.{metric}.json"
            path.write_text(json.dumps({"id": cand.id, "task": metric, "metrics": {metric: value}}))
