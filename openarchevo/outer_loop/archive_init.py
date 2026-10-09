"""Build the initial archive ``D_0`` from expert programs and their measured metrics.

Each program gets its fingerprint from the surrogate (the service executes it),
its rationale from an optional text file, and its fitness from the measured
metrics. The paper's ``D_0`` holds 20 spiking adaptations of linear-recurrent and
state-space models (see ``zoo/experts``).
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

from openarchevo.candidate import Archive, Candidate
from openarchevo.llm.base import TextEmbedder
from openarchevo.surrogate.base import Surrogate

log = logging.getLogger("openarchevo.outer_loop")


def read_metrics_csv(path: Path) -> dict[str, dict[str, float]]:
    """CSV with an ``id`` column (program file stem) and one column per metric."""
    with Path(path).open(newline="") as f:
        return {row.pop("id"): {k: float(v) for k, v in row.items() if v not in ("", None)} for row in csv.DictReader(f)}


def init_archive(programs: Sequence[Path], metrics: Mapping[str, Mapping[str, float]], surrogate: Surrogate,
                 embedder: TextEmbedder, fitness: Callable[[Mapping[str, float]], float],
                 rationale_dir: Optional[Path] = None) -> Archive:
    archive = Archive()
    for path in programs:
        cid = path.stem
        if cid not in metrics:
            raise KeyError(f"No measured metrics for program '{cid}'.")
        code = path.read_text(encoding="utf-8")
        rationale_file = rationale_dir / f"{cid}.txt" if rationale_dir else None
        rationale = rationale_file.read_text(encoding="utf-8").strip() if rationale_file and rationale_file.exists() else ""
        if rationale_dir and not rationale:
            log.warning("No rationale for '%s' (%s); its rationale view stays empty.", cid, rationale_file)
        measured = dict(metrics[cid])
        archive.add(Candidate(
            id=cid, code=code, rationale=rationale, name=cid, fingerprint=surrogate.fingerprint(code),
            embedding=embedder.embed(rationale) if rationale else None,
            measured=measured, fitness=fitness(measured), round=0))
    return archive
