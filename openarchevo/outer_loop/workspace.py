"""On-disk layout of a search run.

::

    runs/<name>/
      archive.jsonl              evaluated archive D_t (one Candidate per line)
      front.jsonl                non-dominated archive members (fitness x novelty)
      similarity_cache.json      cached pairwise three-view similarities
      DRY_RUN                    marker: this run uses fakes (created by --dry-run)
      round_01/
        surrogate_context.json   what the surrogate was fitted on
        roots.jsonl              inner-loop roots P_t0
        pool.jsonl               accepted candidates P_t (written as they arrive)
        shortlist.jsonl          P_f after archive screening (meta.group = top | dissim)
        to_train/
          manifest.jsonl         the training batch Q_t
          <id>.py                one program per selected candidate
        results/
          <id>.<task>.json       written by the training scripts
        INGESTED                 marker written by ingest_results
      round_02/ ...
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

_ROUND = re.compile(r"^round_(\d+)$")


class RunDir:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.archive = self.root / "archive.jsonl"
        self.front = self.root / "front.jsonl"
        self.similarity_cache = self.root / "similarity_cache.json"
        self.dry_run_marker = self.root / "DRY_RUN"

    def round_dir(self, t: int) -> Path:
        return self.root / f"round_{t:02d}"

    def batch_dir(self, t: int) -> Path:
        return self.round_dir(t) / "to_train"

    def manifest(self, t: int) -> Path:
        return self.batch_dir(t) / "manifest.jsonl"

    def results_dir(self, t: int) -> Path:
        return self.round_dir(t) / "results"

    def ingested_marker(self, t: int) -> Path:
        return self.round_dir(t) / "INGESTED"

    def rounds(self) -> list[int]:
        if not self.root.exists():
            return []
        return sorted(int(m.group(1)) for p in self.root.iterdir() if p.is_dir() and (m := _ROUND.match(p.name)))

    def pending_round(self) -> Optional[int]:
        """Latest round whose training batch has been written but not ingested."""
        for t in reversed(self.rounds()):
            if self.manifest(t).exists() and not self.ingested_marker(t).exists():
                return t
        return None

    def next_round(self) -> int:
        done = [t for t in self.rounds() if self.ingested_marker(t).exists()]
        return (max(done) + 1) if done else 1
