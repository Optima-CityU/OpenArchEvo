"""Shared pieces of the proxy-training scripts.

Both tasks wrap the candidate block (``openarchevo.contract``) in the fixed
``openarchevo.models.SequenceModel`` and train with AdamW, linear warmup and
cosine decay, adding the block's returned regularisation loss to the task loss.
Each run writes a small JSON result consumed by ``scripts/ingest_results.py``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # so that `openarchevo` imports without installation

from openarchevo.contract import BLOCK_CLASS  # noqa: E402
from openarchevo.models import SequenceModel, load_block_class  # noqa: E402


def parse_args(description: str, task: str) -> tuple[argparse.Namespace, dict[str, Any]]:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--program", required=True, type=Path, help=f"Python file defining {BLOCK_CLASS}.")
    p.add_argument("--config", default=REPO_ROOT / "configs" / "training" / f"{task}.yaml", type=Path)
    p.add_argument("--result_json", type=Path, help="Where to write the metrics (default: <output_dir>/<id>.<task>.json).")
    p.add_argument("--output_dir", default=Path("./outputs"), type=Path, help="Logs (and checkpoints with --save_checkpoint).")
    p.add_argument("--data_dir", type=str, help="Override data_dir from the config.")
    p.add_argument("--neuron_type", choices=["sdn", "bptt", "sltt"], help="Override model.neuron_type.")
    p.add_argument("--epochs", type=int, help="Override optim.epochs (e.g. for a quick check).")
    p.add_argument("--seed", type=int, help="Override seed.")
    p.add_argument("--save_checkpoint", action="store_true")
    args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if args.data_dir:
        cfg["data_dir"] = args.data_dir
    if args.neuron_type:
        cfg["model"]["neuron_type"] = args.neuron_type
    if args.epochs:
        cfg["optim"]["epochs"] = args.epochs
    if args.seed is not None:
        cfg["seed"] = args.seed
    return args, cfg


def setup(seed: Optional[int]) -> str:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_model(program: Path, model_cfg: dict[str, Any], vocab_size: int, n_outputs: int,
                pad_token_id: int, is_lm: bool) -> SequenceModel:
    block_cls = load_block_class(program)
    return SequenceModel(block_cls, vocab_size=vocab_size, n_outputs=n_outputs, pad_token_id=pad_token_id,
                         is_lm=is_lm, **model_cfg)


def build_optimizer(model: torch.nn.Module, optim_cfg: dict[str, Any], steps_per_epoch: int):
    from transformers import get_cosine_schedule_with_warmup

    optimizer = torch.optim.AdamW(model.parameters(), lr=optim_cfg["lr"], weight_decay=optim_cfg["weight_decay"],
                                  betas=tuple(optim_cfg["betas"]))
    scheduler = get_cosine_schedule_with_warmup(optimizer, optim_cfg["warmup_steps"],
                                                steps_per_epoch * optim_cfg["epochs"])
    return optimizer, scheduler


def mean_fire_rate(spikes: list[torch.Tensor]) -> float:
    return torch.stack([s.float().mean() for s in spikes]).mean().item() if spikes else 0.0


def write_result(args: argparse.Namespace, task: str, metrics: dict[str, float], details: dict[str, Any]) -> Path:
    cand_id = args.program.stem
    path = args.result_json or args.output_dir / f"{cand_id}.{task}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"id": cand_id, "task": task, "metrics": metrics, "details": details}, indent=1))
    print(f"Wrote {path}: {metrics}")
    return path


class Logger:
    def __init__(self, args: argparse.Namespace, task: str, cfg: dict[str, Any]):
        args.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = args.output_dir / f"{args.program.stem}.{task}.log"
        self.path.write_text(f"config: {json.dumps(cfg)}\n")

    def __call__(self, line: str) -> None:
        print(line, flush=True)
        with self.path.open("a") as f:
            f.write(line + "\n")
