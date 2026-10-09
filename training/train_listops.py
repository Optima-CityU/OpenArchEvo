"""Proxy training on ListOps (Long Range Arena), one third of the training data.

    python training/train_listops.py --program path/to/block.py \
        --result_json runs/main/round_01/results/<id>.listops.json

Data: LRA ``basic_{train,val,test}.tsv`` under ``data_dir``. Tokens are split on
whitespace, an ``<eos>`` is appended, and the vocabulary is built from the
*full* training split. The search-time subset is then made by shuffling the
train and validation splits with seed 42 and keeping the first ``int(N / 3)``
examples of each (``Dataset.shuffle(seed=42).select(range(int(N / 3)))``); the
test split is used in full. Reports ``listops_acc`` = test accuracy (percent)
of the checkpoint with the best validation accuracy.
"""

from __future__ import annotations

import copy
import json
import os
import time
from collections import Counter
from itertools import chain

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import Logger, build_model, build_optimizer, mean_fire_rate, parse_args, setup, write_result

N_CLASSES = 10
SPECIALS = ["<pad>", "<unk>", "<eos>"]


def load_data(cfg: dict):
    from datasets import DatasetDict, load_dataset

    cache = os.path.join(cfg["cache_dir"], "listops_processed")
    vocab_path = os.path.join(cache, "vocab.json")
    if os.path.exists(vocab_path):
        return DatasetDict.load_from_disk(cache), json.loads(open(vocab_path).read())
    files = {s: os.path.join(cfg["data_dir"], f"basic_{n}.tsv")
             for s, n in (("train", "train"), ("validation", "val"), ("test", "test"))}
    raw = load_dataset("csv", data_files=files, delimiter="\t", column_names=["source", "target"], skiprows=1)
    tokens = raw.map(lambda e: {"tokens": str(e["source"]).split()}, remove_columns=["source"], num_proc=cfg["num_workers"])
    counts = Counter(chain.from_iterable(tokens["train"]["tokens"]))
    stoi = {t: i for i, t in enumerate(SPECIALS + [t for t, _ in sorted(counts.items(), key=lambda x: x[1], reverse=True)])}

    def numericalize(e):
        return {"input_ids": [stoi.get(t, stoi["<unk>"]) for t in e["tokens"]] + [stoi["<eos>"]], "label": int(e["target"])}

    data = tokens.map(numericalize, remove_columns=["tokens", "target"], num_proc=cfg["num_workers"])
    data.save_to_disk(cache)
    with open(vocab_path, "w") as f:
        json.dump(stoi, f)
    return data, stoi


def subset(split, fraction: float, seed: int):
    return split.shuffle(seed=seed).select(range(int(len(split) * fraction)))


def run_epoch(model, loader, device, optimizer=None, scheduler=None, grad_clip=0.0) -> dict:
    train = optimizer is not None
    model.train(train)
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total, rates = 0.0, 0, 0, []
    with torch.set_grad_enabled(train):
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits, reg_loss, spikes = model(inputs)
            loss = criterion(logits, targets)
            if train:
                if reg_loss is not None:
                    loss = loss + reg_loss
                optimizer.zero_grad()
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()
            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == targets).sum().item()
            total += targets.numel()
            rates.append(mean_fire_rate(spikes))
    return {"loss": total_loss / max(len(loader), 1), "acc": 100.0 * correct / max(total, 1),
            "fire_rate": sum(rates) / max(len(rates), 1)}


def main() -> None:
    args, cfg = parse_args(__doc__.splitlines()[0], "listops")
    device = setup(cfg.get("seed"))
    log = Logger(args, "listops", cfg)
    data, stoi = load_data(cfg)
    data.set_format(type="torch", columns=["input_ids", "label"])
    pad = stoi["<pad>"]

    def collate(batch):
        inputs = nn.utils.rnn.pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=pad)
        return inputs, torch.tensor([b["label"] for b in batch])

    frac, seed = cfg["subset_fraction"], cfg["subset_seed"]
    splits = {"train": subset(data["train"], frac, seed), "validation": subset(data["validation"], frac, seed),
              "test": data["test"]}
    o = cfg["optim"]
    loaders = {s: DataLoader(d, batch_size=o["batch_size"], shuffle=(s == "train"), num_workers=cfg["num_workers"],
                             collate_fn=collate) for s, d in splits.items()}

    model = build_model(args.program, cfg["model"], len(stoi), N_CLASSES, pad, is_lm=False).to(device)
    optimizer, scheduler = build_optimizer(model, o, len(loaders["train"]))
    log(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M | train {len(splits['train'])} examples")

    best = {"acc": -1.0, "epoch": -1, "state": None}
    for epoch in range(o["epochs"]):
        start = time.time()
        tr = run_epoch(model, loaders["train"], device, optimizer, scheduler, o["grad_clip"])
        va = run_epoch(model, loaders["validation"], device)
        if va["acc"] > best["acc"]:
            best = {"acc": va["acc"], "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
        log(f"epoch {epoch + 1:03d} | train acc {tr['acc']:.2f} | val acc {va['acc']:.2f} | "
            f"best val acc {best['acc']:.2f} | fire {tr['fire_rate']:.3f} | {time.time() - start:.0f}s")

    model.load_state_dict(best["state"])
    if args.save_checkpoint:
        torch.save(best["state"], args.output_dir / f"{args.program.stem}.listops.pt")
    te = run_epoch(model, loaders["test"], device)
    log(f"best val acc {best['acc']:.2f} (epoch {best['epoch'] + 1}) | test acc {te['acc']:.2f}")
    write_result(args, "listops", {"listops_acc": te["acc"]},
                 {"best_val_acc": best["acc"], "best_epoch": best["epoch"] + 1, "fire_rate_test": te["fire_rate"]})


if __name__ == "__main__":
    main()
