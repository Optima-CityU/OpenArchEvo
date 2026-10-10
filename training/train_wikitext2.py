"""Proxy training on WikiText-2 language modelling.

    python training/train_wikitext2.py --program path/to/block.py \
        --result_json runs/main/round_01/results/<id>.wikitext2.json

Data: ``wiki.{train,valid,test}.tokens`` under ``data_dir``; GPT-2 BPE tokens,
concatenated and cut into 512-token blocks (processed once and cached).
Reports ``wikitext2_ppl`` = best validation perplexity (the search metric) and,
in ``details``, the test perplexity of that checkpoint.
"""

from __future__ import annotations

import copy
import math
import os
import time
from itertools import chain

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import Logger, build_model, build_optimizer, mean_fire_rate, parse_args, setup, write_result

VOCAB_SIZE, PAD_ID = 50257, 50256  # GPT-2


def load_data(cfg: dict):
    from datasets import DatasetDict, load_dataset
    from transformers import AutoTokenizer

    cache = os.path.join(cfg["cache_dir"], f"wikitext2_{os.path.basename(cfg['tokenizer'])}_seq{cfg['seq_len']}")
    if os.path.exists(cache):
        return DatasetDict.load_from_disk(cache)
    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"])
    files = {s: os.path.join(cfg["data_dir"], f"wiki.{n}.tokens")
             for s, n in (("train", "train"), ("validation", "valid"), ("test", "test"))}
    raw = load_dataset("text", data_files=files)
    tokenized = raw.map(lambda b: tokenizer(b["text"]), batched=True, num_proc=cfg["num_workers"], remove_columns=["text"])

    def group(batch):
        ids = list(chain(*batch["input_ids"]))
        n = len(ids) // cfg["seq_len"] * cfg["seq_len"]
        return {"input_ids": [ids[i:i + cfg["seq_len"]] for i in range(0, n, cfg["seq_len"])]}

    data = tokenized.map(group, batched=True, num_proc=cfg["num_workers"], remove_columns=tokenized["train"].column_names)
    data.save_to_disk(cache)
    return data


def run_epoch(model, loader, device, optimizer=None, scheduler=None, grad_clip=0.0) -> dict:
    train = optimizer is not None
    model.train(train)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    total_loss, total_tokens, rates = 0.0, 0, []
    with torch.set_grad_enabled(train):
        for batch in loader:
            chunk = batch["input_ids"].to(device)
            inputs, targets = chunk[:, :-1], chunk[:, 1:]
            logits, reg_loss, spikes = model(inputs)
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
            if train:
                if reg_loss is not None:
                    loss = loss + reg_loss
                optimizer.zero_grad()
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()
            n = (targets != PAD_ID).sum().item()
            total_loss += loss.item() * n
            total_tokens += n
            rates.append(mean_fire_rate(spikes))
    loss = total_loss / max(total_tokens, 1)
    return {"loss": loss, "ppl": math.exp(min(loss, 50)), "fire_rate": sum(rates) / max(len(rates), 1)}


def main() -> None:
    args, cfg = parse_args(__doc__.splitlines()[0], "wikitext2")
    device = setup(cfg.get("seed"))
    log = Logger(args, "wikitext2", cfg)
    data = load_data(cfg)
    data.set_format(type="torch", columns=["input_ids"])
    o = cfg["optim"]
    loaders = {s: DataLoader(data[s], batch_size=o["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
               for s in ("train", "validation", "test")}

    model = build_model(args.program, cfg["model"], VOCAB_SIZE, VOCAB_SIZE, PAD_ID,
                        is_lm=True, device=device).to(device)
    optimizer, scheduler = build_optimizer(model, o, len(loaders["train"]))
    log(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    best = {"loss": float("inf"), "epoch": -1, "state": None}
    for epoch in range(o["epochs"]):
        start = time.time()
        tr = run_epoch(model, loaders["train"], device, optimizer, scheduler, o["grad_clip"])
        va = run_epoch(model, loaders["validation"], device)
        if va["loss"] < best["loss"]:
            best = {"loss": va["loss"], "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
        log(f"epoch {epoch + 1:03d} | train ppl {tr['ppl']:.2f} | val ppl {va['ppl']:.2f} | "
            f"best val ppl {math.exp(min(best['loss'], 50)):.2f} | fire {tr['fire_rate']:.3f} | {time.time() - start:.0f}s")

    if best["state"] is not None:
        model.load_state_dict(best["state"])
        if args.save_checkpoint:
            torch.save(best["state"], args.output_dir / f"{args.program.stem}.wikitext2.pt")
    te = run_epoch(model, loaders["test"], device)
    val_ppl = math.exp(min(best["loss"], 50))
    log(f"best val ppl {val_ppl:.2f} (epoch {best['epoch'] + 1}) | test ppl {te['ppl']:.2f}")
    write_result(args, "wikitext2", {"wikitext2_ppl": val_ppl},
                 {"test_ppl": te["ppl"], "best_epoch": best["epoch"] + 1, "fire_rate_test": te["fire_rate"]})


if __name__ == "__main__":
    main()
