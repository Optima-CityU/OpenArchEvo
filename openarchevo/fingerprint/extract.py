"""Extract the 21-dim behavioural fingerprint of a program (requires PyTorch).

``extract_fingerprint`` builds the fixed probing wrapper around the program's
block class (:mod:`openarchevo.contract`), computes the seven probes on a fixed minibatch and the
fourteen structural statistics on a single block. ``extract_in_subprocess``
does the same in a fresh process with a timeout, which is how the surrogate
service runs untrusted LLM-written code.
"""

from __future__ import annotations

import multiprocessing as mp
import random
import traceback
from pathlib import Path

import numpy as np

from openarchevo.config import FingerprintConfig
from openarchevo.contract import block_kwargs
from openarchevo.fingerprint.schema import FEATURE_KEYS


class FingerprintError(RuntimeError):
    """The program could not be built, run or probed."""


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def probe_batch(cfg: FingerprintConfig, device: str):
    """The fixed probe minibatch ``(tokens, targets)``.

    Uses ``probe.batch_path`` (a ``torch.save``-d tuple) when given; otherwise a
    synthetic, seeded batch with the ListOps vocabulary size and length. Keep the
    batch fixed across a whole search: fingerprints are only comparable when
    computed on the same probe.
    """
    import torch

    p = cfg.probe
    if p.batch_path:
        tokens, targets = torch.load(Path(p.batch_path), weights_only=True, map_location="cpu")[:2]
    else:
        g = torch.Generator().manual_seed(cfg.seed)
        tokens = torch.randint(1, p.vocab_size, (p.batch_size, p.seq_len), generator=g)
        targets = torch.randint(0, p.n_classes, (p.batch_size,), generator=g)
    return tokens.to(device), targets.to(device)


def extract_fingerprint(code: str, cfg: FingerprintConfig, device: str = "cuda:0") -> list[float]:
    """Return the fingerprint of ``code`` in ``FEATURE_KEYS`` order."""
    import torch

    from openarchevo.fingerprint.probes import compute_probes
    from openarchevo.fingerprint.structure import structural_features
    from openarchevo.models import SequenceModel, load_block_class

    # Search-time feasibility: run the program as the original search did, with whatever
    # optional imports its own try/except fallbacks handle (no strict dependency check).
    block_cls = load_block_class(code, check_imports=False)
    w, p = cfg.wrapper, cfg.probe

    set_seed(cfg.seed)
    model = SequenceModel(block_cls, vocab_size=p.vocab_size, n_outputs=p.n_classes, d_model=w.d_model,
                          n_layers=w.n_layers, num_heads=w.num_heads, d_ffn_mult=w.d_ffn_mult,
                          dropout=w.dropout, neuron_type=cfg.neuron_type).to(device)
    tokens, targets = probe_batch(cfg, device)
    features = compute_probes(model, tokens, targets)
    del model

    set_seed(cfg.seed)
    block = block_cls(**block_kwargs(w.d_model, w.num_heads, w.d_ffn_mult, 0.0, cfg.neuron_type)).to(device)
    x = torch.randn(1, cfg.structure_seq_len, w.d_model, generator=torch.Generator().manual_seed(cfg.seed)).to(device)
    features.update(structural_features(block, x))

    vector = [float(features[k]) for k in FEATURE_KEYS]
    if not np.all(np.isfinite(vector)):
        raise FingerprintError("Non-finite fingerprint value.")
    return vector


def _child(code: str, cfg: FingerprintConfig, device: str, conn) -> None:
    try:
        conn.send(("ok", extract_fingerprint(code, cfg, device)))
    except BaseException as exc:  # report everything, including CUDA OOM
        conn.send(("error", f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=5)}"))
    finally:
        conn.close()


def extract_in_subprocess(code: str, cfg: FingerprintConfig, device: str) -> list[float]:
    """Run :func:`extract_fingerprint` in a spawned process; kill it after ``cfg.timeout_s``."""
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_child, args=(code, cfg, device, child), daemon=True)
    proc.start()
    child.close()
    try:
        if not parent.poll(cfg.timeout_s):
            raise FingerprintError(f"Timeout after {cfg.timeout_s:.0f}s")
        status, payload = parent.recv()
    except EOFError as exc:
        raise FingerprintError("Probe process exited without a result (crash).") from exc
    finally:
        if proc.is_alive():
            proc.terminate()
        proc.join()
        parent.close()
    if status != "ok":
        raise FingerprintError(payload)
    return payload
