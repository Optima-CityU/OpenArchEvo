#!/usr/bin/env python
"""Start the surrogate service (fingerprint extraction + TabPFN prediction).

Requires PyTorch, FastAPI/uvicorn, TabPFN and ``TABPFN_MODEL_PATH`` pointing to
a TabPFN-2.5 regressor checkpoint that you downloaded from the official source.

    export TABPFN_MODEL_PATH=/path/to/tabpfn-v2.5-regressor.ckpt
    export OPENARCHEVO_SDN_PATH=/path/to/sdn_checkpoint     # only for neuron_type: sdn
    python scripts/start_surrogate.py --config configs/surrogate.yaml

WARNING: the service executes any code posted to it. It listens on 127.0.0.1 by
default; reach it from another machine through an SSH tunnel, e.g.
``ssh -N -L 9000:127.0.0.1:9000 gpu-host``. Any other host is refused unless you
pass --allow-remote and set a shared token in SURROGATE_TOKEN (the search side
sends it from the same variable).
"""

from __future__ import annotations

import argparse
import ipaddress
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")

from openarchevo.config import load_service_config  # noqa: E402


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:  # a host name other than localhost
        return False


def main() -> None:
    doc = __doc__.strip()
    p = argparse.ArgumentParser(description=doc.splitlines()[0], epilog="\n".join(doc.splitlines()[1:]),
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs" / "surrogate.yaml")
    p.add_argument("--host", help="Interface to listen on (default: host in the config, 127.0.0.1).")
    p.add_argument("--port", type=int)
    p.add_argument("--allow-remote", action="store_true",
                   help="Allow a non-loopback host. Requires a token in the variable named by token_env.")
    args = p.parse_args()
    cfg = load_service_config(args.config)
    host = args.host or cfg.host
    token = os.environ.get(cfg.token_env) or None
    if not is_loopback(host):
        if not args.allow_remote:
            sys.exit(f"Refusing to listen on {host}: the service executes any code posted to it. Keep it on "
                     f"127.0.0.1 and use an SSH tunnel, or pass --allow-remote and set {cfg.token_env}.")
        if not token:
            sys.exit(f"--allow-remote needs a shared token: export {cfg.token_env}=<secret> here and on the "
                     "search machine.")
        print(f"WARNING: listening on {host}. Anyone with the token can run code on this machine.", file=sys.stderr)

    import uvicorn

    from openarchevo.surrogate.service import create_app

    uvicorn.run(create_app(cfg, token=token), host=host, port=args.port or cfg.port, workers=1)


if __name__ == "__main__":
    main()
