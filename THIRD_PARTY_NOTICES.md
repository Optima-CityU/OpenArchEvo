# Third-party notices

OpenArchEvo is released under the Apache License 2.0 (see `LICENSE`). This file
lists third-party code that the repository contains or adapts, and the
packages, model weights and datasets it depends on. No third-party model
weights or datasets are included.

Upstream licenses below were read from each project's `LICENSE` file on
2026-10-10. "Not audited" means the derivation was not compared line by line.

## Code contained in this repository

| What | Upstream | Upstream license | Where | Status |
|---|---|---|---|---|
| Spiking neurons (hard-reset LIF, BPTT / SLTT unrolling, SDN wrapper) and the piecewise-quadratic surrogate gradient | SpikingSSM / SDN, Shen et al., AAAI 2025, https://github.com/shenshuaijie/SDN | MIT | `openarchevo/models/neurons.py` | adapted |
| Layer code from flash-linear-attention (FLA). 79 files keep the FLA copyright header `Copyright (c) 2023-2025, Songlin Yang, Yu Zhang` | https://github.com/fla-org/flash-linear-attention | MIT | `zoo/` (1 expert, 61 discovered, 1 elite and 16 ASI-Arch files carry the header) | header kept; other expert programs follow FLA layer structure without the header (not audited) |
| `DropoutNd` (sequence-tied dropout), shim for the legacy `src.models.nn.dropout` import | https://github.com/state-spaces/s4 (`src/models/nn/dropout.py`), as vendored by SpikingSSM | Apache-2.0 (SpikingSSM: MIT) | `openarchevo/models/legacy.py` | adapted |
| S4D kernel and SS4D layer (`S4DKernel`, `SS4D`), labelled "SpikingSSM Components" in the source | SpikingSSM (MIT), which builds on the S4D code of https://github.com/state-spaces/s4 (Apache-2.0) | MIT / Apache-2.0 | `zoo/experts/s4.py`, `zoo/asi_arch/20_SpikingSSM.py`, and discovered programs built on `SS4D` | not audited |
| ASI-Arch token-mixer programs and baselines | ASI-Arch, https://github.com/GAIR-NLP/ASI-Arch | Apache-2.0 | `zoo/asi_arch/` (118 token-mixer programs and baselines, as used in an earlier study) and `zoo/asi_arch/fullspike/` (34 fully spiking LLM rewrites of them) | derived; provenance per file not audited |

Other expert programs (`zoo/experts/`) are spiking versions of published
layers (RetNet, GLA, HGRN, RWKV, Mamba, MesaNet, Comba and others). Their
upstream code sources other than FLA were not audited. `zoo/experts/metala.py`
carries the header `Copyright (c) 2024-2026, Ruoyu Zhao`.

The rest of `openarchevo/`, `training/`, `scripts/` and `examples/` was written
for this project. Some parts follow published algorithms without copying their
code: island-based program search (FunSearch), the dominance-dissimilarity
island score (MEoH), sample-wise activation patterns (SWAP), zero-cost proxies
(SynFlow, GradNorm, SNIP, Jacobian covariance, NASWOT), NSGA-II and novelty
search. The example blocks use RetNet-style decays and HGRN / S4D-style
recurrences, reimplemented from the papers.

## Packages used, not vendored

| Package | Use | License |
|---|---|---|
| [`codebleu`](https://github.com/k4black/codebleu) (from Microsoft CodeXGLUE) | code similarity; called as a library, no code copied | MIT |
| [TabPFN](https://github.com/PriorLabs/TabPFN) | surrogate regressor | package: Apache-2.0 (Prior Labs GmbH; NOTICE: "TabPFN, Copyright (c) 2026 Prior Labs GmbH"). Model weights are licensed **separately**: per the TabPFN README (checked 2026-10-10), the TabPFN-2.5 and later weights are under **non-commercial** licenses that you accept through Prior Labs before download, and the TabPFN-2 weights are under the Prior Labs License (Apache-2.0 plus an attribution requirement). No weights are included here. |
| [PyTorch](https://pytorch.org) | models, probes, training | BSD-3-Clause |
| [flash-linear-attention](https://github.com/fla-org/flash-linear-attention), [einops](https://github.com/arogozhnikov/einops) | imported by most `zoo/` programs | MIT / MIT |
| [FastAPI](https://github.com/fastapi/fastapi), [uvicorn](https://github.com/encode/uvicorn), [httpx](https://github.com/encode/httpx) | surrogate service and client | MIT / BSD-3-Clause / BSD-3-Clause |
| [NumPy](https://numpy.org), [scikit-learn](https://scikit-learn.org), [PyYAML](https://pyyaml.org) | core | BSD-3-Clause / BSD-3-Clause / MIT |
| [Hugging Face `datasets`, `transformers`](https://github.com/huggingface) | data loading and the GPT-2 tokenizer | Apache-2.0 |

## Models and data referenced, not included

* **SDN checkpoint** (`neuron_type: sdn`): from the SpikingSSM authors' release;
  set `OPENARCHEVO_SDN_PATH`. The terms for the checkpoint itself were not
  confirmed separately from the code license.
* **TabPFN-2.5 regressor checkpoint** (used in the paper): from Prior Labs under a
  non-commercial license; set `TABPFN_MODEL_PATH`. Commercial users need a
  license from Prior Labs or another regressor (see `docs/extending.md`).
* **WikiText-2** (Merity et al., 2017; CC BY-SA 3.0) and **ListOps** from Long
  Range Arena (Tay et al., 2021; https://github.com/google-research/long-range-arena,
  Apache-2.0): download them yourself.
* **LLM and embedding APIs** are used under the terms of their providers.

## License texts

The Apache-2.0 upstreams (state-spaces S4, ASI-Arch) use the same license text
as this repository's `LICENSE`; neither ships a `NOTICE` file (checked
2026-10-10). The MIT texts follow.

### flash-linear-attention (MIT)

```text
MIT License

Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### SpikingSSM / SDN (MIT)

```text
MIT License

Copyright (c) 2024 Shuaijie Shen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The Apache-2.0 projects above (state-spaces/s4, ASI-Arch) are covered by the
same license as this repository; see `LICENSE`.
