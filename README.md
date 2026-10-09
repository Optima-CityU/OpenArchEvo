# OpenArchEvo

**Architecture search beyond the search space.** An LLM is the mutation operator
of a population-based evolutionary search over executable architecture code;
a surrogate and three-view novelty decide which programs survive.

[Project page](https://thuroi0787.github.io/OpenArchEvo/) · [Paper (arXiv:2609.40258)](https://arxiv.org/abs/2609.40258) · [Architecture zoo](zoo/)

<p align="center"><img src="assets/pipeline.png" width="92%" alt="OpenArchEvo pipeline: inner loop with LLM, islands and surrogate; outer loop with selection, real training and archive update"></p>

In the paper, OpenArchEvo discovers native spiking sequence blocks for language
modelling. The pipeline itself is problem-agnostic: anything you can write as
a Python program and train on a proxy task can be searched.

## How it works

1. **Inner loop.** On islands of related programs, the LLM rewrites parent programs (code + design rationale); thousands of candidates per iteration, none trained.
2. **Outer loop.** Picks a small batch (≤16) by predicted fitness and novelty, hands it off for real training, and adds the measured results to the archive.
3. **Surrogate.** TabPFN, refitted on the archive each iteration, predicts task metrics for every candidate in the inner loop.
4. **Three-view representation.** Each program is compared by its code, its rationale and a *behavioural fingerprint* (21 statistics measured once at initialisation); this drives novelty and duplicate rejection.
5. **Proxy training.** Short, fixed-protocol training runs (here WikiText-2 and ListOps) give the measured fitness.

Details: [docs/pipeline.md](docs/pipeline.md).

## Quickstart (CPU, no GPU or LLM)

```bash
git clone https://github.com/THUROI0787/OpenArchEvo && cd OpenArchEvo
pip install -e ".[dev]"
python scripts/run_round.py --run-dir runs/demo --dry-run --auto
```

`--dry-run` replaces the LLM, surrogate and training with offline fakes and runs
three small outer iterations in seconds; inspect `runs/demo/` to see what each
stage writes. For a real search (LLM endpoint, GPU surrogate service, training
machines) follow [docs/real-run.md](docs/real-run.md).

## Use it for your own problem

* **Task:** write a training script that takes `--program` and `--result_json` and writes `{"metrics": {...}}`; add a fitness mode in [`openarchevo/fitness.py`](openarchevo/fitness.py).
* **Search space:** edit the program contract in [`openarchevo/contract.py`](openarchevo/contract.py) and the prompt templates in [`openarchevo/llm/templates/`](openarchevo/llm/templates/); seed the archive with a few hand-written programs.
* **Fingerprint:** replace the spiking-specific probes in [`openarchevo/fingerprint/`](openarchevo/fingerprint/) with ones that suit your models.
* **LLM / surrogate:** any OpenAI-compatible endpoint works from [`configs/search.yaml`](configs/search.yaml); other backends implement one small interface.
* Keep the dry run passing after every change. Guide: [docs/extending.md](docs/extending.md).

## Architecture zoo

Source code of every architecture trained in the paper's search, with metrics in [`zoo/index.csv`](zoo/index.csv). See [zoo/README.md](zoo/README.md).

| Folder | Files | Contents |
|---|---|---|
| [`zoo/experts/`](zoo/experts/) | 20 | Initial archive: spiking versions of published linear-attention / SSM families |
| [`zoo/discovered/`](zoo/discovered/) | 151 | Programs discovered and trained during the 10 search iterations, with LLM rationales |
| [`zoo/elites/`](zoo/elites/) | 3 | The three architectures analysed in the paper: NeuroGate, HomeoResSSM, LoopMem |
| [`zoo/asi_arch/`](zoo/asi_arch/) | 152 | ASI-Arch-derived programs from an earlier study (separate protocol) |

## Citation

```bibtex
@article{zhao2026openarchevo,
  title   = {Large Language Model-Guided Evolutionary Discovery of Native Neural Architectures for Spiking Sequence Modeling},
  author  = {Zhao, Ruoyu and Wu, Jiaqi and Zhu, Chenyu and Lu, Zhichao},
  journal = {arXiv preprint arXiv:2609.40258},
  year    = {2026},
  url     = {https://arxiv.org/abs/2609.40258}
}
```

Ruoyu Zhao and Jiaqi Wu contributed equally. Apache License 2.0 ([LICENSE](LICENSE));
third-party code, weights and datasets: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
No model weights or datasets are included.
