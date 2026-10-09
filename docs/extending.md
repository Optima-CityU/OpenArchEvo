# Extending OpenArchEvo

The repository is a reference implementation. Most users will replace some of
its parts, such as the search space, the tasks, the LLM, the surrogate or the
similarity. Each part sits behind a small interface.

| Part | Interface | Default | Where |
|---|---|---|---|
| Program generator | `ProgramGenerator.propose(system, user) -> Proposal` | OpenAI-compatible chat API | `openarchevo/llm/` |
| Rationale embedder | `TextEmbedder.embed(text) -> list[float]` | OpenAI-compatible embeddings API | `openarchevo/llm/` |
| Surrogate | `Surrogate.fit / fingerprint / predict / evaluate` | HTTP client to the TabPFN service | `openarchevo/surrogate/` |
| Similarity | `SimilarityModel.components(a, b)` | CodeBLEU + rationale cosine + fingerprint cosine | `openarchevo/novelty/` |
| Fitness | `fitness_fn(mode)(metrics) -> float` | composite WT2 + ListOps | `openarchevo/fitness.py` |
| Training | any command that writes `<id>.<task>.json` | `training/train_*.py` | `training/` |

`openarchevo/components.py` connects the parts. To swap one, edit
`build_components`, or build a `Components` object yourself and call
`run_round(cfg, run, comps, t)`.

A practical order for a new project:

1. Get the dry run working (`scripts/run_round.py --dry-run --auto`) after each change.
2. Change the task: training script, metric names and fitness mode.
3. Change the search space: program contract, prompts and a few seed programs.
4. Adapt the fingerprint to your models, then recalibrate its bounds.
5. Swap the LLM or surrogate only if the defaults do not fit.

## Another LLM

Any provider with an OpenAI-compatible `/chat/completions` endpoint works
without code changes. Set `llm.model` in `configs/search.yaml` and export the
two environment variables it names, by default `OPENAI_BASE_URL` and
`OPENAI_API_KEY`. The embedder works the same way through
`EMBEDDING_BASE_URL` / `EMBEDDING_API_KEY`.

For another API, implement `propose()`. It receives the system prompt and the
filled user prompt, and returns `Proposal(name, rationale, code)`. For several
LLMs, write a generator that picks one per call. The inner loop calls
generators from several threads, so they must be thread-safe.

## Another search space

The search space is defined in three places.

1. **The program contract.** Programs here define `SpikingFLABlock(d_model,
   num_heads, d_ffn_mult, dropout, neuron_type, prenorm, use_spiking_mlp,
   layer_idx)` with `forward(x) -> (y, reg_loss, spikes)`. All of this lives in
   `openarchevo/contract.py` (`BLOCK_CLASS`, `FIXED_KWARGS`, `block_kwargs()`
   and the docstring), which is imported by `inner_loop/search.py`
   (`static_check`), `models/program.py` (loading), `models/wrapper.py` and
   `fingerprint/extract.py` (building blocks), `surrogate/fake.py` and
   `training/`. To change the contract, edit that module, the prompts (which
   describe it in words) and, if the `forward` output changes,
   `SequenceModel.forward` in `models/wrapper.py` and the probes in
   `fingerprint/probes.py`.
2. **The prompts.** Edit `openarchevo/llm/templates/system.md` and `user.md`.
   They hold all the domain constraints of the paper: use `SpikingLinear`
   before every projection, keep the class name and signatures, stay
   sub-quadratic and causal, and return a flat spike tensor. `{context}` in
   `user.md` is replaced by the parent programs. The templates are kept
   verbatim from the paper, including their WikiText-2/ListOps and spiking
   wording, so they are the first thing to rewrite for a new domain.
3. **The fingerprint.** `fingerprint/extract.py` defines what is measured
   before training. For non-spiking spaces, drop SWSP and FireRate or replace
   them with probes that suit your models. Then update `FEATURE_KEYS` and the
   scaling bounds in `fingerprint/schema.py`. Calibrate the bounds as the 5th
   and 95th percentiles of each feature over a reference set of architectures,
   and keep them fixed during a search.

Seed a new search space with an initial archive of hand-written or published
programs:

```bash
python scripts/init_archive.py --run-dir runs/mine --programs my_experts/ --metrics my_experts.csv
```

## Another task

1. Write a training script that takes `--program` and `--result_json`, trains
   the program, and writes
   `{"id": ..., "task": ..., "metrics": {"<metric_name>": value}}`.
2. Add a fitness mode in `openarchevo/fitness.py`, using one `TaskTransform`
   per metric:
   * pick an anchor value that should map to 0.5;
   * set tolerances for gains and losses, and the sigmoid slopes;
   * say whether higher is better.

   Select the mode with `fitness_mode` in `configs/search.yaml`.
3. The metric names in the fitness mode decide what the surrogate is fitted
   on. List any metric that should be predicted in log space (such as
   perplexity) in `tabpfn.log_targets` in `configs/surrogate.yaml`.
4. Point `outer_loop.train_commands` at the new script if you use `--auto`.
   Commands run from the repository root; `{repo}`, `{program}`, `{id}` and
   `{results_dir}` are filled in.

The dry-run fakes (`surrogate/fake.py`, `outer_loop/dry_run.py`) read the
metric names, anchors and directions from the fitness mode, so
`--dry-run` works with a new mode without changes.

The existing `wt2_only` mode is an example of a single-task search.

## Another surrogate

Implement the four methods of `openarchevo.surrogate.base.Surrogate`. `fit` and
`predict` work on fingerprints and metric dicts. `fingerprint` and `evaluate`
take source code, and must raise `InfeasibleProgram` for programs that cannot
run. `FakeSurrogate` is a minimal example. To keep the HTTP service and change
only the regressor, replace `TabPFNPredictor` in `surrogate/tabpfn_model.py`
with any object that has `fit`, `predict`, `version` and `models`.

## Another similarity

`SimilarityModel` caches pairwise view similarities. To add or replace a view,
change `_fill()` (which computes and caches the per-view values) and
`components()`. Novelty, near-duplicate screening, island assignment and
the shortlist all use `similarity()` and `is_near_duplicate()`, so they adapt
automatically. Set `novelty.code_backend: difflib` to drop the CodeBLEU and
tree-sitter dependency.

## Running archived programs

Programs in `zoo/` use the import paths of the original code base
(`src.models.spike.*`, `src.models.nn.dropout`). `openarchevo.models.load_block_class`
and the training scripts map these onto `openarchevo.models`, so archived
programs load unchanged; in your own code, call
`openarchevo.models.install_legacy_spike_alias()` before importing such a file.
Most archived programs also need `flash-linear-attention` and `einops`
(`pip install -e ".[zoo]"`). More in [zoo/README.md](../zoo/README.md).
