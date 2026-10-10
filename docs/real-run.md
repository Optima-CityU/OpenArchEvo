# Running a real search

A real search needs an LLM, an embedding model, the surrogate service (GPU) and
one or more machines for proxy training. Try the dry run first
(`python scripts/run_round.py --run-dir runs/demo --dry-run --auto`).

## 1. Install

No weights or datasets are shipped.

```bash
pip install -e ".[codebleu,service,train,zoo]"   # install a torch wheel for your CUDA version first
```

## 2. Start the surrogate service (GPU machine)

The service fingerprints programs and runs TabPFN.

```bash
export TABPFN_MODEL_PATH=/path/to/tabpfn-v2.5-regressor.ckpt   # from Prior Labs (non-commercial weights license)
export OPENARCHEVO_SDN_PATH=/path/to/sdn_checkpoint            # SpikingSSM SDN; or set neuron_type: bptt
python scripts/start_surrogate.py --config configs/surrogate.yaml   # listens on 127.0.0.1:9000
```

**Security.** The service executes any code posted to it; the per-program
subprocess is not a sandbox. Keep it on 127.0.0.1 and reach it from the search
machine through an SSH tunnel (`ssh -N -L 9000:127.0.0.1:9000 gpu-host`).
`start_surrogate.py` refuses other hosts unless you pass `--allow-remote` and
set a shared token in `SURROGATE_TOKEN` (export the same token on the search
side; the client sends it as a bearer token). The service holds one fitted
model, so run one service per search.

## 3. Point the search at your endpoints

Any OpenAI-compatible API works. `configs/search.yaml` stores only the *names*
of these variables.

```bash
export OPENAI_BASE_URL=...    OPENAI_API_KEY=...       # program generator (llm:)
export EMBEDDING_BASE_URL=... EMBEDDING_API_KEY=...    # rationale embeddings (embedder:)
export SURROGATE_URL=http://127.0.0.1:9000
```

Scripts check these services before starting and exit with a hint if one does
not answer. During a search, `inner_loop.max_consecutive_errors` errors of one
kind in a row stop the round.

## 4. Create the initial archive

```bash
python scripts/init_archive.py --run-dir runs/main --programs zoo/experts \
    --metrics examples/expert_metrics.csv --rationales zoo/experts/rationales
```

Without `--rationales` the seed programs have no rationale view, which weakens
three-view novelty; the script warns about each program without one.

## 5. Repeat round, training hand-off, ingest

All commands run from the repository root.

```bash
python scripts/run_round.py --run-dir runs/main        # search; writes runs/main/round_XX/to_train/ and stops
python training/train_wikitext2.py --program runs/main/round_XX/to_train/<id>.py \
    --result_json runs/main/round_XX/results/<id>.wikitext2.json
python training/train_listops.py   --program runs/main/round_XX/to_train/<id>.py \
    --result_json runs/main/round_XX/results/<id>.listops.json
python scripts/ingest_results.py --run-dir runs/main   # adds measured metrics and fitness to archive.jsonl
```

To train on another machine, copy `runs/main/round_XX/` there with the same
relative path, run the training commands, and copy `results/` back. If search
and training run on one machine, `run_round.py --auto` runs
`outer_loop.train_commands` from `configs/search.yaml`, ingests and continues.
Without `--auto`, you can rehearse the hand-off offline with
`python scripts/ingest_results.py --run-dir runs/demo --dry-run --fake-train`.

Data formats and the run-directory layout: [pipeline.md](pipeline.md).

Causality checks run automatically during candidate evaluation and before
WikiText-2 training. They are necessary to prevent reward hacking through
future-token leakage, a common failure mode during evolution that can inflate
scores without improving causal prediction. Candidates that fail are rejected.
