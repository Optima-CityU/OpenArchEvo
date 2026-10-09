"""End-to-end dry run of the search (fake LLM, embedder, surrogate and training)."""

import subprocess
import sys
from pathlib import Path

from openarchevo.candidate import Archive, read_jsonl
from openarchevo.outer_loop.workspace import RunDir

ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)


def test_auto_rounds(tmp_path):
    out = _run("scripts/run_round.py", "--run-dir", str(tmp_path), "--dry-run", "--auto", "--rounds", "2")
    assert out.returncode == 0, out.stderr
    run = RunDir(tmp_path)
    archive = Archive.load(run.archive)
    assert len(archive.initial()) == 3 and archive.new_evaluations() == 6
    assert all(c.fitness is not None and c.fingerprint for c in archive)
    assert read_jsonl(run.front)


def test_manual_handoff(tmp_path):
    assert _run("scripts/run_round.py", "--run-dir", str(tmp_path), "--dry-run").returncode == 0
    run = RunDir(tmp_path)
    assert run.pending_round() == 1 and len(list(run.batch_dir(1).glob("*.py"))) == 3
    # A new round is refused until results are ingested.
    assert _run("scripts/run_round.py", "--run-dir", str(tmp_path), "--dry-run").returncode != 0
    assert _run("scripts/ingest_results.py", "--run-dir", str(tmp_path), "--dry-run").returncode != 0

    from openarchevo.outer_loop.dry_run import fake_training
    from openarchevo.surrogate import FakeSurrogate

    fake_training(run, 1, FakeSurrogate())
    assert _run("scripts/ingest_results.py", "--run-dir", str(tmp_path), "--dry-run").returncode == 0
    assert run.pending_round() is None and run.next_round() == 2


def test_fake_train_flag(tmp_path):
    assert _run("scripts/run_round.py", "--run-dir", str(tmp_path), "--dry-run").returncode == 0
    run = RunDir(tmp_path)
    assert read_jsonl(run.front)  # front exists before the first ingest
    assert _run("scripts/ingest_results.py", "--run-dir", str(tmp_path), "--fake-train").returncode != 0
    out = _run("scripts/ingest_results.py", "--run-dir", str(tmp_path), "--dry-run", "--fake-train")
    assert out.returncode == 0, out.stderr
    assert run.pending_round() is None and Archive.load(run.archive).new_evaluations() == 3


def test_help_has_description():
    for script in ("run_round", "ingest_results", "init_archive", "run_inner_loop"):
        out = _run(f"scripts/{script}.py", "--help")
        assert out.returncode == 0
        assert out.stdout.split("\n\n")[1].strip(), script  # description line after the usage block


def test_dry_and_real_runs_are_not_mixed(tmp_path):
    assert _run("scripts/run_round.py", "--run-dir", str(tmp_path), "--dry-run").returncode == 0
    out = _run("scripts/run_round.py", "--run-dir", str(tmp_path))
    assert out.returncode != 0 and "dry-run" in out.stderr


def test_other_fitness_mode(tmp_path):
    """A fitness mode with other metrics needs no change to the fakes."""
    text = (ROOT / "configs" / "search.yaml").read_text().replace("fitness_mode: composite", "fitness_mode: wt2_only")
    config = tmp_path / "search.yaml"
    config.write_text(text)
    run_dir = tmp_path / "run"
    out = _run("scripts/run_round.py", "--config", str(config), "--run-dir", str(run_dir), "--dry-run", "--auto",
               "--rounds", "1")
    assert out.returncode == 0, out.stderr
    archive = Archive.load(RunDir(run_dir).archive)
    assert archive.new_evaluations() == 3 and all(set(c.measured) == {"wikitext2_ppl"} for c in archive)


def test_training_commands_run_from_repo_root(tmp_path):
    from openarchevo.candidate import Candidate
    from openarchevo.outer_loop.handoff import REPO_ROOT, run_training_commands, write_training_batch

    run = RunDir(tmp_path)
    write_training_batch(run, 1, [Candidate(id="r01-s0001", code="x = 1\n")])
    template = ("{python} -c \"import os, sys; open(sys.argv[1], 'w').write(os.getcwd() + ' ' + sys.argv[2])\" "
                "{results_dir}/{id}.cwd.txt {repo}")
    run_training_commands(run, 1, [template])
    cwd, repo = (run.results_dir(1) / "r01-s0001.cwd.txt").read_text().split(" ")
    assert Path(cwd) == REPO_ROOT == Path(repo)
