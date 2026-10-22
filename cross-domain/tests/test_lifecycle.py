"""Shared orchestration and retirement tests; no tensor framework or training."""
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import lifecycle as lc
from pipeline.io import artifact_record, read_json, write_json
from pipeline.worker import marker_path


class Fixture:
    def __init__(self, root):
        self.output_dir = root / "output"
        self.config = {"dataset": "fixture", "experiment": "lifecycle_test", "log_root": str(root / "logs"),
                       "upstream": {"variants": ["multi_task"], "seeds": [1, 2]},
                       "refiner": {"seeds": [7, 8]},
                       "workflow": {"mode": "seed_lifecycle", "check_free_space": False, "release_cache": True}}
        self.identity = "fixed-plan"
        self.repo_root = root
        self.events = []
        self.fail_seed = None
        self.backend = SimpleNamespace(validate=lambda c: None, plan=lambda c: {"peak_output_bytes": 100},
            evaluate_seed=self.evaluate, seed_artifacts=self.permanent, cleanup_roots=self.roots,
            cleanup_records=self.cleanup, aggregate=self.aggregate)

    def resolve(self, value):
        return Path(value)

    def module(self, name):
        return self.backend if name == "workflow" else SimpleNamespace(applicable_recipes=lambda c, v: ["embedding", "aux"])

    def roots(self, c, v, s):
        return [self.output_dir / kind / v / f"seed{s}" for kind in ("cache", "scratch")]

    def cache(self, v, s):
        return self.roots(self, v, s)[0] / "b.pt"

    def permanent(self, c, v, s):
        return [self.output_dir / "prepared.json", self.output_dir / "models" / v / f"seed{s}.pt",
                self.output_dir / "metrics" / v / f"seed{s}.json"]

    def cleanup(self, c, v, s):
        return [artifact_record(self.cache(v, s))]

    def evaluate(self, c, v, s):
        lc.verify_seed_ready(c, v, s)
        if s == self.fail_seed:
            raise RuntimeError("interrupted Y evaluation")
        self.events.append(("Y", s))
        path = self.permanent(c, v, s)[-1]
        write_json(path, {"seed": s, "metrics": 0.5})
        return [path]

    def aggregate(self, c):
        lc.verify_receipts(c)
        assert all(("Y", s) in self.events for s in [1, 2])
        self.events.append(("aggregate",))
        return [write_json(self.output_dir / "evaluation.json", {"complete": True})]

    def unit(self, c, unit, gpu, python, retries):
        if lc.unit_complete(c, unit):
            return
        self.events.append((unit, gpu))
        if unit == "prepare":
            target = write_json(self.output_dir / "prepared.json", {"A": [0], "B": [1], "Y": [2]})
        else:
            kind, variant, seed, *_ = unit.split(":")
            seed = int(seed)
            if kind == "upstream":
                if seed == 2 and self.config["workflow"]["release_cache"]:
                    assert not self.cache(variant, 1).exists(), "previous cache must be retired before next upstream"
                target = self.permanent(c, variant, seed)[1]
            elif kind == "cache":
                target = self.cache(variant, seed)
            else:
                target = self.output_dir / "readouts" / (unit.replace(":", "_") + ".pt")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"small fixture, not a tensor")
        write_json(marker_path(c, unit), {"status": "complete", "identity": c.identity, "artifacts": [artifact_record(target)]})

    def execute(self):
        return lc.run(self, ["GPU-fixture"], unit_runner=self.unit,
                      evaluation_runner=lambda c, v, s, gpu, python: lc.evaluate_one(c, v, s))


@pytest.fixture
def fixture(tmp_path):
    return Fixture(tmp_path)


def test_serial_retirement_and_idempotent_resume(fixture):
    fixture.execute()
    assert not fixture.cache("multi_task", 1).exists()
    assert not fixture.cache("multi_task", 2).exists()
    assert all(lc.valid_receipt(fixture, "multi_task", s) for s in [1, 2])
    assert read_json(fixture.output_dir / "lifecycle/complete.json")["status"] == "complete"
    initial_events = list(fixture.events)
    lc.evaluate_one(fixture, "multi_task", 1)  # works after cache marker artifacts are intentionally absent
    fixture.execute()
    assert fixture.events == initial_events + [("aggregate",)]


def test_failure_retains_current_seed_and_blocks_aggregate(fixture):
    fixture.fail_seed = 2
    with pytest.raises(RuntimeError, match="interrupted"):
        fixture.execute()
    assert lc.valid_receipt(fixture, "multi_task", 1)
    assert fixture.cache("multi_task", 2).is_file()
    assert not lc.receipt_path(fixture, "multi_task", 2).exists()
    assert ("aggregate",) not in fixture.events
    fixture.fail_seed = None
    fixture.execute()
    assert fixture.events.count(("Y", 1)) == 1
    assert fixture.events.count(("Y", 2)) == 1


def test_checkpoint_corruption_blocks_retirement(fixture):
    fixture.config["workflow"]["release_cache"] = False
    fixture.execute()
    fixture.permanent(fixture, "multi_task", 1)[1].write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="durable"):
        lc.retire_cache(fixture, "multi_task", 1)
    assert fixture.cache("multi_task", 1).is_file()


def test_modified_cache_and_unknown_files_are_preserved(fixture):
    fixture.config["workflow"]["release_cache"] = False
    fixture.execute()
    cache = fixture.cache("multi_task", 1)
    unknown = cache.with_name("user-notes.txt")
    unknown.write_text("keep me")
    original = cache.read_bytes()
    cache.write_bytes(b"different contents")
    with pytest.raises(ValueError, match="modified"):
        lc.retire_cache(fixture, "multi_task", 1)
    assert cache.exists()
    cache.write_bytes(original)
    lc.retire_cache(fixture, "multi_task", 1)
    lc.retire_cache(fixture, "multi_task", 1)  # partial cleanup/resume allows already absent files
    assert unknown.read_text() == "keep me"


@pytest.mark.parametrize("change", ["escape", "duplicate", "durable_overlap", "identity"])
def test_unsafe_or_stale_receipt_rejected(fixture, change):
    fixture.config["workflow"]["release_cache"] = False
    fixture.execute()
    target = lc.receipt_path(fixture, "multi_task", 1)
    record = read_json(target)
    if change == "escape":
        record["cleanup_records"][0]["path"] = str(fixture.output_dir.parent / "outside.pt")
    elif change == "duplicate":
        record["cleanup_records"] *= 2
    elif change == "durable_overlap":
        record["cleanup_records"][0] = record["artifacts"][0]
    else:
        record["identity"] = "old-plan"
    write_json(target, record)
    with pytest.raises(ValueError):
        lc.retire_cache(fixture, "multi_task", 1)
    assert fixture.cache("multi_task", 1).exists()


def test_incomplete_matrix_cannot_be_sealed(fixture):
    fixture.unit(fixture, "prepare", "0", "unused", 0)
    with pytest.raises(ValueError, match="incomplete"):
        lc.evaluate_one(fixture, "multi_task", 1)
    assert not lc.receipt_path(fixture, "multi_task", 1).exists()


def test_output_space_check_accounts_for_existing_cache(fixture, monkeypatch):
    fixture.config["workflow"].update(check_free_space=True, free_space_reserve_bytes=20)
    monkeypatch.setattr(lc.shutil, "disk_usage", lambda p: SimpleNamespace(free=90))
    with pytest.raises(OSError, match="insufficient"):
        lc.check_output_space(fixture)
    path = fixture.cache("multi_task", 1)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"x" * 40)
    lc.check_output_space(fixture)


def test_gpu_mapping_and_direct_run_validation(fixture, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-A,GPU-B")
    assert lc.gpu_mapping(["1", "0"]) == ["GPU-B", "GPU-A"]
    for invalid in (["2"], ["0", "0"], ["-1"], []):
        with pytest.raises(ValueError):
            lc.gpu_mapping(invalid)
    with pytest.raises(ValueError, match="at least one"):
        lc.run(fixture, [])
