"""Exercise the production scheduler without CUDA, training or source data."""

import importlib.util
from pathlib import Path


def test_width_queue_reuses_gpus_across_configs_and_resumes(tmp_path, monkeypatch):
    path = Path("scripts/production/run_experiment1_ex_recipe_queue.py")
    spec = importlib.util.spec_from_file_location("width_queue", path)
    queue = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(queue)
    configs = sorted(Path(
        "configs/parallel_refine/experiments/experiment2_v2").glob("*.json"))
    assert len(configs) == 3
    active = set()
    events = []
    max_active = 0

    class Worker:
        def __init__(self, command, **kwargs):
            nonlocal max_active
            assert "--stage" in command, "100/0 must not launch DNN aggregation"
            self.stage = command[command.index("--stage") + 1]
            assert self.stage in {"parallel", "y", "eval", "prune"}
            self.gpu = command[command.index("--gpus") + 1]
            self.config = Path(command[command.index("--config") + 1]).stem
            self.seed = int(command[command.index("--seed") + 1])
            assert self.gpu not in active, "a GPU cannot run two units at once"
            active.add(self.gpu)
            max_active = max(max_active, len(active))
            self.pid = len(events) + 100
            # Keep the first B seed busy while other GPUs advance into F/G.
            self.remaining = (40 if "p142k" in self.config
                              and self.seed == 1 and self.stage == "parallel" else 1)
            events.append(("start", self.config, self.seed, self.stage))

        def poll(self):
            self.remaining -= 1
            if self.remaining >= 0:
                return None
            active.remove(self.gpu)
            events.append(("end", self.config, self.seed, self.stage))
            return 0

    monkeypatch.setattr(queue.subprocess, "Popen", Worker)
    monkeypatch.setattr(queue, "discover_external_parallels", lambda *args: {})
    monkeypatch.setattr(queue.signal, "signal", lambda *args: None)
    monkeypatch.setattr(queue.time, "sleep", lambda *args: None)
    args = [item for config in configs for item in ("--config", str(config))]
    args += ["--gpus", "0", "1", "2", "3", "--max-refine", "3",
             "--log-base", str(tmp_path / "logs"),
             "--state-dir", str(tmp_path / "state")]
    assert queue.main(args) == 0
    assert max_active == 4 and not active
    b1_end = events.index(("end", configs[0].stem, 1, "parallel"))
    assert any(event[0] == "start" and event[1] != configs[0].stem
               and event[3] == "parallel" for event in events[:b1_end])
    for config in configs:
        for seed in range(1, 6):
            assert [stage for event, name, number, stage in events
                    if event == "start" and name == config.stem and number == seed
                    ] == ["parallel", "y", "eval", "prune"]
    assert len(list((tmp_path / "state").glob("*.done"))) == 15
    event_count = len(events)
    assert queue.main(args) == 0
    assert len(events) == event_count, "completed runs should launch no workers"
