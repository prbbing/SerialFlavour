"""Configuration, domain-module loading, and experiment isolation."""

from dataclasses import dataclass, field
import importlib
import platform
import re
import sys
from pathlib import Path

from pipeline.io import json_hash, read_json, sha256_file, write_json


@dataclass
class Context:
    config: dict
    domain_root: Path
    config_path: Path
    filters: dict = field(default_factory=dict)

    @property
    def repo_root(self):
        return self.domain_root.parent

    def resolve(self, value):
        path = Path(value).expanduser()
        return (path if path.is_absolute() else self.repo_root / path).resolve()

    @property
    def data_dir(self):
        return self.resolve(self.config["data_root"])

    @property
    def output_dir(self):
        return self.resolve(self.config["output_root"]) / self.config["dataset"] / self.config["experiment"]

    @property
    def config_hash(self):
        return json_hash(self.config)

    @property
    def source_root(self):
        return self.domain_root / "src"

    @property
    def code_hash(self):
        files = list((self.source_root / "pipeline").glob("*.py"))
        files.extend((self.domain_root / "scripts").rglob("*.py"))
        for kind in ("data", "model", "training", "refine", "evaluate", "analysis"):
            files.extend([self.source_root / kind / "__init__.py", self.source_root / kind / f"{self.config['dataset']}.py"])
        return json_hash({str(path.relative_to(self.domain_root)): sha256_file(path) for path in sorted(files)})

    @property
    def identity(self):
        return json_hash({"config": self.config_hash, "code": self.code_hash})

    def module(self, kind):
        return importlib.import_module(f"{kind}.{self.config['dataset']}")

    def initialize(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        manifest = self.output_dir / "run_manifest.json"
        if manifest.exists():
            previous = read_json(manifest)
            if previous["identity"] != self.identity:
                raise ValueError("experiment code/config identity changed; use a new experiment name instead of overwriting its evidence")
            return
        import torch
        import numpy
        write_json(manifest, {
            "identity": self.identity, "config_sha256": self.config_hash,
            "code_sha256": self.code_hash, "config_path": str(self.config_path),
            "data_root": str(self.data_dir), "output_root": str(self.output_dir),
            "python": sys.version, "python_executable": sys.executable,
            "platform": platform.platform(), "torch": torch.__version__,
            "numpy": numpy.__version__, "cuda_available": torch.cuda.is_available(),
        })
        write_json(self.output_dir / "resolved_config.json", self.config)


def load_context(path, domain_root):
    path = Path(path).resolve()
    config = read_json(path)
    for key in ("dataset", "experiment"):
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", config.get(key, "")):
            raise ValueError(f"{key} must be a simple identifier")
    return Context(config, Path(domain_root).resolve(), path)
