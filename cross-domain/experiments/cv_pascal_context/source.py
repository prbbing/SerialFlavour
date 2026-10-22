"""Verify pinned author sources and import them under a private namespace.

Vendored bytes stay unchanged. The only runtime compatibility edits are NumPy
scalar aliases, Torchvision's moved URL loader, and absolute internal imports.
No import/build operation fetches code, datasets, or weights.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parent
COMMIT = "ed3d9ac1c35493d8d22c4e33574919c326d77b6d"
LOCK_SHA256 = "074fa14925d4d5ac02c07971b000c4e17953133833ac2a4e969286d38c9d5439"
PREFIX = "_serialflavour_pascal_author"


def verify_source():
    lock_path = ROOT / "source_lock.json"
    if hashlib.sha256(lock_path.read_bytes()).hexdigest() != LOCK_SHA256:
        raise ValueError("author source lock changed; create a new experiment identity")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock["commit"] != COMMIT:
        raise ValueError("author commit mismatch")
    vendor = ROOT / "vendor" / "author"
    for name, record in lock["files"].items():
        blob = (vendor / name).read_bytes()
        if hashlib.sha256(blob).hexdigest() != record["sha256"]:
            raise ValueError(f"pinned author source corrupted: {name}")
    return {"commit": COMMIT, "lock_sha256": LOCK_SHA256,
            "license": lock["license"], "repository": lock["repository"]}


def author_module(name):
    verify_source()
    full = PREFIX + "." + name
    if full in sys.modules:
        return sys.modules[full]
    vendor = ROOT / "vendor" / "author"
    for package in (PREFIX, PREFIX + ".models", PREFIX + ".data", PREFIX + ".losses", PREFIX + ".utils"):
        if package not in sys.modules:
            module = types.ModuleType(package)
            module.__path__ = [str(vendor / package.split(".")[-1])] if "." in package else [str(vendor)]
            sys.modules[package] = module
    # The author's hard-coded PROJECT_ROOT_DIR is platform dependent. Replace
    # only this path service; the vendored file remains unmodified and hashed.
    path_name = PREFIX + ".utils.mypath"
    if path_name not in sys.modules:
        path_module = types.ModuleType(path_name)
        path_module.PROJECT_ROOT_DIR = str(vendor)
        path_module.MyPath = type("MyPath", (), {"db_root_dir": staticmethod(lambda database="": "")})
        sys.modules[path_name] = path_module
    dependencies = {"models.mti_net": ["models.resnet", "models.layers", "models.padnet"],
                    "models.padnet": ["models.resnet", "models.layers"]}
    for dependency in dependencies.get(name, []):
        author_module(dependency)
    path = vendor / (name.replace(".", "/") + ".py")
    text = path.read_text(encoding="utf-8")
    text = text.replace("from termcolor import colored", "try:\n    from termcolor import colored\nexcept ImportError:\n    def colored(text, *args, **kwargs): return text")
    text = text.replace("from models.", "from " + PREFIX + ".models.")
    text = text.replace("from utils.mypath", "from " + PREFIX + ".utils.mypath")
    text = text.replace("from torchvision.models.utils import load_state_dict_from_url", "from torch.hub import load_state_dict_from_url")
    text = text.replace("np.int(", "int(").replace("dtype=np.float)", "dtype=float)")
    module = types.ModuleType(full)
    module.__file__ = str(path)
    module.__package__ = full.rsplit(".", 1)[0]
    sys.modules[full] = module
    try:
        exec(compile(text, str(path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(full, None)
        raise
    return module
