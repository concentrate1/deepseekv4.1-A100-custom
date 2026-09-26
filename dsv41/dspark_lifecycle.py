"""Load an isolated DSpark implementation without reloading the main model.

A unique module name keeps the previous implementation alive until a replacement
has loaded weights and captured its CUDA graph successfully. Only dspark.py is
reloaded; changes to shared kernels or the target runtime still require a full
worker restart.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from uuid import uuid4


def load_dspark_rows_class(*, reload_code: bool):
    if not reload_code:
        from .dspark import DSparkRows
        return DSparkRows, None

    source = Path(__file__).with_name("dspark.py")
    name = f"dsv41._dspark_live_{uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load DSpark source: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        cls = module.DSparkRows
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return cls, name


def discard_dspark_module(name: str | None) -> None:
    if name:
        sys.modules.pop(name, None)
