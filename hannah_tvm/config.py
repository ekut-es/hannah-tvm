import importlib
import os

from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf


@dataclass
class HardwareParams:
    """Board hardware params used for auto_scheduler"""

    num_cores: int = 4
    vector_unit_bytes: int = 64
    cache_line_bytes: int = 64
    max_shared_memory_per_block: int = 65536
    max_local_memory_per_block: int = 2147483647
    max_threads_per_block: int = 1024
    max_vthread_extent: int = 8  # 32 / 4
    warp_size: int = 32


@dataclass
class MicroConfig:
    "MicroTVM configuration"
    compiler: Any
    prefix: Optional[str] = None
    opts: List[str] = field(default_factory=list)
    cflags: List[str] = field(default_factory=list)
    ccflags: List[str] = field(default_factory=list)
    ldflags: List[str] = field(default_factory=list)
    include_dirs: List[str] = field(default_factory=list)
    libs: List[str] = field(default_factory=list)


@dataclass
class Board:
    name: Any = MISSING
    target: Any = "llvm"
    target_host: Any = ""
    tracker: Optional[str] = None
    opencl: bool = True
    cuda: bool = True
    rebuild_runtime: bool = False
    hardware_params: Optional[HardwareParams] = None
    micro: Any = None
    setup: List[str] = field(default_factory=list)
    teardown: List[str] = field(default_factory=list)
    desired_layouts: Optional[Any] = None


@dataclass
class Model:
    file: str = MISSING
    input_shapes: Any = None


@dataclass
class TunerConfig:
    name: str = MISSING
    task_budget: int = 4


@dataclass
class Config:
    board: Dict[str, Board] = MISSING
    model: Dict[str, Model] = MISSING
    tuner: Optional[Dict[str, TunerConfig]] = None


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)


# Custom OmegaConf resolvers
OmegaConf.register_new_resolver(
    "models_dir",
    lambda: os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"),
)

OmegaConf.register_new_resolver(
    "template_dir",
    lambda: os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "template"),
)


def find_tvm_root():
    spec = importlib.util.find_spec("tvm")
    origin = spec.origin
    return os.path.abspath(os.path.join(os.path.dirname(origin), "..", ".."))


OmegaConf.register_new_resolver("tvm_root", find_tvm_root)


OmegaConf.register_new_resolver(
    "db_dir",
    lambda: os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "database"),
)
