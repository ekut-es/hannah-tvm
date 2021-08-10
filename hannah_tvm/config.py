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
    target_host: Any = "llvm"
    tracker: Optional[str] = None
    opencl: bool = True
    cuda: bool = True
    rebuild_runtime: bool = False
    hardware_params: HardwareParams = HardwareParams()
    micro: Optional[MicroConfig] = None
    setup: List[str] = field(default_factory=list)
    teardown: List[str] = field(default_factory=list)


@dataclass
class Model:
    file: str = MISSING
    input_shapes: Any = None


@dataclass
class QConfig:
    engine: str = MISSING
    calibrate_mode: str = "global_scale"
    global_scale: float = 8.0
    weight_scale: str = "max"
    skip_dense_layer: bool = False
    skip_conv_layers: Any = field(default_factory=list)
    do_simulation: bool = False
    round_for_shift: bool = False
    rounding: str = "UPWARD"
    partition_conversions: str = "enabled"


@dataclass
class Config:
    board: Dict[str, Board] = MISSING
    model: Dict[str, Model] = MISSING
    qconfig: Optional[QConfig] = None


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)


# Custom OmegaConf resolvers


OmegaConf.register_resolver(
    "models_dir",
    lambda: os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"),
)


def find_tvm_root():
    spec = importlib.util.find_spec("tvm")
    origin = spec.origin
    return os.path.abspath(os.path.join(os.path.dirname(origin), "..", ".."))


OmegaConf.register_resolver("tvm_root", find_tvm_root)
