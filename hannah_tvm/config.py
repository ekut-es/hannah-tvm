from dataclasses import dataclass, field
from typing import Any, Optional

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class HardwareParams:
    num_cores: int = 4
    vector_unit_bytes: int = 64
    cache_line_bytes: int = 64
    max_shared_memory_per_block: int = 65536
    max_local_memory_per_block: int = 2147483647
    max_threads_per_block: int = 1024
    max_vthread_extent: int = 8  # 32 / 4
    warp_size: int = 32


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
    micro: bool = False


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
    board: Board = Board()
    model: Model = Model()
    qconfig: Optional[QConfig] = None


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
