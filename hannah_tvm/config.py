from dataclasses import dataclass, field
from typing import Any, Optional

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

@dataclass
class Board:
    name: Any = MISSING
    target: Any = "cuda -arch=sm_62" # "llvm -mtriple=aarch64-linux-gnu -device=arm_cpu -model=jetsontx2"
    target_host: Any = "llvm -mtriple=aarch64-linux-gnu -device=arm_cpu -model=jetsontx2" 
    tracker: Optional[str] = None
    opencl: bool = True
    cuda: bool = True
    rebuild_runtime: bool = False
    

@dataclass
class Config:
    board: Board = Board()
    model: str = MISSING

cs = ConfigStore.instance()
cs.store(name="config", node=Config)