#
# Copyright (c) 2022 University of Tübingen.
#
# This file is part of hannah-tvm.
# See https://atreus.informatik.uni-tuebingen.de/ties/ai/hannah/hannah-tvm for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import importlib
import os
import socket
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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
class ExecutorConfig:
    name: str
    options: Dict[str, Any]


@dataclass
class RuntimeConfig:
    name: str
    options: Dict[str, Any]


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
    build: Dict[str, Any] = field(default_factory=dict)
    executor: Optional[ExecutorConfig] = None
    runtime: Optional[RuntimeConfig] = None
    micro: Any = None
    setup: List[str] = field(default_factory=list)
    teardown: List[str] = field(default_factory=list)
    desired_layouts: Optional[Any] = None
    rpc_runner: Optional[str] = None
    disable_vectorize: Optional[bool] = None
    connector: str = "default"


@dataclass
class Model:
    file: str = MISSING
    input_shapes: Any = None  # Input shapes for models from sources that do not encode input shapes e.g. PyTorch/TorchScript


@dataclass
class TunerConfig:
    name: str = MISSING
    task_budget: int = 4
    mode: str = "xgb"
    equal_task_budget: bool = False  # Run same amount of tuning for each task (only used for auto_scheduler/meta_scheduler)


@dataclass
class Config:
    board: Board = MISSING
    model: Dict[str, Model] = MISSING
    tuner: Optional[TunerConfig] = None


@dataclass
class BackendConfig:
    _target_: str = "hannah_tvm.backend.TVMBackend"
    val_batches: int = 1
    test_batches: int = 1
    val_frequency: int = 1
    board: Board = MISSING
    tuner: Optional[TunerConfig] = None


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)

cs.store(group="backend", name="base_tvm", node=BackendConfig)


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
    if not spec:
        return None
    origin = spec.origin
    return os.path.abspath(os.path.join(os.path.dirname(origin), "..", ".."))


OmegaConf.register_new_resolver("tvm_root", find_tvm_root)


OmegaConf.register_new_resolver(
    "db_dir",
    lambda: os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "database"),
)


OmegaConf.register_new_resolver("hostname", lambda: socket.gethostname())
