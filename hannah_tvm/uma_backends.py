#
# Copyright (c) 2023 hannah-tvm contributors.
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

# Automatically run backend.register for detected uma backends
# TODO: propagate to tvm and write RFC, and upstream

import importlib
import inspect
import logging
import pkgutil
import sys
import warnings
from typing import Any, List, Type

from tvm.relay.backend.contrib.uma.backend import UMABackend

_INITIALIZED = False
_BACKENDS = []

_logger = logging.getLogger(__name__)


def backends():
    if not _INITIALIZED:
        init()

    for backend in _BACKENDS:
        yield backend


def init() -> None:
    top_level = []

    try:
        mod = importlib.import_module("uma_backends")
        top_level.append(mod)
    except ImportError:
        # If no plugins are installed the hydra_plugins package does not exist.
        pass

    backend_classes = _scan_all(top_level)

    for backend_cls in backend_classes:
        backend = backend_cls()
        backend.register()
        _BACKENDS.append(backend)


def _scan_all(top_level: List[Any]) -> List[Type[UMABackend]]:
    scanned_backends: List[Type[UMABackend]] = []
    for mdl in top_level:
        for importer, modname, ispkg in pkgutil.walk_packages(
            path=mdl.__path__, prefix=mdl.__name__ + ".", onerror=lambda x: None
        ):
            try:
                module_name = modname.rsplit(".", 1)[-1]
                # If module's name starts with "_", do not load the module.
                # But if the module's name starts with a "__", then load the
                # module.
                if module_name.startswith("_") and not module_name.startswith("__"):
                    continue

                with warnings.catch_warnings(record=True) as recorded_warnings:
                    if sys.version_info < (3, 10):
                        m = importer.find_module(modname)  # type: ignore
                        assert m is not None
                        loaded_mod = m.load_module(modname)
                    else:
                        spec = importer.find_spec(modname)
                        assert spec is not None
                        if modname in sys.modules:
                            loaded_mod = sys.modules[modname]
                        else:
                            loaded_mod = importlib.util.module_from_spec(spec)
                        if loaded_mod is not None:
                            spec.loader.exec_module(loaded_mod)
                            sys.modules[modname] = loaded_mod

                if len(recorded_warnings) > 0:
                    for w in recorded_warnings:
                        warnings.showwarning(
                            message=w.message,
                            category=w.category,
                            filename=w.filename,
                            lineno=w.lineno,
                            file=w.file,
                            line=w.line,
                        )

                if loaded_mod is not None:
                    for _name, obj in inspect.getmembers(loaded_mod):
                        if _is_concrete_backend_type(obj):
                            scanned_backends.append(obj)
            except ImportError as e:
                warnings.warn(
                    message=f"\n"
                    f"\tError importing uma backend '{modname}'.\n"
                    f"\t\t{type(e).__name__} : {e}",
                    category=UserWarning,
                )

    return scanned_backends


def _is_concrete_backend_type(obj: Any) -> bool:
    return (
        inspect.isclass(obj)
        and issubclass(obj, UMABackend)
        and not inspect.isabstract(obj)
    )
