#
# Copyright (c) 2024 hannah-tvm contributors.
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
import pathlib
import sys

import pytest
from tvm.relay.backend.contrib.uma import uma_available

from hannah_tvm import uma_backends

mod_dir = pathlib.Path(__file__).parent
sys.path.append(mod_dir)

print(sys.path)


@pytest.mark.skipif(
    not uma_available(), reason='Only run if tvm has been compiled with "USE_UMA=ON"'
)
def test_uma_backend_registry():
    uma_backends.init()

    backend_names = set()
    for backend in uma_backends.backends():
        backend_names.add(backend.target_name)

    assert "backend1" in backend_names
    assert "backend2" in backend_names
