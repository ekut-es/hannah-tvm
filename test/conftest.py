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
import pytest


def pytest_addoption(parser):
    parser.addoption("--enable-pulp", action="store_true", help="enable pulp tests")


def pytest_runtest_setup(item):
    if "pulp" in item.keywords and not item.config.getoption("--enable-pulp"):
        pytest.skip("need --pulp option to run this test")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "pulp: mark test to run only when pulp is available"
    )
