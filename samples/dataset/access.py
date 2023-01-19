#
# Copyright (c) 2023 University of Tübingen.
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
from pprint import pprint

from hannah_tvm.dataset import DatasetFull

dataset = DatasetFull()

for network_result in dataset.network_results():
    print("=========================")
    print("Model:", network_result.model)
    print("Board:", network_result.board)
    print("Target:", network_result.target)
    print("Tuner:", network_result.tuner)
    print()
    print("Relay Graph:")
    relay = network_result.relay
    print(relay)
    print()
    print("TIR Primfuncs:")
    tir_funcs = network_result.tir
    if tir_funcs is not None:
        for func in tir_funcs:
            print(func)
    print()
    print("Performance Measurements:")
    measurement = network_result.measurement
    pprint(measurement)
    print()
