<!--
Copyright (c) 2024 hannah-tvm contributors.

This file is part of hannah-tvm.
See https://atreus.informatik.uni-tuebingen.de/ties/ai/hannah/hannah-tvm for further info.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
# Installing

First build llvm with support for pulp instructions.

[pulp-llvm](https://atreus.informatik.uni-tuebingen.de/ties/timing/pulp-llvm) (8da738873bcecba323c95b7f0e95a1359e0e8618)

Then run

`./scripts/install_micro_pulp.sh path/to/llvm_config`

# Usage

To use hannah-tvm with pulp, adjust the paths in `template/pulp/sourceme.sh` and source the file. Alternatively set the variables in your terminal manually and setup the pulp-sdk.

# TVM

TVM is extended with pulp specific compute and schedule functions to tensorize the `nn.conv1d` and `nn.conv2d` operators with following input and kernel layout pairs.

| input | kernel |
| ----- | ------ |
| ncw   | oiw    |
| nwc   | wio    |
| nwc   | owi    |
| nchw  | oihw   |
| nhwc  | hwio   |
| nhwc  | ohwi   |


# MicroTVM

`template/pulp` contains a template project that can be used with microTVM. Generating and building a project compiles the module with ri5cy instructions, generates a c source file to run an execution graph, and links them to an executable binary that runs the graph and prints the cycles used for execution.

The template project offers the option `compiler` that can be set to `llvm` or `gcc` to specify the compiler used by the pulp-sdk.

# Autotvm

To tune a model for pulp, the board configuration must set `rpc_runner` to `pulp`. This makes autotvm use an RPC runner for pulp which utilizes the template project to compile, execute and measure the cycles of a tuning task.

# conf/board

There are five board configurations for the pulpissimo platform.

| Name               | TVM Target | pulp schedules | pulp instructions |
| ------------------ | ---------- | -------------- | ----------------- |
| gvsoc_llvm         | llvm       | yes            | yes               |
| gvsoc_llvm_nopulp  | llvm       | yes            | no                |
| gvsoc_llvm_generic | llvm       | no             | yes               |
| gvsoc_c            | c          | yes            | yes               |
| gvsoc_c_generic    | c          | no             | yes               |
