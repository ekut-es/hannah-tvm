<!--
Copyright (c) 2023 University of Tübingen.

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
# TVM integration for HANNAH

To get the basic installation use:

1. Install `git-lfs` on ubuntu `apt install git-lfs`
2. checkout submodules: `git submodule update --init --recursive`
3. `poetry install -E automate -E dash -E micro`
4. `poetry shell`

For the tvm installation there are the following installation options.

## MicroTVM installation

This installation installs tvm with microtvm support without llvm backend

```
./scripts/install_micro.sh
./scripts/install_zephyr.sh
source env
```

Zephyr installation is optional but is currently needed for host driven execution.
For an example using host driven execution and the micro tvm zephyr runtime on stm32f429i discovery boards
see `samples/micro/zephyr_host_driven_stm32f429i.py`.

## MicroTVM with pulp-llvm support

This installation option activates the pulp-llvm based backend for direct vectorization on xpulpv targets.
For pulp targets no zephyr support is needed at the moment.

```
./scripts/install_micro_pulp.sh
```

## Full installation

The full installation uses the host provided llvm backend and activates cuda and llvm backends if available.

```
./scripts/install_full.sh
```

# Common error reasons

1. Pythonpath not set when using automate runner on schrank boards

  Add `AcceptEnv LANG LC_* PYTHONPATH` to `/etc/ssh/sshd_config` and restart server

# Result  Visualization

Currently a basic result visualization is available via `hannah-tvm-dashboard`
