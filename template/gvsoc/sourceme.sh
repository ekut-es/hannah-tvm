##
## Copyright (c) 2022 University of Tübingen.
##
## This file is part of hannah-tvm.
## See https://atreus.informatik.uni-tuebingen.de/ties/ai/hannah/hannah-tvm for further info.
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.
##
export TVM_HOME=/home/muhcuser/hannah-tvm/external/tvm

export PULP_RISCV_GCC_TOOLCHAIN=/opt/pulp-riscv-gcc
export PULP_RISCV_LLVM_TOOLCHAIN=/home/muhcuser/pulp-llvm/build

pushd /home/muhcuser/pulp-sdk
source configs/pulpissimo.sh
source configs/platform-gvsoc.sh
source pkg/sdk/dev/sourceme.sh
popd
