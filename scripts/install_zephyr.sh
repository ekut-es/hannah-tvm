#!/bin/bash -e
##
## Copyright (c) 2023 University of Tübingen.
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


west init zephyrproject --mr v2.7.0
pushd zephyrproject
west update
pip install -r zephyr/scripts/requirements.txt
wget https://github.com/zephyrproject-rtos/sdk-ng/releases/download/v0.14.1/zephyr-sdk-0.14.1_linux-x86_64_minimal.tar.gz
tar xvzf zephyr-sdk-0.14.1_linux-x86_64_minimal.tar.gz
yes | zephyr-sdk-0.14.1/setup.sh
popd
