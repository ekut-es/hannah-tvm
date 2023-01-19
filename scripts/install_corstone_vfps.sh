#!/bin/bash
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

mkdir -p vp/arm
pushd vp/arm
rm -f FVP_Corstone_SSE-300_11.16_26.tgz
wget -c https://developer.arm.com/-/media/Arm%20Developer%20Community/Downloads/OSS/FVP/Corstone-300/FVP_Corstone_SSE-300_11.16_26.tgz
tar xvzf FVP_Corstone_SSE-300_11.16_26.tgz
./FVP_Corstone_SSE-300.sh --i-agree-to-the-contained-eula -f -d $PWD/FVP_Corstone_SSE-300
popd
