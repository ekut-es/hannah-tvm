#!/bin/bash
##
## Copyright (c) 2023 hannah-tvm contributors.
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

rm -rf project_m3

# wget -c https://github.com/tensorflow/tflite-micro/raw/main/tensorflow/lite/micro/examples/magic_wand/magic_wand.tflite
# tvmc compile magic_wand.tflite \
#    --target='c -keys=cpu -model=host' \
#    --runtime=crt \
#    --runtime-crt-system-lib 1 \
#    --executor='graph' \
#    --executor-graph-link-params 0 \
#    --output model.tar \
#    --output-format mlf \
#    --pass-config tir.disable_vectorize=1 \
#    --disabled-pass=AlterOpLayout


tvmc micro create project_m3 model.tar template --template-dir $PWD/.. --project-option verbose=true project_type=host_driven board=gem5 # --list-options

tvmc micro build \
    project_m3 \
    template \
    --template-dir ..

tvmc micro flash \
    project_m3 \
    zephyr

 tvmc run \
         --device micro \
         project_m3 \
         --fill-mode ones \
         --print-top 4
