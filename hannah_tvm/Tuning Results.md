<!--
Copyright (c) 2022 University of Tübingen.

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
# Tuning Results

## Untuned

board    model               status      tuning_duration      latency    latency_stdev
-------  ------------------  --------  -----------------  -----------  ---------------
n1sdp    conv-net-trax       finished                  0     170.439         0.105763
n1sdp    densenet-121        finished                  0   54917.5         324.541
n1sdp    efficientnet-lite4  finished                  0   46455.3         395.756
n1sdp    mobilenet-v2        failed                    0
n1sdp    resnet-18           finished                  0   23486.7         145.095
n1sdp    resnet-50           finished                  0   70465.4         800.513
n1sdp    shufflenet          finished                  0   12234.9           2.56138
n1sdp    squeezenet-1.1      finished                  0    6925.71         16.6315
n1sdp    vgg16               finished                  0  137653          4181.22
n1sdp    vgg19               finished                  0  163017          5410.69
n1sdp    tinyml_ad01         finished                  0      28.2964        0.0548302
n1sdp    tinyml_ic01         finished                  0     470.091         0.084204
n1sdp    tinyml_kws01        finished                  0     167.489         0.0824796
n1sdp    tinyml_vww01        finished                  0     626.656         0.184308


## Autotvm
