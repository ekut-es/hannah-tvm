#!/bin/bash -e
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

topdir=${0%/*}/..
echo "Running in $topdir"
pushd $topdir

if command -v nvcc;
then
    if command -v g++-7;
    then
        CXX_COMPILER=$(which g++-7)
        C_COMPILER=$(which gcc-7)
        USE_CUDA=ON
    else
       CXX_COMPILER=g++
       C_COMPILER=gcc
       USE_CUDA=OFF
    fi
else
    CXX_COMPILER=g++
    C_COMPILER=gcc
    USE_CUDA=OFF
fi

if command -v cmake3;
then
    CMAKE=cmake3
else
    CMAKE=cmake
fi

if command -v ninja;
then
    GENERATOR=Ninja
else
    GENERATOR="Unix Makefiles"
fi

echo "Installing tvm"
mkdir -p external/tvm/build
if [ $USE_CUDA = "ON" ] ;
then
    cp cmake/cuda_config.cmake external/tvm/build/config.cmake
else
    cp cmake/nocuda_config.cmake external/tvm/build/config.cmake
fi
pushd external/tvm/build
$CMAKE  .. -G "$GENERATOR" -DCMAKE_INSTALL_PREFIX=$VIRTUAL_ENV/ -DCMAKE_C_COMPILER=$C_COMPILER -DCMAKE_CXX_COMPILER=$CXX_COMPILER
$CMAKE --build .
popd
pip install -e external/tvm/python


#echo "Installing utvm_static_runtime"
#mkdir -p external/utvm_staticrt_codegen/build
#pushd external/utvm_staticrt_codegen/build
#$CMAKE ..  -G "$GENERATOR" -DCMAKE_INSTALL_PREFIX=$VIRTUAL_ENV/ -DCMAKE_CXX_COMPILER=$CXX_COMPILER
#$CMAKE --build .
#popd
popd
