#!/bin/bash

mkdir -p external/tvm/build
cp tvm_config.cmake external/tvm/build/config.cmake
pushd external/tvm/build
cmake --cmake-force-configure .. -G Ninja -DCMAKE_INSTALL_PREFIX=$VIRTUAL_ENV/ -DCMAKE_CXX_COMPILER=$(which g++-8)
cmake --build .
popd
pip install -e external/tvm/python 