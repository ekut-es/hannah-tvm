#!/bin/bash

mkdir -p tvm/build
cp tvm_config.cmake tvm/build/config.cmake
pushd tvm/build
cmake .. -G Ninja -DCMAKE_INSTALL_PREFIX=$VIRTUAL_ENV/
cmake --build .
#cmake --build . --target install 
popd
pip install -e tvm/python 