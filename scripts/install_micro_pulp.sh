#!/bin/bash -e

llvm_config_path=$1

echo "Installing tvm"
mkdir -p external/tvm/build
cp cmake/micro_config.cmake  external/tvm/build/config.cmake
echo "set(USE_LLVM ${llvm_config_path})" >> external/tvm/build/config.cmake
pushd external/tvm/build
cmake  .. -G Ninja -DCMAKE_INSTALL_PREFIX=$VIRTUAL_ENV/
cmake --build .
popd
pip install -e external/tvm/python


#echo "Installing utvm_static_runtime"
#mkdir -p external/utvm_staticrt_codegen/build
#pushd external/utvm_staticrt_codegen/build
#cmake ..  -G Ninja -DCMAKE_INSTALL_PREFIX=$VIRTUAL_ENV/
#cmake --build .
#popd
