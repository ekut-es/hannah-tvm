#!/bin/bash -e

pushd external/tvm
git apply ../../patches/tvm.patch
popd

echo "Installing clang"
mkdir -p external/pulp-llvm/build
pushd external/pulp-llvm/build
cmake -DLLVM_ENABLE_PROJECTS=clang -G "Ninja" ../llvm  -DCMAKE_INSTALL_PREFIX=$PWD/../install
cmake --build .
cmake --build . --target install
popd

echo "Installing tvm"
mkdir -p external/tvm/build
cp cmake/install_micro_pulp.sh  external/tvm/build/config.cmake
pushd external/tvm/build
cmake --cmake-force-configure .. -G Ninja -DCMAKE_INSTALL_PREFIX=$VIRTUAL_ENV/
cmake --build .
popd
pip install -e external/tvm/python


echo "Installing utvm_static_runtime"
mkdir -p external/utvm_staticrt_codegen/build
pushd external/utvm_staticrt_codegen/build
cmake ..  -G Ninja -DCMAKE_INSTALL_PREFIX=$VIRTUAL_ENV/
cmake --build .
popd
