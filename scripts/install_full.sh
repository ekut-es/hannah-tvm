#!/bin/bash -e

if command -v nvcc;
then
    CXX_COMPILER=$(which g++-8)
    USE_CUDA=ON
else
    CXX_COMPILER=g++
    USE_CUDA=OFF
fi

echo "Installing clang"
mkdir -p external/pulp-llvm/build
pushd external/pulp-llvm/build
cmake -DLLVM_ENABLE_PROJECTS=clang -G "Ninja" ../llvm -DCMAKE_CXX_COMPILER=$CXX_COMPILER -DCMAKE_INSTALL_PREFIX=$PWD/../install
cmake --build .
cmake --build . --target install
popd

echo "Installing tvm"
mkdir -p external/tvm/build
cp cmake/cuda_config.cmake external/tvm/build/config.cmake
pushd external/tvm/build
cmake --cmake-force-configure .. -G Ninja -DCMAKE_INSTALL_PREFIX=$VIRTUAL_ENV/ -DCMAKE_CXX_COMPILER=$CXX_COMPILER -DUSE_CUDA=$USE_CUDA
cmake --build .
popd
pip install -e external/tvm/python


echo "Installing utvm_static_runtime"
mkdir -p external/utvm_staticrt_codegen/build
pushd external/utvm_staticrt_codegen/build
cmake ..  -G Ninja -DCMAKE_INSTALL_PREFIX=$VIRTUAL_ENV/ -DCMAKE_CXX_COMPILER=$CXX_COMPILER
cmake --build .
popd