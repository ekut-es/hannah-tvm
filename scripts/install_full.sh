#!/bin/bash -e

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

pushd external/tvm
#git apply ../../patches/tvm.patch
popd

echo "Installing tvm"
mkdir -p external/tvm/build
cp cmake/cuda_config.cmake external/tvm/build/config.cmake
pushd external/tvm/build
cmake --cmake-force-configure .. -G Ninja -DCMAKE_INSTALL_PREFIX=$VIRTUAL_ENV/ -DCMAKE_C_COMPILER=$C_COMPILER -DCMAKE_CXX_COMPILER=$CXX_COMPILER -DUSE_CUDA=$USE_CUDA
cmake --build .
popd
pip install -e external/tvm/python


echo "Installing utvm_static_runtime"
mkdir -p external/utvm_staticrt_codegen/build
pushd external/utvm_staticrt_codegen/build
cmake ..  -G Ninja -DCMAKE_INSTALL_PREFIX=$VIRTUAL_ENV/ -DCMAKE_CXX_COMPILER=$CXX_COMPILER
cmake --build .
popd
