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



echo "Installing tvm"
mkdir -p external/tvm/build
if $USE_CUDA == ON ;
then 
    cp cmake/cuda_config.cmake external/tvm/build/config.cmake
else
    cp cmake/nocuda_config.cmake external/tvm/build/config.cmake
fi
pushd external/tvm/build
cmake --cmake-force-configure .. -G Ninja -DCMAKE_INSTALL_PREFIX=$VIRTUAL_ENV/ -DCMAKE_C_COMPILER=$C_COMPILER -DCMAKE_CXX_COMPILER=$CXX_COMPILER 
cmake --build .
popd
pip install -e external/tvm/python


echo "Installing utvm_static_runtime"
mkdir -p external/utvm_staticrt_codegen/build
pushd external/utvm_staticrt_codegen/build
cmake ..  -G Ninja -DCMAKE_INSTALL_PREFIX=$VIRTUAL_ENV/ -DCMAKE_CXX_COMPILER=$CXX_COMPILER
cmake --build .
popd
