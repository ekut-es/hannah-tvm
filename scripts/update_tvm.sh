#!/bin/bash -e 


pushd external/tvm
git format-patch main --stdout > ../../patches/tvm.patch
git checkout main
git pull 
git submodule update --init --recursive
popd
git commit -m "Update tvm" external/tvm
pushd external/tvm
git apply ../../patches/tvm.patch
popd
