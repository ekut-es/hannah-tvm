#!/bin/bash -e 


pushd external/tvm
echo "Updating patch"
git format-patch main --stdout > ../../patches/tvm.patch
echo "Removing patch"
git apply -R ../../patches/tvm.patch
echo "Update tvm"
git checkout main
git pull 
git submodule update --init --recursive
popd

echo "Commiting new tvm"
git commit -m "Update tvm" external/tvm

echo "Applying patch"
pushd external/tvm
git apply ../../patches/tvm.patch
popd
