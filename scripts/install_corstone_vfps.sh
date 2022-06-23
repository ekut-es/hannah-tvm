#!/bin/bash

mkdir -p vp/arm
pushd vp/arm
rm -f FVP_Corstone_SSE-300_11.16_26.tgz
wget -c https://developer.arm.com/-/media/Arm%20Developer%20Community/Downloads/OSS/FVP/Corstone-300/FVP_Corstone_SSE-300_11.16_26.tgz
tar xvzf FVP_Corstone_SSE-300_11.16_26.tgz
./FVP_Corstone_SSE-300.sh --i-agree-to-the-contained-eula -f -d $PWD/FVP_Corstone_SSE-300
popd
