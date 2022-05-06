#!/bin/bash -e


west init zephyrproject --mr v2.7.0
pushd zephyrproject
west update
pip install -r zephyr/scripts/requirements.txt
wget https://github.com/zephyrproject-rtos/sdk-ng/releases/download/v0.14.1/zephyr-sdk-0.14.1_linux-x86_64_minimal.tar.gz
tar xvzf zephyr-sdk-0.14.1_linux-x86_64_minimal.tar.gz
yes | zephyr-sdk-0.14.1/setup.sh
popd
