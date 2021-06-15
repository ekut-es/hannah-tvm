#!/bin/sh -e 

west init zephyrproject --mr v2.5.0
pushd zephyrproject
west update 
pip install -r zephyr/scripts/requirements.txt
wget https://github.com/zephyrproject-rtos/sdk-ng/releases/download/v0.12.4/zephyr-sdk-0.12.4-x86_64-linux-setup.run
chmod +x zephyr-sdk-0.12.4-x86_64-linux-setup.run
./zephyr-sdk-0.12.4-x86_64-linux-setup.run -- -d $PWD/sdk-0.12.4 -y 
popd 