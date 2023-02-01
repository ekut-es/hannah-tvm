#!/bin/bash
##
## Copyright (c) 2023 hannah-tvm contributors.
##
## This file is part of hannah-tvm.
## See https://atreus.informatik.uni-tuebingen.de/ties/ai/hannah/hannah-tvm for further info.
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.
##

M3_REMOTE=https://github.com/Barkhausen-Institut/M3.git
M3_FOLDER=m3
PACKAGES_REQUIRED="clang git build-essential scons zlib1g-dev curl wget \
        m4 libboost-all-dev libssl-dev libgmp3-dev libmpfr-dev \
        libmpc-dev libncurses5-dev texinfo ninja-build libxml2-utils"

export M3_BUILD=release
export M3_ISA=riscv
export M3_TARGET=gem5


SUDO=sudo
if [ "$EUID" -eq 0 ] ; then
    SUDO=""
fi


clone(){
    url=$1
    folder=$2
    echo "Cloning ${url} into ${folder}"
    if ! git clone "${url}" "${folder}" 2>/dev/null && [ -d "${folder}" ] ; then
        echo "Clone failed because the folder ${folder} exists"
    fi
}

update() {
    folder=$1
    git -C "${folder}" pull
    git -C "${folder}" submodule update --init --recursive
}

ensure_packages() {
# Makes sure that required list of packages is installed
    required_packages=$1
    for required_pkg in $required_packages; do
        pkg_ok=$(dpkg-query -W --showformat='${Status}\n' $required_pkg|grep "install ok installed")
        echo Checking for $required_pkg: $pkg_ok
        if [ "" = "$pkg_ok" ]; then
            echo "No $required_pkg. Setting up $required_pkg."
            ${SUDO} apt-get --yes install $required_pkg
        fi
    done
}

build_gem5() {

    git -C ${M3_FOLDER} submodule update --init platform/gem5
    git -C ${M3_FOLDER}/platform/gem5 apply ${PWD}/scripts/patches/m3_gem5.patch
    pushd ${M3_FOLDER}/platform/gem5
    scons build/RISCV/gem5.opt
    popd

}

install_cross() {
    pushd ${M3_FOLDER}/cross
    ./build.sh riscv >& /tmp/gcc_build_log.txt
    popd
}

install_rust() {
    if ! command -v rustup &> /dev/null
    then
        echo "Installing rustup"
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rust-toolchain-install
        bash /tmp/rust-toolchain-install -y

        source "$HOME/.cargo/env"
    fi

    rustup update
    rustup toolchain install stable
    rustup toolchain install nightly
}

ensure_packages "$PACKAGES_REQUIRED"
install_rust

clone $M3_REMOTE $M3_FOLDER
update $M3_FOLDER


install_cross

tail -n200  /tmp/gcc_build_log.txt

build_gem5

pushd $M3_FOLDER
./b
./b run boot/hello.xml
