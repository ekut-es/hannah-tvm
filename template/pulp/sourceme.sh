export TVM_HOME=/home/muhcuser/hannah-tvm/external/tvm

#export PULP_PLATFORM_ROOT=/home/serkan/pulp-platform
export PULP_RISCV_GCC_TOOLCHAIN=/opt/pulp-riscv-gcc
export PULP_RISCV_LLVM_TOOLCHAIN=/home/muhcuser/pulp-llvm/build

pushd /home/muhcuser/pulp-sdk #${PULP_PLATFORM_ROOT}/pulp-sdk
source configs/pulpissimo.sh
source configs/platform-gvsoc.sh
#make all
source pkg/sdk/dev/sourceme.sh
popd
