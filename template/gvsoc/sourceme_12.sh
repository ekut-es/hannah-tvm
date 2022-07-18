#Set Environment variables for
export PULP_RISCV_GCC_TOOLCHAIN=/opt/pulp-riscv-gcc
export PULP_RISCV_LLVM_TOOLCHAIN=/local/gerum/speech_recognition/external/hannah-tvm/external/pulp-llvm/install/

pushd /home/muhcuser/pulp-sdk #TODO(gerum): provide install script
source configs/pulpissimo.sh
source configs/platform-gvsoc.sh
source pkg/sdk/dev/sourceme.sh
popd
