TARGET  = aot_runner
C_SRCS  = $(wildcard *.c) $(wildcard runtime/src/runtime/crt/*/*.c) $(wildcard cxx_support/*.c) $(wildcard codegen/host/src/*.c)
CXX_SRCS = $(wildcard *.cc) $(wildcard runtime/src/runtime/crt/*/*.cc) $(wildcard cxx_support/*.cc)
HEADERS = $(wildcard *.h)
CFLAGS += -O0 -g -Iinclude -Iruntime/include -I. -I codegen/host/include -ffunction-sections -fdata-sections
CXXFLAGS += -O0 -g -fno-exceptions -fno-rtti -ffunction-sections -fdata-sections

BOARD=tgfs-vp
LINK_TARGET=flash
RISCV_ARCH:=rv32i
RISCV_ABI:=ilp32
LDFLAGS := -march=$(RISCV_ARCH) -mabi=$(RISCV_ABI) -Wl,--gc-sections -lm

compiler := $(shell which riscv32-unknown-elf-gcc)
TOOL_DIR=$(dir $(compiler))

TRIPLET=riscv32-unknown-elf
BSP_BASE = $(TGC_VP_HOME)/fw/bsp
include $(BSP_BASE)/env/common-gcc.mk


.PHONY: run
run: $(TARGET)
	${TGC_VP_HOME}/build/src/tgc-vp -l output.log -f $(TARGET)
