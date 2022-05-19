
TARGET  = rpc_server
C_SRCS  = $(wildcard *.c) $(wildcard crt/src/runtime/crt/*/*.c)
CXX_SRCS = $(wildcard *.cc) $(wildcard crt/src/runtime/crt/*/*.cc)
HEADERS = $(wildcard *.h)
CFLAGS += -O0 -g -Icrt/include -I.
CXXFLAGS += -O0 -g -fno-exceptions -fno-rtti

BOARD=tgfs-vp
LINK_TARGET=flash
RISCV_ARCH:=rv32i
RISCV_ABI:=ilp32
LDFLAGS := -march=$(RISCV_ARCH) -mabi=$(RISCV_ABI)

compiler := $(shell which riscv32-unknown-elf-gcc)
TOOL_DIR=$(dir $(compiler))

TRIPLET=riscv32-unknown-elf
BSP_BASE = $(TGC_VP_HOME)/fw/bsp
include $(BSP_BASE)/env/common-gcc.mk
