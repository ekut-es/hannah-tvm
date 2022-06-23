
CONFIG_OPT ?= 'compiler=llvm'
PULP_APP = test
PULP_CFLAGS = -O3 -g -Icrt/include

MODEL_SRC = $(wildcard build/codegen/host/src/*.c) $(wildcard build/model/*.c)
MODEL_OBJ = $(wildcard build/codegen/host/lib/*.o) $(wildcard build/model/*.o)

TESTFILE ?= test.c

PULP_APP_SRCS = $(TESTFILE) build/runner.c utvm_runtime_api.c $(MODEL_SRC)
PULP_LDFLAGS = $(MODEL_OBJ) -lm

conf:
	echo $(PYTHONPATH)
all:: $(MODEL_OBJ)

include $(PULP_SDK_HOME)/install/rules/pulp_rt.mk

print:
	make conf clean all run PULP_CURRENT_CONFIG_ARGS="gvsoc/trace=insn"

test:
	make conf clean all run
	make dis > dump
