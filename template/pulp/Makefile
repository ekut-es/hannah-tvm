
CONFIG_OPT = 'compiler=llvm'
PULP_APP = test
PULP_CFLAGS = -O3 -g -I${TVM_HOME}/include -I${TVM_HOME}/3rdparty/dlpack/include

MODEL_SRC = $(wildcard build/codegen/host/lib/*.c)
MODEL_OBJ =  $(wildcard build/codegen/host/lib/*.o)

TESTFILE ?= test.c

PULP_APP_SRCS = $(TESTFILE) build/runner.c utvm_runtime_api.c $(MODEL_SRC)
PULP_LDFLAGS = $(MODEL_OBJ)

conf:
	echo $(PYTHONPATH)
all:: $(MODEL_OBJ)

include $(PULP_SDK_HOME)/install/rules/pulp_rt.mk

print:
	make conf clean all run PULP_CURRENT_CONFIG_ARGS="gvsoc/trace=insn"

test:
	make conf clean all run
	make dis > dump
