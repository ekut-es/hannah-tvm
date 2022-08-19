
CC = g++  # C compiler
CFLAGS = -c -fPIC -Wall -Wextra -O2 -g  -I../../external/ac_types/include -I../../external/tvm/include -I../../external/tvm/3rdparty/dlpack/include/ # C flags
LDFLAGS = -shared   # linking flags
RM = rm -f   # rm command
TARGET_LIB = libac_types.so  # target lib

SRCS = ac_int.cc  # source files
OBJS = $(SRCS:.cc=.o)

.PHONY: all
all: ${TARGET_LIB}

$(TARGET_LIB): $(OBJS)
	$(CC) ${LDFLAGS} -o $@ $^

%.o: %.cc
	$(CC) $(CFLAGS)  $< -o $@

ac_int.cc: generate.py
	python generate.py

#include $(SRCS:.cc=.d)

.PHONY: clean
clean:
	-${RM} ${TARGET_LIB} ${OBJS} $(SRCS:.cc=.d)
