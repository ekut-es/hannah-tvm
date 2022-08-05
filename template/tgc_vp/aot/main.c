/*
 * Copyright (c) 2022 University of Tübingen.
 *
 * This file is part of hannah-tvm.
 * See https://atreus.informatik.uni-tuebingen.de/ties/ai/hannah/hannah-tvm for further info.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * This is a sample Zephyr-based application that contains the logic
 * needed to control a microTVM-based model via the UART. This is only
 * intended to be a demonstration, since typically you will want to incorporate
 * this logic into your own application.
 */

#include <stdio.h>
#include <stdlib.h>

#include <tvm/runtime/crt/logging.h>
#include <tvm/runtime/crt/microtvm_rpc_server.h>

#include "codegen/host/include/tvmgen_default.h"
#include "crt_config.h"
#include "uart1.h"

#define SERIAL_BUFFER_SIZE (16)
char serial_buffer_data[SERIAL_BUFFER_SIZE];

// Called by TVM to write serial data to the UART.
ssize_t write_serial(void* unused_context, const uint8_t* data, size_t size) {
  size_t written = 0;

  while(written < size){
    written += uart1_write(data+written, size-written);
  }

  return size;
}

// Called by TVM when a message needs to be formatted.
size_t TVMPlatformFormatMessage(char* out_buf, size_t out_buf_size_bytes, const char* fmt,
                                va_list args) {
  return vsnprintf(out_buf, out_buf_size_bytes, fmt, args);
}

// Called by TVM when an internal invariant is violated, and execution cannot continue.
void TVMPlatformAbort(tvm_crt_error_t error) {
  printf("TVMError: 0x%x", error);
  exit(-1);
}

// Called by TVM to generate random data.
tvm_crt_error_t TVMPlatformGenerateRandom(uint8_t* buffer, size_t num_bytes) {
  //TODO: generate actual random numbers

  printf("Generating %d random numbers\n", num_bytes);
  for(size_t i = 0; i < num_bytes; i++){
    buffer[i] = i % 256;
  }

  return kTvmErrorNoError;
}


// Called by TVM to allocate memory.
tvm_crt_error_t TVMPlatformMemoryAllocate(size_t num_bytes, DLDevice dev, void** out_ptr) {
  *out_ptr =  malloc(num_bytes);
  return (*out_ptr == NULL) ? kTvmErrorPlatformNoMemory : kTvmErrorNoError;
}

// Called by TVM to deallocate memory.
tvm_crt_error_t TVMPlatformMemoryFree(void* ptr, DLDevice dev) {
  free(ptr);
  return kTvmErrorNoError;
}


// Called to start system timer.
tvm_crt_error_t TVMPlatformTimerStart() {
  //TODO
  return kTvmErrorNoError;
}

// Called to stop system timer.
tvm_crt_error_t TVMPlatformTimerStop(double* elapsed_time_seconds) {
  //TODO
  return kTvmErrorNoError;
}


// UART interrupt callback.
void uart_irq_cb(const struct device* dev, void* user_data) {

}

float input_data[10000];
float output_data[10000];

// The main function of this application.
int main(void) {
  uart1_init(115200);
  // Initialize microTVM RPC server, which will receive commands from the UART and execute them.
  printf("microTVM TGC runtime - running\r\n");

  struct tvmgen_default_inputs inputs;
  struct tvmgen_default_outputs  outputs;

  inputs.dense_4_input = input_data;
  outputs.Identity = output_data;

  tvmgen_default_run(&inputs, &outputs);

  printf("microTVM TGS runtime - stopped\r\n");

  return 0;
}
