/*
 * Copyright (c) 2024 hannah-tvm contributors.
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


#include <tvm/runtime/crt/logging.h>
#include <tvm/runtime/crt/microtvm_rpc_server.h>
#include <unistd.h>

#include "crt_config.h"


static size_t g_num_bytes_requested = 0;
static size_t g_num_bytes_written = 0;
static size_t g_num_bytes_in_rx_buffer = 0;

// Called by TVM to write serial data to the UART.
ssize_t uart_write(void* unused_context, const uint8_t* data, size_t size) {
#ifdef CONFIG_LED
  gpio_pin_set(led0_pin, LED0_PIN, 1);
#endif
  g_num_bytes_requested += size;

  for (size_t i = 0; i < size; i++) {
//    uart_poll_out(tvm_uart, data[i]);
    g_num_bytes_written++;
  }

  return size;
}

ssize_t serial_write(void* unused_context, const uint8_t* data, size_t size) {
  return uart_write(unused_context, data, size);
}

//TODO: add crash handler

// Called by TVM when a message needs to be formatted.
size_t TVMPlatformFormatMessage(char* out_buf, size_t out_buf_size_bytes, const char* fmt,
                                va_list args) {
  return vsnprintf(out_buf, out_buf_size_bytes, fmt, args);
}

// Called by TVM when an internal invariant is violated, and execution cannot continue.
void TVMPlatformAbort(tvm_crt_error_t error) {
  TVMLogf("TVMError: 0x%x", error);
  //Do reboot if possible
  for (;;)
    ;
}

// Called by TVM to generate random data.
tvm_crt_error_t TVMPlatformGenerateRandom(uint8_t* buffer, size_t num_bytes) {
  uint32_t random;  // one unit of random data.

  // Fill parts of `buffer` which are as large as `random`.
  size_t num_full_blocks = num_bytes / sizeof(random);
  for (int i = 0; i < num_full_blocks; ++i) {
    random = 0; //FIXME: sys_rand32_get();
    memcpy(&buffer[i * sizeof(random)], &random, sizeof(random));
  }

  // Fill any leftover tail which is smaller than `random`.
  size_t num_tail_bytes = num_bytes % sizeof(random);
  if (num_tail_bytes > 0) {
    random = 1; //FIXME: rng sys_rand32_get();
    memcpy(&buffer[num_bytes - num_tail_bytes], &random, num_tail_bytes);
  }
  return kTvmErrorNoError;
}


// Called by TVM to allocate memory.
tvm_crt_error_t TVMPlatformMemoryAllocate(size_t num_bytes, DLDevice dev, void** out_ptr) {
  //allocate
  return (*out_ptr == NULL) ? kTvmErrorPlatformNoMemory : kTvmErrorNoError;
}

// Called by TVM to deallocate memory.
tvm_crt_error_t TVMPlatformMemoryFree(void* ptr, DLDevice dev) {
  //free
  return kTvmErrorNoError;
}

// Called to start system timer.
tvm_crt_error_t TVMPlatformTimerStart() {
  //TODO: start timer

  return kTvmErrorNoError;
}

// Called to stop system timer.
tvm_crt_error_t TVMPlatformTimerStop(double* elapsed_time_seconds) {

  //Stop Timer
  return kTvmErrorNoError;
}


// // UART interrupt callback.
// void uart_irq_cb(const struct device* dev, void* user_data) {
//   uart_irq_update(dev);
//   if (uart_irq_is_pending(dev)) {
//     struct ring_buf* rbuf = (struct ring_buf*)user_data;
//     if (uart_irq_rx_ready(dev) != 0) {
//       uint8_t* data;
//       uint32_t size;
//       size = ring_buf_put_claim(rbuf, &data, RING_BUF_SIZE_BYTES);
//       int rx_size = uart_fifo_read(dev, data, size);
//       // Write it into the ring buffer.
//       g_num_bytes_in_rx_buffer += rx_size;

//       if (g_num_bytes_in_rx_buffer > RING_BUF_SIZE_BYTES) {
//         TVMPlatformAbort((tvm_crt_error_t)0xbeef3);
//       }

//       if (rx_size < 0) {
//         TVMPlatformAbort((tvm_crt_error_t)0xbeef1);
//       }

//       int err = ring_buf_put_finish(rbuf, rx_size);
//       if (err != 0) {
//         TVMPlatformAbort((tvm_crt_error_t)0xbeef2);
//       }
//       // CHECK_EQ(bytes_read, bytes_written, "bytes_read: %d; bytes_written: %d", bytes_read,
//       // bytes_written);
//     }
//   }
// }

// // Used to initialize the UART receiver.
// void uart_rx_init(struct ring_buf* rbuf, const struct device* dev) {
//   uart_irq_callback_user_data_set(dev, uart_irq_cb, (void*)rbuf);
//   uart_irq_rx_enable(dev);
// }

// The main function of this application.
int main(int argc, char **argv) {
  // Claim console device.
  //tvm_uart = device_get_binding(DT_LABEL(DT_CHOSEN(zephyr_console)));
  //uart_rx_init(&uart_rx_rbuf, tvm_uart);

  // Initialize system timing. We could stop and start it every time, but we'll
  // be using it enough we should just keep it enabled.
  //timing_init();
  //timing_start();


  // Initialize microTVM RPC server, which will receive commands from the UART and execute them.
  //microtvm_rpc_server_t server = MicroTVMRpcServerInit(serial_write, NULL);
  TVMLogf("microTVM M3 runtime - running");


  // The main application loop. We continuously read commands from the UART
  // and dispatch them to MicroTVMRpcServerLoop().
  while (1) {
    // uint8_t* data;
    // unsigned int key = irq_lock();
    // uint32_t bytes_read = ring_buf_get_claim(&uart_rx_rbuf, &data, RING_BUF_SIZE_BYTES);
    // if (bytes_read > 0) {
    //   uint8_t* ptr = data;
    //   size_t bytes_remaining = bytes_read;
    //   while (bytes_remaining > 0) {
    //     // Pass the received bytes to the RPC server.
    //     tvm_crt_error_t err = MicroTVMRpcServerLoop(server, &ptr, &bytes_remaining);
    //     if (err != kTvmErrorNoError && err != kTvmErrorFramingShortPacket) {
    //       TVMPlatformAbort(err);
    //     }
    //     g_num_bytes_in_rx_buffer -= bytes_read;
    //     if (g_num_bytes_written != 0 || g_num_bytes_requested != 0) {
    //       if (g_num_bytes_written != g_num_bytes_requested) {
    //         TVMPlatformAbort((tvm_crt_error_t)0xbeef5);
    //       }
    //       g_num_bytes_written = 0;
    //       g_num_bytes_requested = 0;
    //     }
    //   }
    //   int err = ring_buf_get_finish(&uart_rx_rbuf, bytes_read);
    //   if (err != 0) {
    //     TVMPlatformAbort((tvm_crt_error_t)0xbeef6);
    //   }
    // }
    // irq_unlock(key);
  }

}
