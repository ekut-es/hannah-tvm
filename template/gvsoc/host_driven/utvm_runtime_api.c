/*
 * Copyright (c) 2023 hannah-tvm contributors.
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

#include "utvm_runtime_api.h"

#include <stdlib.h>


void* TVMBackendAllocWorkspace(int device_type, int device_id, uint64_t nbytes, int dtype_code_hint,
                               int dtype_bits_hint) {
  void* ret = malloc(nbytes);
  if(!ret){
    TVMAPISetLastError("TVMBackendAllocWorkspace failed\n");
  }
  return ret;
}

int TVMBackendFreeWorkspace(int device_type, int device_id, void* ptr) {
  free(ptr);
  return 0;
}

static const char *g_last_error;
void TVMAPISetLastError(const char* msg) { g_last_error = msg; }
const char* TVMGetLastError(void) { return g_last_error; }

int TVMBackendParallelLaunch(FTVMParallelLambda flambda, void* cdata, int num_task) {
  TVMParallelGroupEnv env;
  env.num_task = 1;
  flambda(0, &env, cdata);
  return 0;
}

int TVMBackendParallelBarrier(int task_id, TVMParallelGroupEnv *penv) {
  return 0;
}
