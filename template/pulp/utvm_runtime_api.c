
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
