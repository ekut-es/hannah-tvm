
#include <stdint.h>


// The subset of the TVM runtime API that is implemented by the minimal runtime API.
#define TVM_MICRO_RUNTIME_API_BACKEND_API __attribute__((weak, visibility("default")))

TVM_MICRO_RUNTIME_API_BACKEND_API int TVMBackendFreeWorkspace(int device_type, int device_id,
                                                              void* ptr);

TVM_MICRO_RUNTIME_API_BACKEND_API void* TVMBackendAllocWorkspace(int device_type, int device_id,
                                                                 uint64_t nbytes,
                                                                 int dtype_code_hint,
                                                                 int dtype_bits_hint);

typedef struct {
  void* sync_handle;
  int32_t num_task;
} TVMParallelGroupEnv;

typedef int (*FTVMParallelLambda)(int task_id, TVMParallelGroupEnv* penv, void* cdata);

TVM_MICRO_RUNTIME_API_BACKEND_API int TVMBackendParallelLaunch(FTVMParallelLambda flambda,
                                                               void* cdata, int num_task);
TVM_MICRO_RUNTIME_API_BACKEND_API int TVMBackendParallelBarrier(int task_id, TVMParallelGroupEnv *penv);

TVM_MICRO_RUNTIME_API_BACKEND_API void TVMAPISetLastError(const char* msg);
TVM_MICRO_RUNTIME_API_BACKEND_API const char* TVMGetLastError(void);

#undef TVM_MICRO_RUNTIME_API_BACKEND_API

