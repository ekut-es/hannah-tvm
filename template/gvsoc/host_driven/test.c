
#include "tvm/runtime/c_runtime_api.h"
#include <stdio.h>

#include "rt/rt_api.h"

#include "utvm_runtime_api.h"
#include "runner.h"

int main()
{

  rt_perf_t perf;
  rt_perf_init(&perf);
  rt_perf_conf(&perf, 1 << RT_PERF_CYCLES);
  rt_perf_reset(&perf);
  rt_perf_start(&perf);

  int error = run();

  rt_perf_stop(&perf);

  if (error)
  {
    const char *msg = TVMGetLastError();
    if (msg)
      printf("%s\n", msg);
    else
      printf("error\n");
    return error;
  }
  else
  {
    printf("cycles:%u\n", rt_perf_read(RT_PERF_CYCLES));
  }
}
