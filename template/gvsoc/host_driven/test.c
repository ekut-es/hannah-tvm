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
