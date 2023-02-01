#
# Copyright (c) 2023 hannah-tvm contributors.
#
# This file is part of hannah-tvm.
# See https://atreus.informatik.uni-tuebingen.de/ties/ai/hannah/hannah-tvm for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import pathlib
import string

import config

header = """
#include <ac_int.h>

#include <tvm/runtime/c_runtime_api.h>

// Helper functions to convert between ac_int and unsigned 32 bit datatypes
template <class T>
static T Uint32ToAC(uint32_t in) {
  T* custom = reinterpret_cast<T*>(&in);
  return *custom;
}

template <class T>
static uint32_t ACToUint32(T in) {
  uint32_t* bits = reinterpret_cast<uint32_t*>(&in);
  return *bits;
}


"""

arithmetic_template = string.Template(
    """
extern "C" {

TVM_DLL uint32_t Min${DT_NAME}() {
  // return minimum representable value
  ac_int<${BITS}, ${SIGNED}> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<${BITS}, ${SIGNED}> >(min);
}


TVM_DLL float  ${DT_NAME}ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<${BITS}, ${SIGNED}> > (in).to_double();
  return static_cast<ac_int<${BITS}, ${SIGNED}> > (custom_datatype);
}

TVM_DLL uint32_t FloatTo${DT_NAME}(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<${BITS}, ${SIGNED}> > (in);
}

TVM_DLL uint32_t ${DT_NAME}Max(uint32_t a, uint32_t b) {
  // max
  ac_int<${BITS}, ${SIGNED}> acustom = Uint32ToAC<ac_int<${BITS}, ${SIGNED}> > (a);
  ac_int<${BITS}, ${SIGNED}> bcustom = Uint32ToAC<ac_int<${BITS}, ${SIGNED}> > (b);
  return ACToUint32<ac_int<${BITS}, ${SIGNED}> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t ${DT_NAME}Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<${BITS}, ${SIGNED}> acustom = Uint32ToAC<ac_int<${BITS}, ${SIGNED}> > (a);
  ac_int<${BITS}, ${SIGNED}> bcustom = Uint32ToAC<ac_int<${BITS}, ${SIGNED}> > (b);
  return ACToUint32<ac_int<${BITS}, ${SIGNED}> > (acustom + bcustom);
}

TVM_DLL uint32_t ${DT_NAME}Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<${BITS}, ${SIGNED}> acustom = Uint32ToAC<ac_int<${BITS}, ${SIGNED}> > (a);
  ac_int<${BITS}, ${SIGNED}> bcustom = Uint32ToAC<ac_int<${BITS}, ${SIGNED}> > (b);
  return ACToUint32<ac_int<${BITS}, ${SIGNED}> > (acustom - bcustom);
}

TVM_DLL uint32_t ${DT_NAME}Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<${BITS}, ${SIGNED}> acustom = Uint32ToAC<ac_int<${BITS}, ${SIGNED}> > (a);
  ac_int<${BITS}, ${SIGNED}> bcustom = Uint32ToAC<ac_int<${BITS}, ${SIGNED}> > (b);
  return ACToUint32<ac_int<${BITS}, ${SIGNED}> > (acustom * bcustom);
}

TVM_DLL uint32_t ${DT_NAME}Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<${BITS}, ${SIGNED}> acustom = Uint32ToAC<ac_int<${BITS}, ${SIGNED}> > (a);
  ac_int<${BITS}, ${SIGNED}> bcustom = Uint32ToAC<ac_int<${BITS}, ${SIGNED}> > (b);
  return ACToUint32<ac_int<${BITS}, ${SIGNED}> > (acustom / bcustom);
}


}
"""
)


result = header


for width in config.BITS:
    for sign in config.SIGNED:
        BITS = width
        SIGNED = "true" if sign else "false"
        NAME_PREFIX = "S" if sign else "U"
        DT_NAME = f"{NAME_PREFIX}INT{BITS}"

        result += f"\n// Generated code for dtype: {DT_NAME}\n"
        result += arithmetic_template.safe_substitute(
            BITS=BITS, SIGNED=SIGNED, DT_NAME=DT_NAME
        )

print("result:\n", result)

with pathlib.Path("ac_int.cc").open("w") as f:
    f.write(result)
