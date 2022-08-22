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



// Generated code for dtype: SINT1

extern "C" {

TVM_DLL uint32_t MinSINT1() {
  // return minimum representable value
  ac_int<1, true> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<1, true> >(min);
}


TVM_DLL float  SINT1ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<1, true> > (in).to_double();
  return static_cast<ac_int<1, true> > (custom_datatype);
}

TVM_DLL uint32_t FloatToSINT1(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<1, true> > (in);
}

TVM_DLL uint32_t SINT1Max(uint32_t a, uint32_t b) {
  // max
  ac_int<1, true> acustom = Uint32ToAC<ac_int<1, true> > (a);
  ac_int<1, true> bcustom = Uint32ToAC<ac_int<1, true> > (b);
  return ACToUint32<ac_int<1, true> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t SINT1Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<1, true> acustom = Uint32ToAC<ac_int<1, true> > (a);
  ac_int<1, true> bcustom = Uint32ToAC<ac_int<1, true> > (b);
  return ACToUint32<ac_int<1, true> > (acustom + bcustom);
}

TVM_DLL uint32_t SINT1Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<1, true> acustom = Uint32ToAC<ac_int<1, true> > (a);
  ac_int<1, true> bcustom = Uint32ToAC<ac_int<1, true> > (b);
  return ACToUint32<ac_int<1, true> > (acustom - bcustom);
}

TVM_DLL uint32_t SINT1Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<1, true> acustom = Uint32ToAC<ac_int<1, true> > (a);
  ac_int<1, true> bcustom = Uint32ToAC<ac_int<1, true> > (b);
  return ACToUint32<ac_int<1, true> > (acustom * bcustom);
}

TVM_DLL uint32_t SINT1Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<1, true> acustom = Uint32ToAC<ac_int<1, true> > (a);
  ac_int<1, true> bcustom = Uint32ToAC<ac_int<1, true> > (b);
  return ACToUint32<ac_int<1, true> > (acustom / bcustom);
}


}

// Generated code for dtype: UINT1

extern "C" {

TVM_DLL uint32_t MinUINT1() {
  // return minimum representable value
  ac_int<1, false> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<1, false> >(min);
}


TVM_DLL float  UINT1ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<1, false> > (in).to_double();
  return static_cast<ac_int<1, false> > (custom_datatype);
}

TVM_DLL uint32_t FloatToUINT1(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<1, false> > (in);
}

TVM_DLL uint32_t UINT1Max(uint32_t a, uint32_t b) {
  // max
  ac_int<1, false> acustom = Uint32ToAC<ac_int<1, false> > (a);
  ac_int<1, false> bcustom = Uint32ToAC<ac_int<1, false> > (b);
  return ACToUint32<ac_int<1, false> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t UINT1Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<1, false> acustom = Uint32ToAC<ac_int<1, false> > (a);
  ac_int<1, false> bcustom = Uint32ToAC<ac_int<1, false> > (b);
  return ACToUint32<ac_int<1, false> > (acustom + bcustom);
}

TVM_DLL uint32_t UINT1Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<1, false> acustom = Uint32ToAC<ac_int<1, false> > (a);
  ac_int<1, false> bcustom = Uint32ToAC<ac_int<1, false> > (b);
  return ACToUint32<ac_int<1, false> > (acustom - bcustom);
}

TVM_DLL uint32_t UINT1Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<1, false> acustom = Uint32ToAC<ac_int<1, false> > (a);
  ac_int<1, false> bcustom = Uint32ToAC<ac_int<1, false> > (b);
  return ACToUint32<ac_int<1, false> > (acustom * bcustom);
}

TVM_DLL uint32_t UINT1Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<1, false> acustom = Uint32ToAC<ac_int<1, false> > (a);
  ac_int<1, false> bcustom = Uint32ToAC<ac_int<1, false> > (b);
  return ACToUint32<ac_int<1, false> > (acustom / bcustom);
}


}

// Generated code for dtype: SINT2

extern "C" {

TVM_DLL uint32_t MinSINT2() {
  // return minimum representable value
  ac_int<2, true> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<2, true> >(min);
}


TVM_DLL float  SINT2ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<2, true> > (in).to_double();
  return static_cast<ac_int<2, true> > (custom_datatype);
}

TVM_DLL uint32_t FloatToSINT2(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<2, true> > (in);
}

TVM_DLL uint32_t SINT2Max(uint32_t a, uint32_t b) {
  // max
  ac_int<2, true> acustom = Uint32ToAC<ac_int<2, true> > (a);
  ac_int<2, true> bcustom = Uint32ToAC<ac_int<2, true> > (b);
  return ACToUint32<ac_int<2, true> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t SINT2Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<2, true> acustom = Uint32ToAC<ac_int<2, true> > (a);
  ac_int<2, true> bcustom = Uint32ToAC<ac_int<2, true> > (b);
  return ACToUint32<ac_int<2, true> > (acustom + bcustom);
}

TVM_DLL uint32_t SINT2Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<2, true> acustom = Uint32ToAC<ac_int<2, true> > (a);
  ac_int<2, true> bcustom = Uint32ToAC<ac_int<2, true> > (b);
  return ACToUint32<ac_int<2, true> > (acustom - bcustom);
}

TVM_DLL uint32_t SINT2Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<2, true> acustom = Uint32ToAC<ac_int<2, true> > (a);
  ac_int<2, true> bcustom = Uint32ToAC<ac_int<2, true> > (b);
  return ACToUint32<ac_int<2, true> > (acustom * bcustom);
}

TVM_DLL uint32_t SINT2Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<2, true> acustom = Uint32ToAC<ac_int<2, true> > (a);
  ac_int<2, true> bcustom = Uint32ToAC<ac_int<2, true> > (b);
  return ACToUint32<ac_int<2, true> > (acustom / bcustom);
}


}

// Generated code for dtype: UINT2

extern "C" {

TVM_DLL uint32_t MinUINT2() {
  // return minimum representable value
  ac_int<2, false> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<2, false> >(min);
}


TVM_DLL float  UINT2ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<2, false> > (in).to_double();
  return static_cast<ac_int<2, false> > (custom_datatype);
}

TVM_DLL uint32_t FloatToUINT2(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<2, false> > (in);
}

TVM_DLL uint32_t UINT2Max(uint32_t a, uint32_t b) {
  // max
  ac_int<2, false> acustom = Uint32ToAC<ac_int<2, false> > (a);
  ac_int<2, false> bcustom = Uint32ToAC<ac_int<2, false> > (b);
  return ACToUint32<ac_int<2, false> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t UINT2Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<2, false> acustom = Uint32ToAC<ac_int<2, false> > (a);
  ac_int<2, false> bcustom = Uint32ToAC<ac_int<2, false> > (b);
  return ACToUint32<ac_int<2, false> > (acustom + bcustom);
}

TVM_DLL uint32_t UINT2Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<2, false> acustom = Uint32ToAC<ac_int<2, false> > (a);
  ac_int<2, false> bcustom = Uint32ToAC<ac_int<2, false> > (b);
  return ACToUint32<ac_int<2, false> > (acustom - bcustom);
}

TVM_DLL uint32_t UINT2Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<2, false> acustom = Uint32ToAC<ac_int<2, false> > (a);
  ac_int<2, false> bcustom = Uint32ToAC<ac_int<2, false> > (b);
  return ACToUint32<ac_int<2, false> > (acustom * bcustom);
}

TVM_DLL uint32_t UINT2Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<2, false> acustom = Uint32ToAC<ac_int<2, false> > (a);
  ac_int<2, false> bcustom = Uint32ToAC<ac_int<2, false> > (b);
  return ACToUint32<ac_int<2, false> > (acustom / bcustom);
}


}

// Generated code for dtype: SINT3

extern "C" {

TVM_DLL uint32_t MinSINT3() {
  // return minimum representable value
  ac_int<3, true> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<3, true> >(min);
}


TVM_DLL float  SINT3ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<3, true> > (in).to_double();
  return static_cast<ac_int<3, true> > (custom_datatype);
}

TVM_DLL uint32_t FloatToSINT3(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<3, true> > (in);
}

TVM_DLL uint32_t SINT3Max(uint32_t a, uint32_t b) {
  // max
  ac_int<3, true> acustom = Uint32ToAC<ac_int<3, true> > (a);
  ac_int<3, true> bcustom = Uint32ToAC<ac_int<3, true> > (b);
  return ACToUint32<ac_int<3, true> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t SINT3Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<3, true> acustom = Uint32ToAC<ac_int<3, true> > (a);
  ac_int<3, true> bcustom = Uint32ToAC<ac_int<3, true> > (b);
  return ACToUint32<ac_int<3, true> > (acustom + bcustom);
}

TVM_DLL uint32_t SINT3Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<3, true> acustom = Uint32ToAC<ac_int<3, true> > (a);
  ac_int<3, true> bcustom = Uint32ToAC<ac_int<3, true> > (b);
  return ACToUint32<ac_int<3, true> > (acustom - bcustom);
}

TVM_DLL uint32_t SINT3Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<3, true> acustom = Uint32ToAC<ac_int<3, true> > (a);
  ac_int<3, true> bcustom = Uint32ToAC<ac_int<3, true> > (b);
  return ACToUint32<ac_int<3, true> > (acustom * bcustom);
}

TVM_DLL uint32_t SINT3Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<3, true> acustom = Uint32ToAC<ac_int<3, true> > (a);
  ac_int<3, true> bcustom = Uint32ToAC<ac_int<3, true> > (b);
  return ACToUint32<ac_int<3, true> > (acustom / bcustom);
}


}

// Generated code for dtype: UINT3

extern "C" {

TVM_DLL uint32_t MinUINT3() {
  // return minimum representable value
  ac_int<3, false> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<3, false> >(min);
}


TVM_DLL float  UINT3ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<3, false> > (in).to_double();
  return static_cast<ac_int<3, false> > (custom_datatype);
}

TVM_DLL uint32_t FloatToUINT3(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<3, false> > (in);
}

TVM_DLL uint32_t UINT3Max(uint32_t a, uint32_t b) {
  // max
  ac_int<3, false> acustom = Uint32ToAC<ac_int<3, false> > (a);
  ac_int<3, false> bcustom = Uint32ToAC<ac_int<3, false> > (b);
  return ACToUint32<ac_int<3, false> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t UINT3Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<3, false> acustom = Uint32ToAC<ac_int<3, false> > (a);
  ac_int<3, false> bcustom = Uint32ToAC<ac_int<3, false> > (b);
  return ACToUint32<ac_int<3, false> > (acustom + bcustom);
}

TVM_DLL uint32_t UINT3Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<3, false> acustom = Uint32ToAC<ac_int<3, false> > (a);
  ac_int<3, false> bcustom = Uint32ToAC<ac_int<3, false> > (b);
  return ACToUint32<ac_int<3, false> > (acustom - bcustom);
}

TVM_DLL uint32_t UINT3Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<3, false> acustom = Uint32ToAC<ac_int<3, false> > (a);
  ac_int<3, false> bcustom = Uint32ToAC<ac_int<3, false> > (b);
  return ACToUint32<ac_int<3, false> > (acustom * bcustom);
}

TVM_DLL uint32_t UINT3Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<3, false> acustom = Uint32ToAC<ac_int<3, false> > (a);
  ac_int<3, false> bcustom = Uint32ToAC<ac_int<3, false> > (b);
  return ACToUint32<ac_int<3, false> > (acustom / bcustom);
}


}

// Generated code for dtype: SINT4

extern "C" {

TVM_DLL uint32_t MinSINT4() {
  // return minimum representable value
  ac_int<4, true> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<4, true> >(min);
}


TVM_DLL float  SINT4ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<4, true> > (in).to_double();
  return static_cast<ac_int<4, true> > (custom_datatype);
}

TVM_DLL uint32_t FloatToSINT4(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<4, true> > (in);
}

TVM_DLL uint32_t SINT4Max(uint32_t a, uint32_t b) {
  // max
  ac_int<4, true> acustom = Uint32ToAC<ac_int<4, true> > (a);
  ac_int<4, true> bcustom = Uint32ToAC<ac_int<4, true> > (b);
  return ACToUint32<ac_int<4, true> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t SINT4Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<4, true> acustom = Uint32ToAC<ac_int<4, true> > (a);
  ac_int<4, true> bcustom = Uint32ToAC<ac_int<4, true> > (b);
  return ACToUint32<ac_int<4, true> > (acustom + bcustom);
}

TVM_DLL uint32_t SINT4Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<4, true> acustom = Uint32ToAC<ac_int<4, true> > (a);
  ac_int<4, true> bcustom = Uint32ToAC<ac_int<4, true> > (b);
  return ACToUint32<ac_int<4, true> > (acustom - bcustom);
}

TVM_DLL uint32_t SINT4Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<4, true> acustom = Uint32ToAC<ac_int<4, true> > (a);
  ac_int<4, true> bcustom = Uint32ToAC<ac_int<4, true> > (b);
  return ACToUint32<ac_int<4, true> > (acustom * bcustom);
}

TVM_DLL uint32_t SINT4Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<4, true> acustom = Uint32ToAC<ac_int<4, true> > (a);
  ac_int<4, true> bcustom = Uint32ToAC<ac_int<4, true> > (b);
  return ACToUint32<ac_int<4, true> > (acustom / bcustom);
}


}

// Generated code for dtype: UINT4

extern "C" {

TVM_DLL uint32_t MinUINT4() {
  // return minimum representable value
  ac_int<4, false> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<4, false> >(min);
}


TVM_DLL float  UINT4ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<4, false> > (in).to_double();
  return static_cast<ac_int<4, false> > (custom_datatype);
}

TVM_DLL uint32_t FloatToUINT4(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<4, false> > (in);
}

TVM_DLL uint32_t UINT4Max(uint32_t a, uint32_t b) {
  // max
  ac_int<4, false> acustom = Uint32ToAC<ac_int<4, false> > (a);
  ac_int<4, false> bcustom = Uint32ToAC<ac_int<4, false> > (b);
  return ACToUint32<ac_int<4, false> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t UINT4Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<4, false> acustom = Uint32ToAC<ac_int<4, false> > (a);
  ac_int<4, false> bcustom = Uint32ToAC<ac_int<4, false> > (b);
  return ACToUint32<ac_int<4, false> > (acustom + bcustom);
}

TVM_DLL uint32_t UINT4Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<4, false> acustom = Uint32ToAC<ac_int<4, false> > (a);
  ac_int<4, false> bcustom = Uint32ToAC<ac_int<4, false> > (b);
  return ACToUint32<ac_int<4, false> > (acustom - bcustom);
}

TVM_DLL uint32_t UINT4Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<4, false> acustom = Uint32ToAC<ac_int<4, false> > (a);
  ac_int<4, false> bcustom = Uint32ToAC<ac_int<4, false> > (b);
  return ACToUint32<ac_int<4, false> > (acustom * bcustom);
}

TVM_DLL uint32_t UINT4Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<4, false> acustom = Uint32ToAC<ac_int<4, false> > (a);
  ac_int<4, false> bcustom = Uint32ToAC<ac_int<4, false> > (b);
  return ACToUint32<ac_int<4, false> > (acustom / bcustom);
}


}

// Generated code for dtype: SINT5

extern "C" {

TVM_DLL uint32_t MinSINT5() {
  // return minimum representable value
  ac_int<5, true> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<5, true> >(min);
}


TVM_DLL float  SINT5ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<5, true> > (in).to_double();
  return static_cast<ac_int<5, true> > (custom_datatype);
}

TVM_DLL uint32_t FloatToSINT5(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<5, true> > (in);
}

TVM_DLL uint32_t SINT5Max(uint32_t a, uint32_t b) {
  // max
  ac_int<5, true> acustom = Uint32ToAC<ac_int<5, true> > (a);
  ac_int<5, true> bcustom = Uint32ToAC<ac_int<5, true> > (b);
  return ACToUint32<ac_int<5, true> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t SINT5Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<5, true> acustom = Uint32ToAC<ac_int<5, true> > (a);
  ac_int<5, true> bcustom = Uint32ToAC<ac_int<5, true> > (b);
  return ACToUint32<ac_int<5, true> > (acustom + bcustom);
}

TVM_DLL uint32_t SINT5Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<5, true> acustom = Uint32ToAC<ac_int<5, true> > (a);
  ac_int<5, true> bcustom = Uint32ToAC<ac_int<5, true> > (b);
  return ACToUint32<ac_int<5, true> > (acustom - bcustom);
}

TVM_DLL uint32_t SINT5Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<5, true> acustom = Uint32ToAC<ac_int<5, true> > (a);
  ac_int<5, true> bcustom = Uint32ToAC<ac_int<5, true> > (b);
  return ACToUint32<ac_int<5, true> > (acustom * bcustom);
}

TVM_DLL uint32_t SINT5Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<5, true> acustom = Uint32ToAC<ac_int<5, true> > (a);
  ac_int<5, true> bcustom = Uint32ToAC<ac_int<5, true> > (b);
  return ACToUint32<ac_int<5, true> > (acustom / bcustom);
}


}

// Generated code for dtype: UINT5

extern "C" {

TVM_DLL uint32_t MinUINT5() {
  // return minimum representable value
  ac_int<5, false> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<5, false> >(min);
}


TVM_DLL float  UINT5ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<5, false> > (in).to_double();
  return static_cast<ac_int<5, false> > (custom_datatype);
}

TVM_DLL uint32_t FloatToUINT5(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<5, false> > (in);
}

TVM_DLL uint32_t UINT5Max(uint32_t a, uint32_t b) {
  // max
  ac_int<5, false> acustom = Uint32ToAC<ac_int<5, false> > (a);
  ac_int<5, false> bcustom = Uint32ToAC<ac_int<5, false> > (b);
  return ACToUint32<ac_int<5, false> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t UINT5Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<5, false> acustom = Uint32ToAC<ac_int<5, false> > (a);
  ac_int<5, false> bcustom = Uint32ToAC<ac_int<5, false> > (b);
  return ACToUint32<ac_int<5, false> > (acustom + bcustom);
}

TVM_DLL uint32_t UINT5Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<5, false> acustom = Uint32ToAC<ac_int<5, false> > (a);
  ac_int<5, false> bcustom = Uint32ToAC<ac_int<5, false> > (b);
  return ACToUint32<ac_int<5, false> > (acustom - bcustom);
}

TVM_DLL uint32_t UINT5Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<5, false> acustom = Uint32ToAC<ac_int<5, false> > (a);
  ac_int<5, false> bcustom = Uint32ToAC<ac_int<5, false> > (b);
  return ACToUint32<ac_int<5, false> > (acustom * bcustom);
}

TVM_DLL uint32_t UINT5Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<5, false> acustom = Uint32ToAC<ac_int<5, false> > (a);
  ac_int<5, false> bcustom = Uint32ToAC<ac_int<5, false> > (b);
  return ACToUint32<ac_int<5, false> > (acustom / bcustom);
}


}

// Generated code for dtype: SINT6

extern "C" {

TVM_DLL uint32_t MinSINT6() {
  // return minimum representable value
  ac_int<6, true> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<6, true> >(min);
}


TVM_DLL float  SINT6ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<6, true> > (in).to_double();
  return static_cast<ac_int<6, true> > (custom_datatype);
}

TVM_DLL uint32_t FloatToSINT6(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<6, true> > (in);
}

TVM_DLL uint32_t SINT6Max(uint32_t a, uint32_t b) {
  // max
  ac_int<6, true> acustom = Uint32ToAC<ac_int<6, true> > (a);
  ac_int<6, true> bcustom = Uint32ToAC<ac_int<6, true> > (b);
  return ACToUint32<ac_int<6, true> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t SINT6Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<6, true> acustom = Uint32ToAC<ac_int<6, true> > (a);
  ac_int<6, true> bcustom = Uint32ToAC<ac_int<6, true> > (b);
  return ACToUint32<ac_int<6, true> > (acustom + bcustom);
}

TVM_DLL uint32_t SINT6Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<6, true> acustom = Uint32ToAC<ac_int<6, true> > (a);
  ac_int<6, true> bcustom = Uint32ToAC<ac_int<6, true> > (b);
  return ACToUint32<ac_int<6, true> > (acustom - bcustom);
}

TVM_DLL uint32_t SINT6Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<6, true> acustom = Uint32ToAC<ac_int<6, true> > (a);
  ac_int<6, true> bcustom = Uint32ToAC<ac_int<6, true> > (b);
  return ACToUint32<ac_int<6, true> > (acustom * bcustom);
}

TVM_DLL uint32_t SINT6Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<6, true> acustom = Uint32ToAC<ac_int<6, true> > (a);
  ac_int<6, true> bcustom = Uint32ToAC<ac_int<6, true> > (b);
  return ACToUint32<ac_int<6, true> > (acustom / bcustom);
}


}

// Generated code for dtype: UINT6

extern "C" {

TVM_DLL uint32_t MinUINT6() {
  // return minimum representable value
  ac_int<6, false> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<6, false> >(min);
}


TVM_DLL float  UINT6ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<6, false> > (in).to_double();
  return static_cast<ac_int<6, false> > (custom_datatype);
}

TVM_DLL uint32_t FloatToUINT6(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<6, false> > (in);
}

TVM_DLL uint32_t UINT6Max(uint32_t a, uint32_t b) {
  // max
  ac_int<6, false> acustom = Uint32ToAC<ac_int<6, false> > (a);
  ac_int<6, false> bcustom = Uint32ToAC<ac_int<6, false> > (b);
  return ACToUint32<ac_int<6, false> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t UINT6Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<6, false> acustom = Uint32ToAC<ac_int<6, false> > (a);
  ac_int<6, false> bcustom = Uint32ToAC<ac_int<6, false> > (b);
  return ACToUint32<ac_int<6, false> > (acustom + bcustom);
}

TVM_DLL uint32_t UINT6Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<6, false> acustom = Uint32ToAC<ac_int<6, false> > (a);
  ac_int<6, false> bcustom = Uint32ToAC<ac_int<6, false> > (b);
  return ACToUint32<ac_int<6, false> > (acustom - bcustom);
}

TVM_DLL uint32_t UINT6Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<6, false> acustom = Uint32ToAC<ac_int<6, false> > (a);
  ac_int<6, false> bcustom = Uint32ToAC<ac_int<6, false> > (b);
  return ACToUint32<ac_int<6, false> > (acustom * bcustom);
}

TVM_DLL uint32_t UINT6Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<6, false> acustom = Uint32ToAC<ac_int<6, false> > (a);
  ac_int<6, false> bcustom = Uint32ToAC<ac_int<6, false> > (b);
  return ACToUint32<ac_int<6, false> > (acustom / bcustom);
}


}

// Generated code for dtype: SINT7

extern "C" {

TVM_DLL uint32_t MinSINT7() {
  // return minimum representable value
  ac_int<7, true> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<7, true> >(min);
}


TVM_DLL float  SINT7ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<7, true> > (in).to_double();
  return static_cast<ac_int<7, true> > (custom_datatype);
}

TVM_DLL uint32_t FloatToSINT7(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<7, true> > (in);
}

TVM_DLL uint32_t SINT7Max(uint32_t a, uint32_t b) {
  // max
  ac_int<7, true> acustom = Uint32ToAC<ac_int<7, true> > (a);
  ac_int<7, true> bcustom = Uint32ToAC<ac_int<7, true> > (b);
  return ACToUint32<ac_int<7, true> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t SINT7Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<7, true> acustom = Uint32ToAC<ac_int<7, true> > (a);
  ac_int<7, true> bcustom = Uint32ToAC<ac_int<7, true> > (b);
  return ACToUint32<ac_int<7, true> > (acustom + bcustom);
}

TVM_DLL uint32_t SINT7Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<7, true> acustom = Uint32ToAC<ac_int<7, true> > (a);
  ac_int<7, true> bcustom = Uint32ToAC<ac_int<7, true> > (b);
  return ACToUint32<ac_int<7, true> > (acustom - bcustom);
}

TVM_DLL uint32_t SINT7Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<7, true> acustom = Uint32ToAC<ac_int<7, true> > (a);
  ac_int<7, true> bcustom = Uint32ToAC<ac_int<7, true> > (b);
  return ACToUint32<ac_int<7, true> > (acustom * bcustom);
}

TVM_DLL uint32_t SINT7Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<7, true> acustom = Uint32ToAC<ac_int<7, true> > (a);
  ac_int<7, true> bcustom = Uint32ToAC<ac_int<7, true> > (b);
  return ACToUint32<ac_int<7, true> > (acustom / bcustom);
}


}

// Generated code for dtype: UINT7

extern "C" {

TVM_DLL uint32_t MinUINT7() {
  // return minimum representable value
  ac_int<7, false> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<7, false> >(min);
}


TVM_DLL float  UINT7ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<7, false> > (in).to_double();
  return static_cast<ac_int<7, false> > (custom_datatype);
}

TVM_DLL uint32_t FloatToUINT7(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<7, false> > (in);
}

TVM_DLL uint32_t UINT7Max(uint32_t a, uint32_t b) {
  // max
  ac_int<7, false> acustom = Uint32ToAC<ac_int<7, false> > (a);
  ac_int<7, false> bcustom = Uint32ToAC<ac_int<7, false> > (b);
  return ACToUint32<ac_int<7, false> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t UINT7Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<7, false> acustom = Uint32ToAC<ac_int<7, false> > (a);
  ac_int<7, false> bcustom = Uint32ToAC<ac_int<7, false> > (b);
  return ACToUint32<ac_int<7, false> > (acustom + bcustom);
}

TVM_DLL uint32_t UINT7Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<7, false> acustom = Uint32ToAC<ac_int<7, false> > (a);
  ac_int<7, false> bcustom = Uint32ToAC<ac_int<7, false> > (b);
  return ACToUint32<ac_int<7, false> > (acustom - bcustom);
}

TVM_DLL uint32_t UINT7Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<7, false> acustom = Uint32ToAC<ac_int<7, false> > (a);
  ac_int<7, false> bcustom = Uint32ToAC<ac_int<7, false> > (b);
  return ACToUint32<ac_int<7, false> > (acustom * bcustom);
}

TVM_DLL uint32_t UINT7Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<7, false> acustom = Uint32ToAC<ac_int<7, false> > (a);
  ac_int<7, false> bcustom = Uint32ToAC<ac_int<7, false> > (b);
  return ACToUint32<ac_int<7, false> > (acustom / bcustom);
}


}

// Generated code for dtype: SINT8

extern "C" {

TVM_DLL uint32_t MinSINT8() {
  // return minimum representable value
  ac_int<8, true> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<8, true> >(min);
}


TVM_DLL float  SINT8ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<8, true> > (in).to_double();
  return static_cast<ac_int<8, true> > (custom_datatype);
}

TVM_DLL uint32_t FloatToSINT8(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<8, true> > (in);
}

TVM_DLL uint32_t SINT8Max(uint32_t a, uint32_t b) {
  // max
  ac_int<8, true> acustom = Uint32ToAC<ac_int<8, true> > (a);
  ac_int<8, true> bcustom = Uint32ToAC<ac_int<8, true> > (b);
  return ACToUint32<ac_int<8, true> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t SINT8Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<8, true> acustom = Uint32ToAC<ac_int<8, true> > (a);
  ac_int<8, true> bcustom = Uint32ToAC<ac_int<8, true> > (b);
  return ACToUint32<ac_int<8, true> > (acustom + bcustom);
}

TVM_DLL uint32_t SINT8Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<8, true> acustom = Uint32ToAC<ac_int<8, true> > (a);
  ac_int<8, true> bcustom = Uint32ToAC<ac_int<8, true> > (b);
  return ACToUint32<ac_int<8, true> > (acustom - bcustom);
}

TVM_DLL uint32_t SINT8Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<8, true> acustom = Uint32ToAC<ac_int<8, true> > (a);
  ac_int<8, true> bcustom = Uint32ToAC<ac_int<8, true> > (b);
  return ACToUint32<ac_int<8, true> > (acustom * bcustom);
}

TVM_DLL uint32_t SINT8Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<8, true> acustom = Uint32ToAC<ac_int<8, true> > (a);
  ac_int<8, true> bcustom = Uint32ToAC<ac_int<8, true> > (b);
  return ACToUint32<ac_int<8, true> > (acustom / bcustom);
}


}

// Generated code for dtype: UINT8

extern "C" {

TVM_DLL uint32_t MinUINT8() {
  // return minimum representable value
  ac_int<8, false> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<8, false> >(min);
}


TVM_DLL float  UINT8ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<8, false> > (in).to_double();
  return static_cast<ac_int<8, false> > (custom_datatype);
}

TVM_DLL uint32_t FloatToUINT8(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<8, false> > (in);
}

TVM_DLL uint32_t UINT8Max(uint32_t a, uint32_t b) {
  // max
  ac_int<8, false> acustom = Uint32ToAC<ac_int<8, false> > (a);
  ac_int<8, false> bcustom = Uint32ToAC<ac_int<8, false> > (b);
  return ACToUint32<ac_int<8, false> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t UINT8Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<8, false> acustom = Uint32ToAC<ac_int<8, false> > (a);
  ac_int<8, false> bcustom = Uint32ToAC<ac_int<8, false> > (b);
  return ACToUint32<ac_int<8, false> > (acustom + bcustom);
}

TVM_DLL uint32_t UINT8Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<8, false> acustom = Uint32ToAC<ac_int<8, false> > (a);
  ac_int<8, false> bcustom = Uint32ToAC<ac_int<8, false> > (b);
  return ACToUint32<ac_int<8, false> > (acustom - bcustom);
}

TVM_DLL uint32_t UINT8Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<8, false> acustom = Uint32ToAC<ac_int<8, false> > (a);
  ac_int<8, false> bcustom = Uint32ToAC<ac_int<8, false> > (b);
  return ACToUint32<ac_int<8, false> > (acustom * bcustom);
}

TVM_DLL uint32_t UINT8Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<8, false> acustom = Uint32ToAC<ac_int<8, false> > (a);
  ac_int<8, false> bcustom = Uint32ToAC<ac_int<8, false> > (b);
  return ACToUint32<ac_int<8, false> > (acustom / bcustom);
}


}

// Generated code for dtype: SINT9

extern "C" {

TVM_DLL uint32_t MinSINT9() {
  // return minimum representable value
  ac_int<9, true> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<9, true> >(min);
}


TVM_DLL float  SINT9ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<9, true> > (in).to_double();
  return static_cast<ac_int<9, true> > (custom_datatype);
}

TVM_DLL uint32_t FloatToSINT9(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<9, true> > (in);
}

TVM_DLL uint32_t SINT9Max(uint32_t a, uint32_t b) {
  // max
  ac_int<9, true> acustom = Uint32ToAC<ac_int<9, true> > (a);
  ac_int<9, true> bcustom = Uint32ToAC<ac_int<9, true> > (b);
  return ACToUint32<ac_int<9, true> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t SINT9Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<9, true> acustom = Uint32ToAC<ac_int<9, true> > (a);
  ac_int<9, true> bcustom = Uint32ToAC<ac_int<9, true> > (b);
  return ACToUint32<ac_int<9, true> > (acustom + bcustom);
}

TVM_DLL uint32_t SINT9Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<9, true> acustom = Uint32ToAC<ac_int<9, true> > (a);
  ac_int<9, true> bcustom = Uint32ToAC<ac_int<9, true> > (b);
  return ACToUint32<ac_int<9, true> > (acustom - bcustom);
}

TVM_DLL uint32_t SINT9Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<9, true> acustom = Uint32ToAC<ac_int<9, true> > (a);
  ac_int<9, true> bcustom = Uint32ToAC<ac_int<9, true> > (b);
  return ACToUint32<ac_int<9, true> > (acustom * bcustom);
}

TVM_DLL uint32_t SINT9Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<9, true> acustom = Uint32ToAC<ac_int<9, true> > (a);
  ac_int<9, true> bcustom = Uint32ToAC<ac_int<9, true> > (b);
  return ACToUint32<ac_int<9, true> > (acustom / bcustom);
}


}

// Generated code for dtype: UINT9

extern "C" {

TVM_DLL uint32_t MinUINT9() {
  // return minimum representable value
  ac_int<9, false> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<9, false> >(min);
}


TVM_DLL float  UINT9ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<9, false> > (in).to_double();
  return static_cast<ac_int<9, false> > (custom_datatype);
}

TVM_DLL uint32_t FloatToUINT9(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<9, false> > (in);
}

TVM_DLL uint32_t UINT9Max(uint32_t a, uint32_t b) {
  // max
  ac_int<9, false> acustom = Uint32ToAC<ac_int<9, false> > (a);
  ac_int<9, false> bcustom = Uint32ToAC<ac_int<9, false> > (b);
  return ACToUint32<ac_int<9, false> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t UINT9Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<9, false> acustom = Uint32ToAC<ac_int<9, false> > (a);
  ac_int<9, false> bcustom = Uint32ToAC<ac_int<9, false> > (b);
  return ACToUint32<ac_int<9, false> > (acustom + bcustom);
}

TVM_DLL uint32_t UINT9Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<9, false> acustom = Uint32ToAC<ac_int<9, false> > (a);
  ac_int<9, false> bcustom = Uint32ToAC<ac_int<9, false> > (b);
  return ACToUint32<ac_int<9, false> > (acustom - bcustom);
}

TVM_DLL uint32_t UINT9Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<9, false> acustom = Uint32ToAC<ac_int<9, false> > (a);
  ac_int<9, false> bcustom = Uint32ToAC<ac_int<9, false> > (b);
  return ACToUint32<ac_int<9, false> > (acustom * bcustom);
}

TVM_DLL uint32_t UINT9Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<9, false> acustom = Uint32ToAC<ac_int<9, false> > (a);
  ac_int<9, false> bcustom = Uint32ToAC<ac_int<9, false> > (b);
  return ACToUint32<ac_int<9, false> > (acustom / bcustom);
}


}

// Generated code for dtype: SINT10

extern "C" {

TVM_DLL uint32_t MinSINT10() {
  // return minimum representable value
  ac_int<10, true> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<10, true> >(min);
}


TVM_DLL float  SINT10ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<10, true> > (in).to_double();
  return static_cast<ac_int<10, true> > (custom_datatype);
}

TVM_DLL uint32_t FloatToSINT10(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<10, true> > (in);
}

TVM_DLL uint32_t SINT10Max(uint32_t a, uint32_t b) {
  // max
  ac_int<10, true> acustom = Uint32ToAC<ac_int<10, true> > (a);
  ac_int<10, true> bcustom = Uint32ToAC<ac_int<10, true> > (b);
  return ACToUint32<ac_int<10, true> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t SINT10Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<10, true> acustom = Uint32ToAC<ac_int<10, true> > (a);
  ac_int<10, true> bcustom = Uint32ToAC<ac_int<10, true> > (b);
  return ACToUint32<ac_int<10, true> > (acustom + bcustom);
}

TVM_DLL uint32_t SINT10Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<10, true> acustom = Uint32ToAC<ac_int<10, true> > (a);
  ac_int<10, true> bcustom = Uint32ToAC<ac_int<10, true> > (b);
  return ACToUint32<ac_int<10, true> > (acustom - bcustom);
}

TVM_DLL uint32_t SINT10Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<10, true> acustom = Uint32ToAC<ac_int<10, true> > (a);
  ac_int<10, true> bcustom = Uint32ToAC<ac_int<10, true> > (b);
  return ACToUint32<ac_int<10, true> > (acustom * bcustom);
}

TVM_DLL uint32_t SINT10Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<10, true> acustom = Uint32ToAC<ac_int<10, true> > (a);
  ac_int<10, true> bcustom = Uint32ToAC<ac_int<10, true> > (b);
  return ACToUint32<ac_int<10, true> > (acustom / bcustom);
}


}

// Generated code for dtype: UINT10

extern "C" {

TVM_DLL uint32_t MinUINT10() {
  // return minimum representable value
  ac_int<10, false> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<10, false> >(min);
}


TVM_DLL float  UINT10ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<10, false> > (in).to_double();
  return static_cast<ac_int<10, false> > (custom_datatype);
}

TVM_DLL uint32_t FloatToUINT10(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<10, false> > (in);
}

TVM_DLL uint32_t UINT10Max(uint32_t a, uint32_t b) {
  // max
  ac_int<10, false> acustom = Uint32ToAC<ac_int<10, false> > (a);
  ac_int<10, false> bcustom = Uint32ToAC<ac_int<10, false> > (b);
  return ACToUint32<ac_int<10, false> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t UINT10Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<10, false> acustom = Uint32ToAC<ac_int<10, false> > (a);
  ac_int<10, false> bcustom = Uint32ToAC<ac_int<10, false> > (b);
  return ACToUint32<ac_int<10, false> > (acustom + bcustom);
}

TVM_DLL uint32_t UINT10Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<10, false> acustom = Uint32ToAC<ac_int<10, false> > (a);
  ac_int<10, false> bcustom = Uint32ToAC<ac_int<10, false> > (b);
  return ACToUint32<ac_int<10, false> > (acustom - bcustom);
}

TVM_DLL uint32_t UINT10Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<10, false> acustom = Uint32ToAC<ac_int<10, false> > (a);
  ac_int<10, false> bcustom = Uint32ToAC<ac_int<10, false> > (b);
  return ACToUint32<ac_int<10, false> > (acustom * bcustom);
}

TVM_DLL uint32_t UINT10Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<10, false> acustom = Uint32ToAC<ac_int<10, false> > (a);
  ac_int<10, false> bcustom = Uint32ToAC<ac_int<10, false> > (b);
  return ACToUint32<ac_int<10, false> > (acustom / bcustom);
}


}

// Generated code for dtype: SINT11

extern "C" {

TVM_DLL uint32_t MinSINT11() {
  // return minimum representable value
  ac_int<11, true> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<11, true> >(min);
}


TVM_DLL float  SINT11ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<11, true> > (in).to_double();
  return static_cast<ac_int<11, true> > (custom_datatype);
}

TVM_DLL uint32_t FloatToSINT11(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<11, true> > (in);
}

TVM_DLL uint32_t SINT11Max(uint32_t a, uint32_t b) {
  // max
  ac_int<11, true> acustom = Uint32ToAC<ac_int<11, true> > (a);
  ac_int<11, true> bcustom = Uint32ToAC<ac_int<11, true> > (b);
  return ACToUint32<ac_int<11, true> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t SINT11Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<11, true> acustom = Uint32ToAC<ac_int<11, true> > (a);
  ac_int<11, true> bcustom = Uint32ToAC<ac_int<11, true> > (b);
  return ACToUint32<ac_int<11, true> > (acustom + bcustom);
}

TVM_DLL uint32_t SINT11Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<11, true> acustom = Uint32ToAC<ac_int<11, true> > (a);
  ac_int<11, true> bcustom = Uint32ToAC<ac_int<11, true> > (b);
  return ACToUint32<ac_int<11, true> > (acustom - bcustom);
}

TVM_DLL uint32_t SINT11Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<11, true> acustom = Uint32ToAC<ac_int<11, true> > (a);
  ac_int<11, true> bcustom = Uint32ToAC<ac_int<11, true> > (b);
  return ACToUint32<ac_int<11, true> > (acustom * bcustom);
}

TVM_DLL uint32_t SINT11Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<11, true> acustom = Uint32ToAC<ac_int<11, true> > (a);
  ac_int<11, true> bcustom = Uint32ToAC<ac_int<11, true> > (b);
  return ACToUint32<ac_int<11, true> > (acustom / bcustom);
}


}

// Generated code for dtype: UINT11

extern "C" {

TVM_DLL uint32_t MinUINT11() {
  // return minimum representable value
  ac_int<11, false> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<11, false> >(min);
}


TVM_DLL float  UINT11ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<11, false> > (in).to_double();
  return static_cast<ac_int<11, false> > (custom_datatype);
}

TVM_DLL uint32_t FloatToUINT11(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<11, false> > (in);
}

TVM_DLL uint32_t UINT11Max(uint32_t a, uint32_t b) {
  // max
  ac_int<11, false> acustom = Uint32ToAC<ac_int<11, false> > (a);
  ac_int<11, false> bcustom = Uint32ToAC<ac_int<11, false> > (b);
  return ACToUint32<ac_int<11, false> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t UINT11Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<11, false> acustom = Uint32ToAC<ac_int<11, false> > (a);
  ac_int<11, false> bcustom = Uint32ToAC<ac_int<11, false> > (b);
  return ACToUint32<ac_int<11, false> > (acustom + bcustom);
}

TVM_DLL uint32_t UINT11Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<11, false> acustom = Uint32ToAC<ac_int<11, false> > (a);
  ac_int<11, false> bcustom = Uint32ToAC<ac_int<11, false> > (b);
  return ACToUint32<ac_int<11, false> > (acustom - bcustom);
}

TVM_DLL uint32_t UINT11Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<11, false> acustom = Uint32ToAC<ac_int<11, false> > (a);
  ac_int<11, false> bcustom = Uint32ToAC<ac_int<11, false> > (b);
  return ACToUint32<ac_int<11, false> > (acustom * bcustom);
}

TVM_DLL uint32_t UINT11Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<11, false> acustom = Uint32ToAC<ac_int<11, false> > (a);
  ac_int<11, false> bcustom = Uint32ToAC<ac_int<11, false> > (b);
  return ACToUint32<ac_int<11, false> > (acustom / bcustom);
}


}

// Generated code for dtype: SINT12

extern "C" {

TVM_DLL uint32_t MinSINT12() {
  // return minimum representable value
  ac_int<12, true> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<12, true> >(min);
}


TVM_DLL float  SINT12ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<12, true> > (in).to_double();
  return static_cast<ac_int<12, true> > (custom_datatype);
}

TVM_DLL uint32_t FloatToSINT12(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<12, true> > (in);
}

TVM_DLL uint32_t SINT12Max(uint32_t a, uint32_t b) {
  // max
  ac_int<12, true> acustom = Uint32ToAC<ac_int<12, true> > (a);
  ac_int<12, true> bcustom = Uint32ToAC<ac_int<12, true> > (b);
  return ACToUint32<ac_int<12, true> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t SINT12Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<12, true> acustom = Uint32ToAC<ac_int<12, true> > (a);
  ac_int<12, true> bcustom = Uint32ToAC<ac_int<12, true> > (b);
  return ACToUint32<ac_int<12, true> > (acustom + bcustom);
}

TVM_DLL uint32_t SINT12Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<12, true> acustom = Uint32ToAC<ac_int<12, true> > (a);
  ac_int<12, true> bcustom = Uint32ToAC<ac_int<12, true> > (b);
  return ACToUint32<ac_int<12, true> > (acustom - bcustom);
}

TVM_DLL uint32_t SINT12Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<12, true> acustom = Uint32ToAC<ac_int<12, true> > (a);
  ac_int<12, true> bcustom = Uint32ToAC<ac_int<12, true> > (b);
  return ACToUint32<ac_int<12, true> > (acustom * bcustom);
}

TVM_DLL uint32_t SINT12Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<12, true> acustom = Uint32ToAC<ac_int<12, true> > (a);
  ac_int<12, true> bcustom = Uint32ToAC<ac_int<12, true> > (b);
  return ACToUint32<ac_int<12, true> > (acustom / bcustom);
}


}

// Generated code for dtype: UINT12

extern "C" {

TVM_DLL uint32_t MinUINT12() {
  // return minimum representable value
  ac_int<12, false> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<12, false> >(min);
}


TVM_DLL float  UINT12ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<12, false> > (in).to_double();
  return static_cast<ac_int<12, false> > (custom_datatype);
}

TVM_DLL uint32_t FloatToUINT12(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<12, false> > (in);
}

TVM_DLL uint32_t UINT12Max(uint32_t a, uint32_t b) {
  // max
  ac_int<12, false> acustom = Uint32ToAC<ac_int<12, false> > (a);
  ac_int<12, false> bcustom = Uint32ToAC<ac_int<12, false> > (b);
  return ACToUint32<ac_int<12, false> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t UINT12Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<12, false> acustom = Uint32ToAC<ac_int<12, false> > (a);
  ac_int<12, false> bcustom = Uint32ToAC<ac_int<12, false> > (b);
  return ACToUint32<ac_int<12, false> > (acustom + bcustom);
}

TVM_DLL uint32_t UINT12Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<12, false> acustom = Uint32ToAC<ac_int<12, false> > (a);
  ac_int<12, false> bcustom = Uint32ToAC<ac_int<12, false> > (b);
  return ACToUint32<ac_int<12, false> > (acustom - bcustom);
}

TVM_DLL uint32_t UINT12Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<12, false> acustom = Uint32ToAC<ac_int<12, false> > (a);
  ac_int<12, false> bcustom = Uint32ToAC<ac_int<12, false> > (b);
  return ACToUint32<ac_int<12, false> > (acustom * bcustom);
}

TVM_DLL uint32_t UINT12Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<12, false> acustom = Uint32ToAC<ac_int<12, false> > (a);
  ac_int<12, false> bcustom = Uint32ToAC<ac_int<12, false> > (b);
  return ACToUint32<ac_int<12, false> > (acustom / bcustom);
}


}

// Generated code for dtype: SINT13

extern "C" {

TVM_DLL uint32_t MinSINT13() {
  // return minimum representable value
  ac_int<13, true> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<13, true> >(min);
}


TVM_DLL float  SINT13ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<13, true> > (in).to_double();
  return static_cast<ac_int<13, true> > (custom_datatype);
}

TVM_DLL uint32_t FloatToSINT13(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<13, true> > (in);
}

TVM_DLL uint32_t SINT13Max(uint32_t a, uint32_t b) {
  // max
  ac_int<13, true> acustom = Uint32ToAC<ac_int<13, true> > (a);
  ac_int<13, true> bcustom = Uint32ToAC<ac_int<13, true> > (b);
  return ACToUint32<ac_int<13, true> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t SINT13Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<13, true> acustom = Uint32ToAC<ac_int<13, true> > (a);
  ac_int<13, true> bcustom = Uint32ToAC<ac_int<13, true> > (b);
  return ACToUint32<ac_int<13, true> > (acustom + bcustom);
}

TVM_DLL uint32_t SINT13Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<13, true> acustom = Uint32ToAC<ac_int<13, true> > (a);
  ac_int<13, true> bcustom = Uint32ToAC<ac_int<13, true> > (b);
  return ACToUint32<ac_int<13, true> > (acustom - bcustom);
}

TVM_DLL uint32_t SINT13Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<13, true> acustom = Uint32ToAC<ac_int<13, true> > (a);
  ac_int<13, true> bcustom = Uint32ToAC<ac_int<13, true> > (b);
  return ACToUint32<ac_int<13, true> > (acustom * bcustom);
}

TVM_DLL uint32_t SINT13Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<13, true> acustom = Uint32ToAC<ac_int<13, true> > (a);
  ac_int<13, true> bcustom = Uint32ToAC<ac_int<13, true> > (b);
  return ACToUint32<ac_int<13, true> > (acustom / bcustom);
}


}

// Generated code for dtype: UINT13

extern "C" {

TVM_DLL uint32_t MinUINT13() {
  // return minimum representable value
  ac_int<13, false> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<13, false> >(min);
}


TVM_DLL float  UINT13ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<13, false> > (in).to_double();
  return static_cast<ac_int<13, false> > (custom_datatype);
}

TVM_DLL uint32_t FloatToUINT13(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<13, false> > (in);
}

TVM_DLL uint32_t UINT13Max(uint32_t a, uint32_t b) {
  // max
  ac_int<13, false> acustom = Uint32ToAC<ac_int<13, false> > (a);
  ac_int<13, false> bcustom = Uint32ToAC<ac_int<13, false> > (b);
  return ACToUint32<ac_int<13, false> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t UINT13Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<13, false> acustom = Uint32ToAC<ac_int<13, false> > (a);
  ac_int<13, false> bcustom = Uint32ToAC<ac_int<13, false> > (b);
  return ACToUint32<ac_int<13, false> > (acustom + bcustom);
}

TVM_DLL uint32_t UINT13Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<13, false> acustom = Uint32ToAC<ac_int<13, false> > (a);
  ac_int<13, false> bcustom = Uint32ToAC<ac_int<13, false> > (b);
  return ACToUint32<ac_int<13, false> > (acustom - bcustom);
}

TVM_DLL uint32_t UINT13Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<13, false> acustom = Uint32ToAC<ac_int<13, false> > (a);
  ac_int<13, false> bcustom = Uint32ToAC<ac_int<13, false> > (b);
  return ACToUint32<ac_int<13, false> > (acustom * bcustom);
}

TVM_DLL uint32_t UINT13Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<13, false> acustom = Uint32ToAC<ac_int<13, false> > (a);
  ac_int<13, false> bcustom = Uint32ToAC<ac_int<13, false> > (b);
  return ACToUint32<ac_int<13, false> > (acustom / bcustom);
}


}

// Generated code for dtype: SINT14

extern "C" {

TVM_DLL uint32_t MinSINT14() {
  // return minimum representable value
  ac_int<14, true> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<14, true> >(min);
}


TVM_DLL float  SINT14ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<14, true> > (in).to_double();
  return static_cast<ac_int<14, true> > (custom_datatype);
}

TVM_DLL uint32_t FloatToSINT14(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<14, true> > (in);
}

TVM_DLL uint32_t SINT14Max(uint32_t a, uint32_t b) {
  // max
  ac_int<14, true> acustom = Uint32ToAC<ac_int<14, true> > (a);
  ac_int<14, true> bcustom = Uint32ToAC<ac_int<14, true> > (b);
  return ACToUint32<ac_int<14, true> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t SINT14Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<14, true> acustom = Uint32ToAC<ac_int<14, true> > (a);
  ac_int<14, true> bcustom = Uint32ToAC<ac_int<14, true> > (b);
  return ACToUint32<ac_int<14, true> > (acustom + bcustom);
}

TVM_DLL uint32_t SINT14Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<14, true> acustom = Uint32ToAC<ac_int<14, true> > (a);
  ac_int<14, true> bcustom = Uint32ToAC<ac_int<14, true> > (b);
  return ACToUint32<ac_int<14, true> > (acustom - bcustom);
}

TVM_DLL uint32_t SINT14Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<14, true> acustom = Uint32ToAC<ac_int<14, true> > (a);
  ac_int<14, true> bcustom = Uint32ToAC<ac_int<14, true> > (b);
  return ACToUint32<ac_int<14, true> > (acustom * bcustom);
}

TVM_DLL uint32_t SINT14Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<14, true> acustom = Uint32ToAC<ac_int<14, true> > (a);
  ac_int<14, true> bcustom = Uint32ToAC<ac_int<14, true> > (b);
  return ACToUint32<ac_int<14, true> > (acustom / bcustom);
}


}

// Generated code for dtype: UINT14

extern "C" {

TVM_DLL uint32_t MinUINT14() {
  // return minimum representable value
  ac_int<14, false> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<14, false> >(min);
}


TVM_DLL float  UINT14ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<14, false> > (in).to_double();
  return static_cast<ac_int<14, false> > (custom_datatype);
}

TVM_DLL uint32_t FloatToUINT14(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<14, false> > (in);
}

TVM_DLL uint32_t UINT14Max(uint32_t a, uint32_t b) {
  // max
  ac_int<14, false> acustom = Uint32ToAC<ac_int<14, false> > (a);
  ac_int<14, false> bcustom = Uint32ToAC<ac_int<14, false> > (b);
  return ACToUint32<ac_int<14, false> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t UINT14Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<14, false> acustom = Uint32ToAC<ac_int<14, false> > (a);
  ac_int<14, false> bcustom = Uint32ToAC<ac_int<14, false> > (b);
  return ACToUint32<ac_int<14, false> > (acustom + bcustom);
}

TVM_DLL uint32_t UINT14Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<14, false> acustom = Uint32ToAC<ac_int<14, false> > (a);
  ac_int<14, false> bcustom = Uint32ToAC<ac_int<14, false> > (b);
  return ACToUint32<ac_int<14, false> > (acustom - bcustom);
}

TVM_DLL uint32_t UINT14Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<14, false> acustom = Uint32ToAC<ac_int<14, false> > (a);
  ac_int<14, false> bcustom = Uint32ToAC<ac_int<14, false> > (b);
  return ACToUint32<ac_int<14, false> > (acustom * bcustom);
}

TVM_DLL uint32_t UINT14Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<14, false> acustom = Uint32ToAC<ac_int<14, false> > (a);
  ac_int<14, false> bcustom = Uint32ToAC<ac_int<14, false> > (b);
  return ACToUint32<ac_int<14, false> > (acustom / bcustom);
}


}

// Generated code for dtype: SINT15

extern "C" {

TVM_DLL uint32_t MinSINT15() {
  // return minimum representable value
  ac_int<15, true> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<15, true> >(min);
}


TVM_DLL float  SINT15ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<15, true> > (in).to_double();
  return static_cast<ac_int<15, true> > (custom_datatype);
}

TVM_DLL uint32_t FloatToSINT15(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<15, true> > (in);
}

TVM_DLL uint32_t SINT15Max(uint32_t a, uint32_t b) {
  // max
  ac_int<15, true> acustom = Uint32ToAC<ac_int<15, true> > (a);
  ac_int<15, true> bcustom = Uint32ToAC<ac_int<15, true> > (b);
  return ACToUint32<ac_int<15, true> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t SINT15Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<15, true> acustom = Uint32ToAC<ac_int<15, true> > (a);
  ac_int<15, true> bcustom = Uint32ToAC<ac_int<15, true> > (b);
  return ACToUint32<ac_int<15, true> > (acustom + bcustom);
}

TVM_DLL uint32_t SINT15Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<15, true> acustom = Uint32ToAC<ac_int<15, true> > (a);
  ac_int<15, true> bcustom = Uint32ToAC<ac_int<15, true> > (b);
  return ACToUint32<ac_int<15, true> > (acustom - bcustom);
}

TVM_DLL uint32_t SINT15Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<15, true> acustom = Uint32ToAC<ac_int<15, true> > (a);
  ac_int<15, true> bcustom = Uint32ToAC<ac_int<15, true> > (b);
  return ACToUint32<ac_int<15, true> > (acustom * bcustom);
}

TVM_DLL uint32_t SINT15Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<15, true> acustom = Uint32ToAC<ac_int<15, true> > (a);
  ac_int<15, true> bcustom = Uint32ToAC<ac_int<15, true> > (b);
  return ACToUint32<ac_int<15, true> > (acustom / bcustom);
}


}

// Generated code for dtype: UINT15

extern "C" {

TVM_DLL uint32_t MinUINT15() {
  // return minimum representable value
  ac_int<15, false> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<15, false> >(min);
}


TVM_DLL float  UINT15ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<15, false> > (in).to_double();
  return static_cast<ac_int<15, false> > (custom_datatype);
}

TVM_DLL uint32_t FloatToUINT15(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<15, false> > (in);
}

TVM_DLL uint32_t UINT15Max(uint32_t a, uint32_t b) {
  // max
  ac_int<15, false> acustom = Uint32ToAC<ac_int<15, false> > (a);
  ac_int<15, false> bcustom = Uint32ToAC<ac_int<15, false> > (b);
  return ACToUint32<ac_int<15, false> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t UINT15Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<15, false> acustom = Uint32ToAC<ac_int<15, false> > (a);
  ac_int<15, false> bcustom = Uint32ToAC<ac_int<15, false> > (b);
  return ACToUint32<ac_int<15, false> > (acustom + bcustom);
}

TVM_DLL uint32_t UINT15Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<15, false> acustom = Uint32ToAC<ac_int<15, false> > (a);
  ac_int<15, false> bcustom = Uint32ToAC<ac_int<15, false> > (b);
  return ACToUint32<ac_int<15, false> > (acustom - bcustom);
}

TVM_DLL uint32_t UINT15Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<15, false> acustom = Uint32ToAC<ac_int<15, false> > (a);
  ac_int<15, false> bcustom = Uint32ToAC<ac_int<15, false> > (b);
  return ACToUint32<ac_int<15, false> > (acustom * bcustom);
}

TVM_DLL uint32_t UINT15Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<15, false> acustom = Uint32ToAC<ac_int<15, false> > (a);
  ac_int<15, false> bcustom = Uint32ToAC<ac_int<15, false> > (b);
  return ACToUint32<ac_int<15, false> > (acustom / bcustom);
}


}

// Generated code for dtype: SINT16

extern "C" {

TVM_DLL uint32_t MinSINT16() {
  // return minimum representable value
  ac_int<16, true> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<16, true> >(min);
}


TVM_DLL float  SINT16ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<16, true> > (in).to_double();
  return static_cast<ac_int<16, true> > (custom_datatype);
}

TVM_DLL uint32_t FloatToSINT16(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<16, true> > (in);
}

TVM_DLL uint32_t SINT16Max(uint32_t a, uint32_t b) {
  // max
  ac_int<16, true> acustom = Uint32ToAC<ac_int<16, true> > (a);
  ac_int<16, true> bcustom = Uint32ToAC<ac_int<16, true> > (b);
  return ACToUint32<ac_int<16, true> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t SINT16Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<16, true> acustom = Uint32ToAC<ac_int<16, true> > (a);
  ac_int<16, true> bcustom = Uint32ToAC<ac_int<16, true> > (b);
  return ACToUint32<ac_int<16, true> > (acustom + bcustom);
}

TVM_DLL uint32_t SINT16Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<16, true> acustom = Uint32ToAC<ac_int<16, true> > (a);
  ac_int<16, true> bcustom = Uint32ToAC<ac_int<16, true> > (b);
  return ACToUint32<ac_int<16, true> > (acustom - bcustom);
}

TVM_DLL uint32_t SINT16Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<16, true> acustom = Uint32ToAC<ac_int<16, true> > (a);
  ac_int<16, true> bcustom = Uint32ToAC<ac_int<16, true> > (b);
  return ACToUint32<ac_int<16, true> > (acustom * bcustom);
}

TVM_DLL uint32_t SINT16Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<16, true> acustom = Uint32ToAC<ac_int<16, true> > (a);
  ac_int<16, true> bcustom = Uint32ToAC<ac_int<16, true> > (b);
  return ACToUint32<ac_int<16, true> > (acustom / bcustom);
}


}

// Generated code for dtype: UINT16

extern "C" {

TVM_DLL uint32_t MinUINT16() {
  // return minimum representable value
  ac_int<16, false> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<16, false> >(min);
}


TVM_DLL float  UINT16ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<16, false> > (in).to_double();
  return static_cast<ac_int<16, false> > (custom_datatype);
}

TVM_DLL uint32_t FloatToUINT16(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<16, false> > (in);
}

TVM_DLL uint32_t UINT16Max(uint32_t a, uint32_t b) {
  // max
  ac_int<16, false> acustom = Uint32ToAC<ac_int<16, false> > (a);
  ac_int<16, false> bcustom = Uint32ToAC<ac_int<16, false> > (b);
  return ACToUint32<ac_int<16, false> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t UINT16Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<16, false> acustom = Uint32ToAC<ac_int<16, false> > (a);
  ac_int<16, false> bcustom = Uint32ToAC<ac_int<16, false> > (b);
  return ACToUint32<ac_int<16, false> > (acustom + bcustom);
}

TVM_DLL uint32_t UINT16Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<16, false> acustom = Uint32ToAC<ac_int<16, false> > (a);
  ac_int<16, false> bcustom = Uint32ToAC<ac_int<16, false> > (b);
  return ACToUint32<ac_int<16, false> > (acustom - bcustom);
}

TVM_DLL uint32_t UINT16Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<16, false> acustom = Uint32ToAC<ac_int<16, false> > (a);
  ac_int<16, false> bcustom = Uint32ToAC<ac_int<16, false> > (b);
  return ACToUint32<ac_int<16, false> > (acustom * bcustom);
}

TVM_DLL uint32_t UINT16Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<16, false> acustom = Uint32ToAC<ac_int<16, false> > (a);
  ac_int<16, false> bcustom = Uint32ToAC<ac_int<16, false> > (b);
  return ACToUint32<ac_int<16, false> > (acustom / bcustom);
}


}

// Generated code for dtype: SINT17

extern "C" {

TVM_DLL uint32_t MinSINT17() {
  // return minimum representable value
  ac_int<17, true> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<17, true> >(min);
}


TVM_DLL float  SINT17ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<17, true> > (in).to_double();
  return static_cast<ac_int<17, true> > (custom_datatype);
}

TVM_DLL uint32_t FloatToSINT17(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<17, true> > (in);
}

TVM_DLL uint32_t SINT17Max(uint32_t a, uint32_t b) {
  // max
  ac_int<17, true> acustom = Uint32ToAC<ac_int<17, true> > (a);
  ac_int<17, true> bcustom = Uint32ToAC<ac_int<17, true> > (b);
  return ACToUint32<ac_int<17, true> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t SINT17Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<17, true> acustom = Uint32ToAC<ac_int<17, true> > (a);
  ac_int<17, true> bcustom = Uint32ToAC<ac_int<17, true> > (b);
  return ACToUint32<ac_int<17, true> > (acustom + bcustom);
}

TVM_DLL uint32_t SINT17Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<17, true> acustom = Uint32ToAC<ac_int<17, true> > (a);
  ac_int<17, true> bcustom = Uint32ToAC<ac_int<17, true> > (b);
  return ACToUint32<ac_int<17, true> > (acustom - bcustom);
}

TVM_DLL uint32_t SINT17Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<17, true> acustom = Uint32ToAC<ac_int<17, true> > (a);
  ac_int<17, true> bcustom = Uint32ToAC<ac_int<17, true> > (b);
  return ACToUint32<ac_int<17, true> > (acustom * bcustom);
}

TVM_DLL uint32_t SINT17Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<17, true> acustom = Uint32ToAC<ac_int<17, true> > (a);
  ac_int<17, true> bcustom = Uint32ToAC<ac_int<17, true> > (b);
  return ACToUint32<ac_int<17, true> > (acustom / bcustom);
}


}

// Generated code for dtype: UINT17

extern "C" {

TVM_DLL uint32_t MinUINT17() {
  // return minimum representable value
  ac_int<17, false> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<17, false> >(min);
}


TVM_DLL float  UINT17ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<17, false> > (in).to_double();
  return static_cast<ac_int<17, false> > (custom_datatype);
}

TVM_DLL uint32_t FloatToUINT17(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<17, false> > (in);
}

TVM_DLL uint32_t UINT17Max(uint32_t a, uint32_t b) {
  // max
  ac_int<17, false> acustom = Uint32ToAC<ac_int<17, false> > (a);
  ac_int<17, false> bcustom = Uint32ToAC<ac_int<17, false> > (b);
  return ACToUint32<ac_int<17, false> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t UINT17Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<17, false> acustom = Uint32ToAC<ac_int<17, false> > (a);
  ac_int<17, false> bcustom = Uint32ToAC<ac_int<17, false> > (b);
  return ACToUint32<ac_int<17, false> > (acustom + bcustom);
}

TVM_DLL uint32_t UINT17Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<17, false> acustom = Uint32ToAC<ac_int<17, false> > (a);
  ac_int<17, false> bcustom = Uint32ToAC<ac_int<17, false> > (b);
  return ACToUint32<ac_int<17, false> > (acustom - bcustom);
}

TVM_DLL uint32_t UINT17Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<17, false> acustom = Uint32ToAC<ac_int<17, false> > (a);
  ac_int<17, false> bcustom = Uint32ToAC<ac_int<17, false> > (b);
  return ACToUint32<ac_int<17, false> > (acustom * bcustom);
}

TVM_DLL uint32_t UINT17Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<17, false> acustom = Uint32ToAC<ac_int<17, false> > (a);
  ac_int<17, false> bcustom = Uint32ToAC<ac_int<17, false> > (b);
  return ACToUint32<ac_int<17, false> > (acustom / bcustom);
}


}

// Generated code for dtype: SINT18

extern "C" {

TVM_DLL uint32_t MinSINT18() {
  // return minimum representable value
  ac_int<18, true> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<18, true> >(min);
}


TVM_DLL float  SINT18ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<18, true> > (in).to_double();
  return static_cast<ac_int<18, true> > (custom_datatype);
}

TVM_DLL uint32_t FloatToSINT18(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<18, true> > (in);
}

TVM_DLL uint32_t SINT18Max(uint32_t a, uint32_t b) {
  // max
  ac_int<18, true> acustom = Uint32ToAC<ac_int<18, true> > (a);
  ac_int<18, true> bcustom = Uint32ToAC<ac_int<18, true> > (b);
  return ACToUint32<ac_int<18, true> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t SINT18Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<18, true> acustom = Uint32ToAC<ac_int<18, true> > (a);
  ac_int<18, true> bcustom = Uint32ToAC<ac_int<18, true> > (b);
  return ACToUint32<ac_int<18, true> > (acustom + bcustom);
}

TVM_DLL uint32_t SINT18Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<18, true> acustom = Uint32ToAC<ac_int<18, true> > (a);
  ac_int<18, true> bcustom = Uint32ToAC<ac_int<18, true> > (b);
  return ACToUint32<ac_int<18, true> > (acustom - bcustom);
}

TVM_DLL uint32_t SINT18Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<18, true> acustom = Uint32ToAC<ac_int<18, true> > (a);
  ac_int<18, true> bcustom = Uint32ToAC<ac_int<18, true> > (b);
  return ACToUint32<ac_int<18, true> > (acustom * bcustom);
}

TVM_DLL uint32_t SINT18Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<18, true> acustom = Uint32ToAC<ac_int<18, true> > (a);
  ac_int<18, true> bcustom = Uint32ToAC<ac_int<18, true> > (b);
  return ACToUint32<ac_int<18, true> > (acustom / bcustom);
}


}

// Generated code for dtype: UINT18

extern "C" {

TVM_DLL uint32_t MinUINT18() {
  // return minimum representable value
  ac_int<18, false> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<18, false> >(min);
}


TVM_DLL float  UINT18ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<18, false> > (in).to_double();
  return static_cast<ac_int<18, false> > (custom_datatype);
}

TVM_DLL uint32_t FloatToUINT18(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<18, false> > (in);
}

TVM_DLL uint32_t UINT18Max(uint32_t a, uint32_t b) {
  // max
  ac_int<18, false> acustom = Uint32ToAC<ac_int<18, false> > (a);
  ac_int<18, false> bcustom = Uint32ToAC<ac_int<18, false> > (b);
  return ACToUint32<ac_int<18, false> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t UINT18Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<18, false> acustom = Uint32ToAC<ac_int<18, false> > (a);
  ac_int<18, false> bcustom = Uint32ToAC<ac_int<18, false> > (b);
  return ACToUint32<ac_int<18, false> > (acustom + bcustom);
}

TVM_DLL uint32_t UINT18Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<18, false> acustom = Uint32ToAC<ac_int<18, false> > (a);
  ac_int<18, false> bcustom = Uint32ToAC<ac_int<18, false> > (b);
  return ACToUint32<ac_int<18, false> > (acustom - bcustom);
}

TVM_DLL uint32_t UINT18Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<18, false> acustom = Uint32ToAC<ac_int<18, false> > (a);
  ac_int<18, false> bcustom = Uint32ToAC<ac_int<18, false> > (b);
  return ACToUint32<ac_int<18, false> > (acustom * bcustom);
}

TVM_DLL uint32_t UINT18Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<18, false> acustom = Uint32ToAC<ac_int<18, false> > (a);
  ac_int<18, false> bcustom = Uint32ToAC<ac_int<18, false> > (b);
  return ACToUint32<ac_int<18, false> > (acustom / bcustom);
}


}

// Generated code for dtype: SINT19

extern "C" {

TVM_DLL uint32_t MinSINT19() {
  // return minimum representable value
  ac_int<19, true> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<19, true> >(min);
}


TVM_DLL float  SINT19ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<19, true> > (in).to_double();
  return static_cast<ac_int<19, true> > (custom_datatype);
}

TVM_DLL uint32_t FloatToSINT19(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<19, true> > (in);
}

TVM_DLL uint32_t SINT19Max(uint32_t a, uint32_t b) {
  // max
  ac_int<19, true> acustom = Uint32ToAC<ac_int<19, true> > (a);
  ac_int<19, true> bcustom = Uint32ToAC<ac_int<19, true> > (b);
  return ACToUint32<ac_int<19, true> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t SINT19Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<19, true> acustom = Uint32ToAC<ac_int<19, true> > (a);
  ac_int<19, true> bcustom = Uint32ToAC<ac_int<19, true> > (b);
  return ACToUint32<ac_int<19, true> > (acustom + bcustom);
}

TVM_DLL uint32_t SINT19Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<19, true> acustom = Uint32ToAC<ac_int<19, true> > (a);
  ac_int<19, true> bcustom = Uint32ToAC<ac_int<19, true> > (b);
  return ACToUint32<ac_int<19, true> > (acustom - bcustom);
}

TVM_DLL uint32_t SINT19Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<19, true> acustom = Uint32ToAC<ac_int<19, true> > (a);
  ac_int<19, true> bcustom = Uint32ToAC<ac_int<19, true> > (b);
  return ACToUint32<ac_int<19, true> > (acustom * bcustom);
}

TVM_DLL uint32_t SINT19Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<19, true> acustom = Uint32ToAC<ac_int<19, true> > (a);
  ac_int<19, true> bcustom = Uint32ToAC<ac_int<19, true> > (b);
  return ACToUint32<ac_int<19, true> > (acustom / bcustom);
}


}

// Generated code for dtype: UINT19

extern "C" {

TVM_DLL uint32_t MinUINT19() {
  // return minimum representable value
  ac_int<19, false> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<19, false> >(min);
}


TVM_DLL float  UINT19ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<19, false> > (in).to_double();
  return static_cast<ac_int<19, false> > (custom_datatype);
}

TVM_DLL uint32_t FloatToUINT19(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<19, false> > (in);
}

TVM_DLL uint32_t UINT19Max(uint32_t a, uint32_t b) {
  // max
  ac_int<19, false> acustom = Uint32ToAC<ac_int<19, false> > (a);
  ac_int<19, false> bcustom = Uint32ToAC<ac_int<19, false> > (b);
  return ACToUint32<ac_int<19, false> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t UINT19Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<19, false> acustom = Uint32ToAC<ac_int<19, false> > (a);
  ac_int<19, false> bcustom = Uint32ToAC<ac_int<19, false> > (b);
  return ACToUint32<ac_int<19, false> > (acustom + bcustom);
}

TVM_DLL uint32_t UINT19Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<19, false> acustom = Uint32ToAC<ac_int<19, false> > (a);
  ac_int<19, false> bcustom = Uint32ToAC<ac_int<19, false> > (b);
  return ACToUint32<ac_int<19, false> > (acustom - bcustom);
}

TVM_DLL uint32_t UINT19Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<19, false> acustom = Uint32ToAC<ac_int<19, false> > (a);
  ac_int<19, false> bcustom = Uint32ToAC<ac_int<19, false> > (b);
  return ACToUint32<ac_int<19, false> > (acustom * bcustom);
}

TVM_DLL uint32_t UINT19Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<19, false> acustom = Uint32ToAC<ac_int<19, false> > (a);
  ac_int<19, false> bcustom = Uint32ToAC<ac_int<19, false> > (b);
  return ACToUint32<ac_int<19, false> > (acustom / bcustom);
}


}

// Generated code for dtype: SINT20

extern "C" {

TVM_DLL uint32_t MinSINT20() {
  // return minimum representable value
  ac_int<20, true> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<20, true> >(min);
}


TVM_DLL float  SINT20ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<20, true> > (in).to_double();
  return static_cast<ac_int<20, true> > (custom_datatype);
}

TVM_DLL uint32_t FloatToSINT20(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<20, true> > (in);
}

TVM_DLL uint32_t SINT20Max(uint32_t a, uint32_t b) {
  // max
  ac_int<20, true> acustom = Uint32ToAC<ac_int<20, true> > (a);
  ac_int<20, true> bcustom = Uint32ToAC<ac_int<20, true> > (b);
  return ACToUint32<ac_int<20, true> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t SINT20Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<20, true> acustom = Uint32ToAC<ac_int<20, true> > (a);
  ac_int<20, true> bcustom = Uint32ToAC<ac_int<20, true> > (b);
  return ACToUint32<ac_int<20, true> > (acustom + bcustom);
}

TVM_DLL uint32_t SINT20Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<20, true> acustom = Uint32ToAC<ac_int<20, true> > (a);
  ac_int<20, true> bcustom = Uint32ToAC<ac_int<20, true> > (b);
  return ACToUint32<ac_int<20, true> > (acustom - bcustom);
}

TVM_DLL uint32_t SINT20Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<20, true> acustom = Uint32ToAC<ac_int<20, true> > (a);
  ac_int<20, true> bcustom = Uint32ToAC<ac_int<20, true> > (b);
  return ACToUint32<ac_int<20, true> > (acustom * bcustom);
}

TVM_DLL uint32_t SINT20Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<20, true> acustom = Uint32ToAC<ac_int<20, true> > (a);
  ac_int<20, true> bcustom = Uint32ToAC<ac_int<20, true> > (b);
  return ACToUint32<ac_int<20, true> > (acustom / bcustom);
}


}

// Generated code for dtype: UINT20

extern "C" {

TVM_DLL uint32_t MinUINT20() {
  // return minimum representable value
  ac_int<20, false> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<20, false> >(min);
}


TVM_DLL float  UINT20ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<20, false> > (in).to_double();
  return static_cast<ac_int<20, false> > (custom_datatype);
}

TVM_DLL uint32_t FloatToUINT20(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<20, false> > (in);
}

TVM_DLL uint32_t UINT20Max(uint32_t a, uint32_t b) {
  // max
  ac_int<20, false> acustom = Uint32ToAC<ac_int<20, false> > (a);
  ac_int<20, false> bcustom = Uint32ToAC<ac_int<20, false> > (b);
  return ACToUint32<ac_int<20, false> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t UINT20Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<20, false> acustom = Uint32ToAC<ac_int<20, false> > (a);
  ac_int<20, false> bcustom = Uint32ToAC<ac_int<20, false> > (b);
  return ACToUint32<ac_int<20, false> > (acustom + bcustom);
}

TVM_DLL uint32_t UINT20Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<20, false> acustom = Uint32ToAC<ac_int<20, false> > (a);
  ac_int<20, false> bcustom = Uint32ToAC<ac_int<20, false> > (b);
  return ACToUint32<ac_int<20, false> > (acustom - bcustom);
}

TVM_DLL uint32_t UINT20Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<20, false> acustom = Uint32ToAC<ac_int<20, false> > (a);
  ac_int<20, false> bcustom = Uint32ToAC<ac_int<20, false> > (b);
  return ACToUint32<ac_int<20, false> > (acustom * bcustom);
}

TVM_DLL uint32_t UINT20Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<20, false> acustom = Uint32ToAC<ac_int<20, false> > (a);
  ac_int<20, false> bcustom = Uint32ToAC<ac_int<20, false> > (b);
  return ACToUint32<ac_int<20, false> > (acustom / bcustom);
}


}

// Generated code for dtype: SINT21

extern "C" {

TVM_DLL uint32_t MinSINT21() {
  // return minimum representable value
  ac_int<21, true> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<21, true> >(min);
}


TVM_DLL float  SINT21ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<21, true> > (in).to_double();
  return static_cast<ac_int<21, true> > (custom_datatype);
}

TVM_DLL uint32_t FloatToSINT21(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<21, true> > (in);
}

TVM_DLL uint32_t SINT21Max(uint32_t a, uint32_t b) {
  // max
  ac_int<21, true> acustom = Uint32ToAC<ac_int<21, true> > (a);
  ac_int<21, true> bcustom = Uint32ToAC<ac_int<21, true> > (b);
  return ACToUint32<ac_int<21, true> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t SINT21Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<21, true> acustom = Uint32ToAC<ac_int<21, true> > (a);
  ac_int<21, true> bcustom = Uint32ToAC<ac_int<21, true> > (b);
  return ACToUint32<ac_int<21, true> > (acustom + bcustom);
}

TVM_DLL uint32_t SINT21Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<21, true> acustom = Uint32ToAC<ac_int<21, true> > (a);
  ac_int<21, true> bcustom = Uint32ToAC<ac_int<21, true> > (b);
  return ACToUint32<ac_int<21, true> > (acustom - bcustom);
}

TVM_DLL uint32_t SINT21Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<21, true> acustom = Uint32ToAC<ac_int<21, true> > (a);
  ac_int<21, true> bcustom = Uint32ToAC<ac_int<21, true> > (b);
  return ACToUint32<ac_int<21, true> > (acustom * bcustom);
}

TVM_DLL uint32_t SINT21Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<21, true> acustom = Uint32ToAC<ac_int<21, true> > (a);
  ac_int<21, true> bcustom = Uint32ToAC<ac_int<21, true> > (b);
  return ACToUint32<ac_int<21, true> > (acustom / bcustom);
}


}

// Generated code for dtype: UINT21

extern "C" {

TVM_DLL uint32_t MinUINT21() {
  // return minimum representable value
  ac_int<21, false> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<21, false> >(min);
}


TVM_DLL float  UINT21ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<21, false> > (in).to_double();
  return static_cast<ac_int<21, false> > (custom_datatype);
}

TVM_DLL uint32_t FloatToUINT21(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<21, false> > (in);
}

TVM_DLL uint32_t UINT21Max(uint32_t a, uint32_t b) {
  // max
  ac_int<21, false> acustom = Uint32ToAC<ac_int<21, false> > (a);
  ac_int<21, false> bcustom = Uint32ToAC<ac_int<21, false> > (b);
  return ACToUint32<ac_int<21, false> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t UINT21Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<21, false> acustom = Uint32ToAC<ac_int<21, false> > (a);
  ac_int<21, false> bcustom = Uint32ToAC<ac_int<21, false> > (b);
  return ACToUint32<ac_int<21, false> > (acustom + bcustom);
}

TVM_DLL uint32_t UINT21Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<21, false> acustom = Uint32ToAC<ac_int<21, false> > (a);
  ac_int<21, false> bcustom = Uint32ToAC<ac_int<21, false> > (b);
  return ACToUint32<ac_int<21, false> > (acustom - bcustom);
}

TVM_DLL uint32_t UINT21Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<21, false> acustom = Uint32ToAC<ac_int<21, false> > (a);
  ac_int<21, false> bcustom = Uint32ToAC<ac_int<21, false> > (b);
  return ACToUint32<ac_int<21, false> > (acustom * bcustom);
}

TVM_DLL uint32_t UINT21Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<21, false> acustom = Uint32ToAC<ac_int<21, false> > (a);
  ac_int<21, false> bcustom = Uint32ToAC<ac_int<21, false> > (b);
  return ACToUint32<ac_int<21, false> > (acustom / bcustom);
}


}

// Generated code for dtype: SINT22

extern "C" {

TVM_DLL uint32_t MinSINT22() {
  // return minimum representable value
  ac_int<22, true> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<22, true> >(min);
}


TVM_DLL float  SINT22ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<22, true> > (in).to_double();
  return static_cast<ac_int<22, true> > (custom_datatype);
}

TVM_DLL uint32_t FloatToSINT22(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<22, true> > (in);
}

TVM_DLL uint32_t SINT22Max(uint32_t a, uint32_t b) {
  // max
  ac_int<22, true> acustom = Uint32ToAC<ac_int<22, true> > (a);
  ac_int<22, true> bcustom = Uint32ToAC<ac_int<22, true> > (b);
  return ACToUint32<ac_int<22, true> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t SINT22Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<22, true> acustom = Uint32ToAC<ac_int<22, true> > (a);
  ac_int<22, true> bcustom = Uint32ToAC<ac_int<22, true> > (b);
  return ACToUint32<ac_int<22, true> > (acustom + bcustom);
}

TVM_DLL uint32_t SINT22Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<22, true> acustom = Uint32ToAC<ac_int<22, true> > (a);
  ac_int<22, true> bcustom = Uint32ToAC<ac_int<22, true> > (b);
  return ACToUint32<ac_int<22, true> > (acustom - bcustom);
}

TVM_DLL uint32_t SINT22Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<22, true> acustom = Uint32ToAC<ac_int<22, true> > (a);
  ac_int<22, true> bcustom = Uint32ToAC<ac_int<22, true> > (b);
  return ACToUint32<ac_int<22, true> > (acustom * bcustom);
}

TVM_DLL uint32_t SINT22Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<22, true> acustom = Uint32ToAC<ac_int<22, true> > (a);
  ac_int<22, true> bcustom = Uint32ToAC<ac_int<22, true> > (b);
  return ACToUint32<ac_int<22, true> > (acustom / bcustom);
}


}

// Generated code for dtype: UINT22

extern "C" {

TVM_DLL uint32_t MinUINT22() {
  // return minimum representable value
  ac_int<22, false> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<22, false> >(min);
}


TVM_DLL float  UINT22ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<22, false> > (in).to_double();
  return static_cast<ac_int<22, false> > (custom_datatype);
}

TVM_DLL uint32_t FloatToUINT22(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<22, false> > (in);
}

TVM_DLL uint32_t UINT22Max(uint32_t a, uint32_t b) {
  // max
  ac_int<22, false> acustom = Uint32ToAC<ac_int<22, false> > (a);
  ac_int<22, false> bcustom = Uint32ToAC<ac_int<22, false> > (b);
  return ACToUint32<ac_int<22, false> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t UINT22Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<22, false> acustom = Uint32ToAC<ac_int<22, false> > (a);
  ac_int<22, false> bcustom = Uint32ToAC<ac_int<22, false> > (b);
  return ACToUint32<ac_int<22, false> > (acustom + bcustom);
}

TVM_DLL uint32_t UINT22Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<22, false> acustom = Uint32ToAC<ac_int<22, false> > (a);
  ac_int<22, false> bcustom = Uint32ToAC<ac_int<22, false> > (b);
  return ACToUint32<ac_int<22, false> > (acustom - bcustom);
}

TVM_DLL uint32_t UINT22Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<22, false> acustom = Uint32ToAC<ac_int<22, false> > (a);
  ac_int<22, false> bcustom = Uint32ToAC<ac_int<22, false> > (b);
  return ACToUint32<ac_int<22, false> > (acustom * bcustom);
}

TVM_DLL uint32_t UINT22Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<22, false> acustom = Uint32ToAC<ac_int<22, false> > (a);
  ac_int<22, false> bcustom = Uint32ToAC<ac_int<22, false> > (b);
  return ACToUint32<ac_int<22, false> > (acustom / bcustom);
}


}

// Generated code for dtype: SINT23

extern "C" {

TVM_DLL uint32_t MinSINT23() {
  // return minimum representable value
  ac_int<23, true> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<23, true> >(min);
}


TVM_DLL float  SINT23ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<23, true> > (in).to_double();
  return static_cast<ac_int<23, true> > (custom_datatype);
}

TVM_DLL uint32_t FloatToSINT23(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<23, true> > (in);
}

TVM_DLL uint32_t SINT23Max(uint32_t a, uint32_t b) {
  // max
  ac_int<23, true> acustom = Uint32ToAC<ac_int<23, true> > (a);
  ac_int<23, true> bcustom = Uint32ToAC<ac_int<23, true> > (b);
  return ACToUint32<ac_int<23, true> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t SINT23Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<23, true> acustom = Uint32ToAC<ac_int<23, true> > (a);
  ac_int<23, true> bcustom = Uint32ToAC<ac_int<23, true> > (b);
  return ACToUint32<ac_int<23, true> > (acustom + bcustom);
}

TVM_DLL uint32_t SINT23Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<23, true> acustom = Uint32ToAC<ac_int<23, true> > (a);
  ac_int<23, true> bcustom = Uint32ToAC<ac_int<23, true> > (b);
  return ACToUint32<ac_int<23, true> > (acustom - bcustom);
}

TVM_DLL uint32_t SINT23Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<23, true> acustom = Uint32ToAC<ac_int<23, true> > (a);
  ac_int<23, true> bcustom = Uint32ToAC<ac_int<23, true> > (b);
  return ACToUint32<ac_int<23, true> > (acustom * bcustom);
}

TVM_DLL uint32_t SINT23Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<23, true> acustom = Uint32ToAC<ac_int<23, true> > (a);
  ac_int<23, true> bcustom = Uint32ToAC<ac_int<23, true> > (b);
  return ACToUint32<ac_int<23, true> > (acustom / bcustom);
}


}

// Generated code for dtype: UINT23

extern "C" {

TVM_DLL uint32_t MinUINT23() {
  // return minimum representable value
  ac_int<23, false> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<23, false> >(min);
}


TVM_DLL float  UINT23ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<23, false> > (in).to_double();
  return static_cast<ac_int<23, false> > (custom_datatype);
}

TVM_DLL uint32_t FloatToUINT23(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<23, false> > (in);
}

TVM_DLL uint32_t UINT23Max(uint32_t a, uint32_t b) {
  // max
  ac_int<23, false> acustom = Uint32ToAC<ac_int<23, false> > (a);
  ac_int<23, false> bcustom = Uint32ToAC<ac_int<23, false> > (b);
  return ACToUint32<ac_int<23, false> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t UINT23Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<23, false> acustom = Uint32ToAC<ac_int<23, false> > (a);
  ac_int<23, false> bcustom = Uint32ToAC<ac_int<23, false> > (b);
  return ACToUint32<ac_int<23, false> > (acustom + bcustom);
}

TVM_DLL uint32_t UINT23Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<23, false> acustom = Uint32ToAC<ac_int<23, false> > (a);
  ac_int<23, false> bcustom = Uint32ToAC<ac_int<23, false> > (b);
  return ACToUint32<ac_int<23, false> > (acustom - bcustom);
}

TVM_DLL uint32_t UINT23Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<23, false> acustom = Uint32ToAC<ac_int<23, false> > (a);
  ac_int<23, false> bcustom = Uint32ToAC<ac_int<23, false> > (b);
  return ACToUint32<ac_int<23, false> > (acustom * bcustom);
}

TVM_DLL uint32_t UINT23Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<23, false> acustom = Uint32ToAC<ac_int<23, false> > (a);
  ac_int<23, false> bcustom = Uint32ToAC<ac_int<23, false> > (b);
  return ACToUint32<ac_int<23, false> > (acustom / bcustom);
}


}

// Generated code for dtype: SINT24

extern "C" {

TVM_DLL uint32_t MinSINT24() {
  // return minimum representable value
  ac_int<24, true> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<24, true> >(min);
}


TVM_DLL float  SINT24ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<24, true> > (in).to_double();
  return static_cast<ac_int<24, true> > (custom_datatype);
}

TVM_DLL uint32_t FloatToSINT24(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<24, true> > (in);
}

TVM_DLL uint32_t SINT24Max(uint32_t a, uint32_t b) {
  // max
  ac_int<24, true> acustom = Uint32ToAC<ac_int<24, true> > (a);
  ac_int<24, true> bcustom = Uint32ToAC<ac_int<24, true> > (b);
  return ACToUint32<ac_int<24, true> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t SINT24Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<24, true> acustom = Uint32ToAC<ac_int<24, true> > (a);
  ac_int<24, true> bcustom = Uint32ToAC<ac_int<24, true> > (b);
  return ACToUint32<ac_int<24, true> > (acustom + bcustom);
}

TVM_DLL uint32_t SINT24Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<24, true> acustom = Uint32ToAC<ac_int<24, true> > (a);
  ac_int<24, true> bcustom = Uint32ToAC<ac_int<24, true> > (b);
  return ACToUint32<ac_int<24, true> > (acustom - bcustom);
}

TVM_DLL uint32_t SINT24Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<24, true> acustom = Uint32ToAC<ac_int<24, true> > (a);
  ac_int<24, true> bcustom = Uint32ToAC<ac_int<24, true> > (b);
  return ACToUint32<ac_int<24, true> > (acustom * bcustom);
}

TVM_DLL uint32_t SINT24Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<24, true> acustom = Uint32ToAC<ac_int<24, true> > (a);
  ac_int<24, true> bcustom = Uint32ToAC<ac_int<24, true> > (b);
  return ACToUint32<ac_int<24, true> > (acustom / bcustom);
}


}

// Generated code for dtype: UINT24

extern "C" {

TVM_DLL uint32_t MinUINT24() {
  // return minimum representable value
  ac_int<24, false> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<24, false> >(min);
}


TVM_DLL float  UINT24ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<24, false> > (in).to_double();
  return static_cast<ac_int<24, false> > (custom_datatype);
}

TVM_DLL uint32_t FloatToUINT24(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<24, false> > (in);
}

TVM_DLL uint32_t UINT24Max(uint32_t a, uint32_t b) {
  // max
  ac_int<24, false> acustom = Uint32ToAC<ac_int<24, false> > (a);
  ac_int<24, false> bcustom = Uint32ToAC<ac_int<24, false> > (b);
  return ACToUint32<ac_int<24, false> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t UINT24Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<24, false> acustom = Uint32ToAC<ac_int<24, false> > (a);
  ac_int<24, false> bcustom = Uint32ToAC<ac_int<24, false> > (b);
  return ACToUint32<ac_int<24, false> > (acustom + bcustom);
}

TVM_DLL uint32_t UINT24Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<24, false> acustom = Uint32ToAC<ac_int<24, false> > (a);
  ac_int<24, false> bcustom = Uint32ToAC<ac_int<24, false> > (b);
  return ACToUint32<ac_int<24, false> > (acustom - bcustom);
}

TVM_DLL uint32_t UINT24Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<24, false> acustom = Uint32ToAC<ac_int<24, false> > (a);
  ac_int<24, false> bcustom = Uint32ToAC<ac_int<24, false> > (b);
  return ACToUint32<ac_int<24, false> > (acustom * bcustom);
}

TVM_DLL uint32_t UINT24Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<24, false> acustom = Uint32ToAC<ac_int<24, false> > (a);
  ac_int<24, false> bcustom = Uint32ToAC<ac_int<24, false> > (b);
  return ACToUint32<ac_int<24, false> > (acustom / bcustom);
}


}

// Generated code for dtype: SINT25

extern "C" {

TVM_DLL uint32_t MinSINT25() {
  // return minimum representable value
  ac_int<25, true> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<25, true> >(min);
}


TVM_DLL float  SINT25ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<25, true> > (in).to_double();
  return static_cast<ac_int<25, true> > (custom_datatype);
}

TVM_DLL uint32_t FloatToSINT25(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<25, true> > (in);
}

TVM_DLL uint32_t SINT25Max(uint32_t a, uint32_t b) {
  // max
  ac_int<25, true> acustom = Uint32ToAC<ac_int<25, true> > (a);
  ac_int<25, true> bcustom = Uint32ToAC<ac_int<25, true> > (b);
  return ACToUint32<ac_int<25, true> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t SINT25Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<25, true> acustom = Uint32ToAC<ac_int<25, true> > (a);
  ac_int<25, true> bcustom = Uint32ToAC<ac_int<25, true> > (b);
  return ACToUint32<ac_int<25, true> > (acustom + bcustom);
}

TVM_DLL uint32_t SINT25Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<25, true> acustom = Uint32ToAC<ac_int<25, true> > (a);
  ac_int<25, true> bcustom = Uint32ToAC<ac_int<25, true> > (b);
  return ACToUint32<ac_int<25, true> > (acustom - bcustom);
}

TVM_DLL uint32_t SINT25Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<25, true> acustom = Uint32ToAC<ac_int<25, true> > (a);
  ac_int<25, true> bcustom = Uint32ToAC<ac_int<25, true> > (b);
  return ACToUint32<ac_int<25, true> > (acustom * bcustom);
}

TVM_DLL uint32_t SINT25Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<25, true> acustom = Uint32ToAC<ac_int<25, true> > (a);
  ac_int<25, true> bcustom = Uint32ToAC<ac_int<25, true> > (b);
  return ACToUint32<ac_int<25, true> > (acustom / bcustom);
}


}

// Generated code for dtype: UINT25

extern "C" {

TVM_DLL uint32_t MinUINT25() {
  // return minimum representable value
  ac_int<25, false> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<25, false> >(min);
}


TVM_DLL float  UINT25ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<25, false> > (in).to_double();
  return static_cast<ac_int<25, false> > (custom_datatype);
}

TVM_DLL uint32_t FloatToUINT25(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<25, false> > (in);
}

TVM_DLL uint32_t UINT25Max(uint32_t a, uint32_t b) {
  // max
  ac_int<25, false> acustom = Uint32ToAC<ac_int<25, false> > (a);
  ac_int<25, false> bcustom = Uint32ToAC<ac_int<25, false> > (b);
  return ACToUint32<ac_int<25, false> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t UINT25Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<25, false> acustom = Uint32ToAC<ac_int<25, false> > (a);
  ac_int<25, false> bcustom = Uint32ToAC<ac_int<25, false> > (b);
  return ACToUint32<ac_int<25, false> > (acustom + bcustom);
}

TVM_DLL uint32_t UINT25Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<25, false> acustom = Uint32ToAC<ac_int<25, false> > (a);
  ac_int<25, false> bcustom = Uint32ToAC<ac_int<25, false> > (b);
  return ACToUint32<ac_int<25, false> > (acustom - bcustom);
}

TVM_DLL uint32_t UINT25Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<25, false> acustom = Uint32ToAC<ac_int<25, false> > (a);
  ac_int<25, false> bcustom = Uint32ToAC<ac_int<25, false> > (b);
  return ACToUint32<ac_int<25, false> > (acustom * bcustom);
}

TVM_DLL uint32_t UINT25Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<25, false> acustom = Uint32ToAC<ac_int<25, false> > (a);
  ac_int<25, false> bcustom = Uint32ToAC<ac_int<25, false> > (b);
  return ACToUint32<ac_int<25, false> > (acustom / bcustom);
}


}

// Generated code for dtype: SINT26

extern "C" {

TVM_DLL uint32_t MinSINT26() {
  // return minimum representable value
  ac_int<26, true> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<26, true> >(min);
}


TVM_DLL float  SINT26ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<26, true> > (in).to_double();
  return static_cast<ac_int<26, true> > (custom_datatype);
}

TVM_DLL uint32_t FloatToSINT26(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<26, true> > (in);
}

TVM_DLL uint32_t SINT26Max(uint32_t a, uint32_t b) {
  // max
  ac_int<26, true> acustom = Uint32ToAC<ac_int<26, true> > (a);
  ac_int<26, true> bcustom = Uint32ToAC<ac_int<26, true> > (b);
  return ACToUint32<ac_int<26, true> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t SINT26Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<26, true> acustom = Uint32ToAC<ac_int<26, true> > (a);
  ac_int<26, true> bcustom = Uint32ToAC<ac_int<26, true> > (b);
  return ACToUint32<ac_int<26, true> > (acustom + bcustom);
}

TVM_DLL uint32_t SINT26Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<26, true> acustom = Uint32ToAC<ac_int<26, true> > (a);
  ac_int<26, true> bcustom = Uint32ToAC<ac_int<26, true> > (b);
  return ACToUint32<ac_int<26, true> > (acustom - bcustom);
}

TVM_DLL uint32_t SINT26Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<26, true> acustom = Uint32ToAC<ac_int<26, true> > (a);
  ac_int<26, true> bcustom = Uint32ToAC<ac_int<26, true> > (b);
  return ACToUint32<ac_int<26, true> > (acustom * bcustom);
}

TVM_DLL uint32_t SINT26Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<26, true> acustom = Uint32ToAC<ac_int<26, true> > (a);
  ac_int<26, true> bcustom = Uint32ToAC<ac_int<26, true> > (b);
  return ACToUint32<ac_int<26, true> > (acustom / bcustom);
}


}

// Generated code for dtype: UINT26

extern "C" {

TVM_DLL uint32_t MinUINT26() {
  // return minimum representable value
  ac_int<26, false> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<26, false> >(min);
}


TVM_DLL float  UINT26ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<26, false> > (in).to_double();
  return static_cast<ac_int<26, false> > (custom_datatype);
}

TVM_DLL uint32_t FloatToUINT26(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<26, false> > (in);
}

TVM_DLL uint32_t UINT26Max(uint32_t a, uint32_t b) {
  // max
  ac_int<26, false> acustom = Uint32ToAC<ac_int<26, false> > (a);
  ac_int<26, false> bcustom = Uint32ToAC<ac_int<26, false> > (b);
  return ACToUint32<ac_int<26, false> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t UINT26Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<26, false> acustom = Uint32ToAC<ac_int<26, false> > (a);
  ac_int<26, false> bcustom = Uint32ToAC<ac_int<26, false> > (b);
  return ACToUint32<ac_int<26, false> > (acustom + bcustom);
}

TVM_DLL uint32_t UINT26Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<26, false> acustom = Uint32ToAC<ac_int<26, false> > (a);
  ac_int<26, false> bcustom = Uint32ToAC<ac_int<26, false> > (b);
  return ACToUint32<ac_int<26, false> > (acustom - bcustom);
}

TVM_DLL uint32_t UINT26Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<26, false> acustom = Uint32ToAC<ac_int<26, false> > (a);
  ac_int<26, false> bcustom = Uint32ToAC<ac_int<26, false> > (b);
  return ACToUint32<ac_int<26, false> > (acustom * bcustom);
}

TVM_DLL uint32_t UINT26Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<26, false> acustom = Uint32ToAC<ac_int<26, false> > (a);
  ac_int<26, false> bcustom = Uint32ToAC<ac_int<26, false> > (b);
  return ACToUint32<ac_int<26, false> > (acustom / bcustom);
}


}

// Generated code for dtype: SINT27

extern "C" {

TVM_DLL uint32_t MinSINT27() {
  // return minimum representable value
  ac_int<27, true> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<27, true> >(min);
}


TVM_DLL float  SINT27ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<27, true> > (in).to_double();
  return static_cast<ac_int<27, true> > (custom_datatype);
}

TVM_DLL uint32_t FloatToSINT27(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<27, true> > (in);
}

TVM_DLL uint32_t SINT27Max(uint32_t a, uint32_t b) {
  // max
  ac_int<27, true> acustom = Uint32ToAC<ac_int<27, true> > (a);
  ac_int<27, true> bcustom = Uint32ToAC<ac_int<27, true> > (b);
  return ACToUint32<ac_int<27, true> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t SINT27Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<27, true> acustom = Uint32ToAC<ac_int<27, true> > (a);
  ac_int<27, true> bcustom = Uint32ToAC<ac_int<27, true> > (b);
  return ACToUint32<ac_int<27, true> > (acustom + bcustom);
}

TVM_DLL uint32_t SINT27Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<27, true> acustom = Uint32ToAC<ac_int<27, true> > (a);
  ac_int<27, true> bcustom = Uint32ToAC<ac_int<27, true> > (b);
  return ACToUint32<ac_int<27, true> > (acustom - bcustom);
}

TVM_DLL uint32_t SINT27Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<27, true> acustom = Uint32ToAC<ac_int<27, true> > (a);
  ac_int<27, true> bcustom = Uint32ToAC<ac_int<27, true> > (b);
  return ACToUint32<ac_int<27, true> > (acustom * bcustom);
}

TVM_DLL uint32_t SINT27Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<27, true> acustom = Uint32ToAC<ac_int<27, true> > (a);
  ac_int<27, true> bcustom = Uint32ToAC<ac_int<27, true> > (b);
  return ACToUint32<ac_int<27, true> > (acustom / bcustom);
}


}

// Generated code for dtype: UINT27

extern "C" {

TVM_DLL uint32_t MinUINT27() {
  // return minimum representable value
  ac_int<27, false> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<27, false> >(min);
}


TVM_DLL float  UINT27ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<27, false> > (in).to_double();
  return static_cast<ac_int<27, false> > (custom_datatype);
}

TVM_DLL uint32_t FloatToUINT27(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<27, false> > (in);
}

TVM_DLL uint32_t UINT27Max(uint32_t a, uint32_t b) {
  // max
  ac_int<27, false> acustom = Uint32ToAC<ac_int<27, false> > (a);
  ac_int<27, false> bcustom = Uint32ToAC<ac_int<27, false> > (b);
  return ACToUint32<ac_int<27, false> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t UINT27Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<27, false> acustom = Uint32ToAC<ac_int<27, false> > (a);
  ac_int<27, false> bcustom = Uint32ToAC<ac_int<27, false> > (b);
  return ACToUint32<ac_int<27, false> > (acustom + bcustom);
}

TVM_DLL uint32_t UINT27Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<27, false> acustom = Uint32ToAC<ac_int<27, false> > (a);
  ac_int<27, false> bcustom = Uint32ToAC<ac_int<27, false> > (b);
  return ACToUint32<ac_int<27, false> > (acustom - bcustom);
}

TVM_DLL uint32_t UINT27Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<27, false> acustom = Uint32ToAC<ac_int<27, false> > (a);
  ac_int<27, false> bcustom = Uint32ToAC<ac_int<27, false> > (b);
  return ACToUint32<ac_int<27, false> > (acustom * bcustom);
}

TVM_DLL uint32_t UINT27Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<27, false> acustom = Uint32ToAC<ac_int<27, false> > (a);
  ac_int<27, false> bcustom = Uint32ToAC<ac_int<27, false> > (b);
  return ACToUint32<ac_int<27, false> > (acustom / bcustom);
}


}

// Generated code for dtype: SINT28

extern "C" {

TVM_DLL uint32_t MinSINT28() {
  // return minimum representable value
  ac_int<28, true> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<28, true> >(min);
}


TVM_DLL float  SINT28ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<28, true> > (in).to_double();
  return static_cast<ac_int<28, true> > (custom_datatype);
}

TVM_DLL uint32_t FloatToSINT28(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<28, true> > (in);
}

TVM_DLL uint32_t SINT28Max(uint32_t a, uint32_t b) {
  // max
  ac_int<28, true> acustom = Uint32ToAC<ac_int<28, true> > (a);
  ac_int<28, true> bcustom = Uint32ToAC<ac_int<28, true> > (b);
  return ACToUint32<ac_int<28, true> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t SINT28Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<28, true> acustom = Uint32ToAC<ac_int<28, true> > (a);
  ac_int<28, true> bcustom = Uint32ToAC<ac_int<28, true> > (b);
  return ACToUint32<ac_int<28, true> > (acustom + bcustom);
}

TVM_DLL uint32_t SINT28Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<28, true> acustom = Uint32ToAC<ac_int<28, true> > (a);
  ac_int<28, true> bcustom = Uint32ToAC<ac_int<28, true> > (b);
  return ACToUint32<ac_int<28, true> > (acustom - bcustom);
}

TVM_DLL uint32_t SINT28Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<28, true> acustom = Uint32ToAC<ac_int<28, true> > (a);
  ac_int<28, true> bcustom = Uint32ToAC<ac_int<28, true> > (b);
  return ACToUint32<ac_int<28, true> > (acustom * bcustom);
}

TVM_DLL uint32_t SINT28Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<28, true> acustom = Uint32ToAC<ac_int<28, true> > (a);
  ac_int<28, true> bcustom = Uint32ToAC<ac_int<28, true> > (b);
  return ACToUint32<ac_int<28, true> > (acustom / bcustom);
}


}

// Generated code for dtype: UINT28

extern "C" {

TVM_DLL uint32_t MinUINT28() {
  // return minimum representable value
  ac_int<28, false> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<28, false> >(min);
}


TVM_DLL float  UINT28ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<28, false> > (in).to_double();
  return static_cast<ac_int<28, false> > (custom_datatype);
}

TVM_DLL uint32_t FloatToUINT28(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<28, false> > (in);
}

TVM_DLL uint32_t UINT28Max(uint32_t a, uint32_t b) {
  // max
  ac_int<28, false> acustom = Uint32ToAC<ac_int<28, false> > (a);
  ac_int<28, false> bcustom = Uint32ToAC<ac_int<28, false> > (b);
  return ACToUint32<ac_int<28, false> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t UINT28Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<28, false> acustom = Uint32ToAC<ac_int<28, false> > (a);
  ac_int<28, false> bcustom = Uint32ToAC<ac_int<28, false> > (b);
  return ACToUint32<ac_int<28, false> > (acustom + bcustom);
}

TVM_DLL uint32_t UINT28Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<28, false> acustom = Uint32ToAC<ac_int<28, false> > (a);
  ac_int<28, false> bcustom = Uint32ToAC<ac_int<28, false> > (b);
  return ACToUint32<ac_int<28, false> > (acustom - bcustom);
}

TVM_DLL uint32_t UINT28Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<28, false> acustom = Uint32ToAC<ac_int<28, false> > (a);
  ac_int<28, false> bcustom = Uint32ToAC<ac_int<28, false> > (b);
  return ACToUint32<ac_int<28, false> > (acustom * bcustom);
}

TVM_DLL uint32_t UINT28Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<28, false> acustom = Uint32ToAC<ac_int<28, false> > (a);
  ac_int<28, false> bcustom = Uint32ToAC<ac_int<28, false> > (b);
  return ACToUint32<ac_int<28, false> > (acustom / bcustom);
}


}

// Generated code for dtype: SINT29

extern "C" {

TVM_DLL uint32_t MinSINT29() {
  // return minimum representable value
  ac_int<29, true> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<29, true> >(min);
}


TVM_DLL float  SINT29ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<29, true> > (in).to_double();
  return static_cast<ac_int<29, true> > (custom_datatype);
}

TVM_DLL uint32_t FloatToSINT29(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<29, true> > (in);
}

TVM_DLL uint32_t SINT29Max(uint32_t a, uint32_t b) {
  // max
  ac_int<29, true> acustom = Uint32ToAC<ac_int<29, true> > (a);
  ac_int<29, true> bcustom = Uint32ToAC<ac_int<29, true> > (b);
  return ACToUint32<ac_int<29, true> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t SINT29Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<29, true> acustom = Uint32ToAC<ac_int<29, true> > (a);
  ac_int<29, true> bcustom = Uint32ToAC<ac_int<29, true> > (b);
  return ACToUint32<ac_int<29, true> > (acustom + bcustom);
}

TVM_DLL uint32_t SINT29Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<29, true> acustom = Uint32ToAC<ac_int<29, true> > (a);
  ac_int<29, true> bcustom = Uint32ToAC<ac_int<29, true> > (b);
  return ACToUint32<ac_int<29, true> > (acustom - bcustom);
}

TVM_DLL uint32_t SINT29Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<29, true> acustom = Uint32ToAC<ac_int<29, true> > (a);
  ac_int<29, true> bcustom = Uint32ToAC<ac_int<29, true> > (b);
  return ACToUint32<ac_int<29, true> > (acustom * bcustom);
}

TVM_DLL uint32_t SINT29Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<29, true> acustom = Uint32ToAC<ac_int<29, true> > (a);
  ac_int<29, true> bcustom = Uint32ToAC<ac_int<29, true> > (b);
  return ACToUint32<ac_int<29, true> > (acustom / bcustom);
}


}

// Generated code for dtype: UINT29

extern "C" {

TVM_DLL uint32_t MinUINT29() {
  // return minimum representable value
  ac_int<29, false> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<29, false> >(min);
}


TVM_DLL float  UINT29ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<29, false> > (in).to_double();
  return static_cast<ac_int<29, false> > (custom_datatype);
}

TVM_DLL uint32_t FloatToUINT29(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<29, false> > (in);
}

TVM_DLL uint32_t UINT29Max(uint32_t a, uint32_t b) {
  // max
  ac_int<29, false> acustom = Uint32ToAC<ac_int<29, false> > (a);
  ac_int<29, false> bcustom = Uint32ToAC<ac_int<29, false> > (b);
  return ACToUint32<ac_int<29, false> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t UINT29Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<29, false> acustom = Uint32ToAC<ac_int<29, false> > (a);
  ac_int<29, false> bcustom = Uint32ToAC<ac_int<29, false> > (b);
  return ACToUint32<ac_int<29, false> > (acustom + bcustom);
}

TVM_DLL uint32_t UINT29Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<29, false> acustom = Uint32ToAC<ac_int<29, false> > (a);
  ac_int<29, false> bcustom = Uint32ToAC<ac_int<29, false> > (b);
  return ACToUint32<ac_int<29, false> > (acustom - bcustom);
}

TVM_DLL uint32_t UINT29Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<29, false> acustom = Uint32ToAC<ac_int<29, false> > (a);
  ac_int<29, false> bcustom = Uint32ToAC<ac_int<29, false> > (b);
  return ACToUint32<ac_int<29, false> > (acustom * bcustom);
}

TVM_DLL uint32_t UINT29Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<29, false> acustom = Uint32ToAC<ac_int<29, false> > (a);
  ac_int<29, false> bcustom = Uint32ToAC<ac_int<29, false> > (b);
  return ACToUint32<ac_int<29, false> > (acustom / bcustom);
}


}

// Generated code for dtype: SINT30

extern "C" {

TVM_DLL uint32_t MinSINT30() {
  // return minimum representable value
  ac_int<30, true> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<30, true> >(min);
}


TVM_DLL float  SINT30ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<30, true> > (in).to_double();
  return static_cast<ac_int<30, true> > (custom_datatype);
}

TVM_DLL uint32_t FloatToSINT30(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<30, true> > (in);
}

TVM_DLL uint32_t SINT30Max(uint32_t a, uint32_t b) {
  // max
  ac_int<30, true> acustom = Uint32ToAC<ac_int<30, true> > (a);
  ac_int<30, true> bcustom = Uint32ToAC<ac_int<30, true> > (b);
  return ACToUint32<ac_int<30, true> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t SINT30Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<30, true> acustom = Uint32ToAC<ac_int<30, true> > (a);
  ac_int<30, true> bcustom = Uint32ToAC<ac_int<30, true> > (b);
  return ACToUint32<ac_int<30, true> > (acustom + bcustom);
}

TVM_DLL uint32_t SINT30Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<30, true> acustom = Uint32ToAC<ac_int<30, true> > (a);
  ac_int<30, true> bcustom = Uint32ToAC<ac_int<30, true> > (b);
  return ACToUint32<ac_int<30, true> > (acustom - bcustom);
}

TVM_DLL uint32_t SINT30Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<30, true> acustom = Uint32ToAC<ac_int<30, true> > (a);
  ac_int<30, true> bcustom = Uint32ToAC<ac_int<30, true> > (b);
  return ACToUint32<ac_int<30, true> > (acustom * bcustom);
}

TVM_DLL uint32_t SINT30Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<30, true> acustom = Uint32ToAC<ac_int<30, true> > (a);
  ac_int<30, true> bcustom = Uint32ToAC<ac_int<30, true> > (b);
  return ACToUint32<ac_int<30, true> > (acustom / bcustom);
}


}

// Generated code for dtype: UINT30

extern "C" {

TVM_DLL uint32_t MinUINT30() {
  // return minimum representable value
  ac_int<30, false> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<30, false> >(min);
}


TVM_DLL float  UINT30ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<30, false> > (in).to_double();
  return static_cast<ac_int<30, false> > (custom_datatype);
}

TVM_DLL uint32_t FloatToUINT30(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<30, false> > (in);
}

TVM_DLL uint32_t UINT30Max(uint32_t a, uint32_t b) {
  // max
  ac_int<30, false> acustom = Uint32ToAC<ac_int<30, false> > (a);
  ac_int<30, false> bcustom = Uint32ToAC<ac_int<30, false> > (b);
  return ACToUint32<ac_int<30, false> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t UINT30Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<30, false> acustom = Uint32ToAC<ac_int<30, false> > (a);
  ac_int<30, false> bcustom = Uint32ToAC<ac_int<30, false> > (b);
  return ACToUint32<ac_int<30, false> > (acustom + bcustom);
}

TVM_DLL uint32_t UINT30Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<30, false> acustom = Uint32ToAC<ac_int<30, false> > (a);
  ac_int<30, false> bcustom = Uint32ToAC<ac_int<30, false> > (b);
  return ACToUint32<ac_int<30, false> > (acustom - bcustom);
}

TVM_DLL uint32_t UINT30Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<30, false> acustom = Uint32ToAC<ac_int<30, false> > (a);
  ac_int<30, false> bcustom = Uint32ToAC<ac_int<30, false> > (b);
  return ACToUint32<ac_int<30, false> > (acustom * bcustom);
}

TVM_DLL uint32_t UINT30Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<30, false> acustom = Uint32ToAC<ac_int<30, false> > (a);
  ac_int<30, false> bcustom = Uint32ToAC<ac_int<30, false> > (b);
  return ACToUint32<ac_int<30, false> > (acustom / bcustom);
}


}

// Generated code for dtype: SINT31

extern "C" {

TVM_DLL uint32_t MinSINT31() {
  // return minimum representable value
  ac_int<31, true> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<31, true> >(min);
}


TVM_DLL float  SINT31ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<31, true> > (in).to_double();
  return static_cast<ac_int<31, true> > (custom_datatype);
}

TVM_DLL uint32_t FloatToSINT31(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<31, true> > (in);
}

TVM_DLL uint32_t SINT31Max(uint32_t a, uint32_t b) {
  // max
  ac_int<31, true> acustom = Uint32ToAC<ac_int<31, true> > (a);
  ac_int<31, true> bcustom = Uint32ToAC<ac_int<31, true> > (b);
  return ACToUint32<ac_int<31, true> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t SINT31Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<31, true> acustom = Uint32ToAC<ac_int<31, true> > (a);
  ac_int<31, true> bcustom = Uint32ToAC<ac_int<31, true> > (b);
  return ACToUint32<ac_int<31, true> > (acustom + bcustom);
}

TVM_DLL uint32_t SINT31Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<31, true> acustom = Uint32ToAC<ac_int<31, true> > (a);
  ac_int<31, true> bcustom = Uint32ToAC<ac_int<31, true> > (b);
  return ACToUint32<ac_int<31, true> > (acustom - bcustom);
}

TVM_DLL uint32_t SINT31Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<31, true> acustom = Uint32ToAC<ac_int<31, true> > (a);
  ac_int<31, true> bcustom = Uint32ToAC<ac_int<31, true> > (b);
  return ACToUint32<ac_int<31, true> > (acustom * bcustom);
}

TVM_DLL uint32_t SINT31Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<31, true> acustom = Uint32ToAC<ac_int<31, true> > (a);
  ac_int<31, true> bcustom = Uint32ToAC<ac_int<31, true> > (b);
  return ACToUint32<ac_int<31, true> > (acustom / bcustom);
}


}

// Generated code for dtype: UINT31

extern "C" {

TVM_DLL uint32_t MinUINT31() {
  // return minimum representable value
  ac_int<31, false> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<31, false> >(min);
}


TVM_DLL float  UINT31ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<31, false> > (in).to_double();
  return static_cast<ac_int<31, false> > (custom_datatype);
}

TVM_DLL uint32_t FloatToUINT31(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<31, false> > (in);
}

TVM_DLL uint32_t UINT31Max(uint32_t a, uint32_t b) {
  // max
  ac_int<31, false> acustom = Uint32ToAC<ac_int<31, false> > (a);
  ac_int<31, false> bcustom = Uint32ToAC<ac_int<31, false> > (b);
  return ACToUint32<ac_int<31, false> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t UINT31Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<31, false> acustom = Uint32ToAC<ac_int<31, false> > (a);
  ac_int<31, false> bcustom = Uint32ToAC<ac_int<31, false> > (b);
  return ACToUint32<ac_int<31, false> > (acustom + bcustom);
}

TVM_DLL uint32_t UINT31Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<31, false> acustom = Uint32ToAC<ac_int<31, false> > (a);
  ac_int<31, false> bcustom = Uint32ToAC<ac_int<31, false> > (b);
  return ACToUint32<ac_int<31, false> > (acustom - bcustom);
}

TVM_DLL uint32_t UINT31Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<31, false> acustom = Uint32ToAC<ac_int<31, false> > (a);
  ac_int<31, false> bcustom = Uint32ToAC<ac_int<31, false> > (b);
  return ACToUint32<ac_int<31, false> > (acustom * bcustom);
}

TVM_DLL uint32_t UINT31Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<31, false> acustom = Uint32ToAC<ac_int<31, false> > (a);
  ac_int<31, false> bcustom = Uint32ToAC<ac_int<31, false> > (b);
  return ACToUint32<ac_int<31, false> > (acustom / bcustom);
}


}

// Generated code for dtype: SINT32

extern "C" {

TVM_DLL uint32_t MinSINT32() {
  // return minimum representable value
  ac_int<32, true> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<32, true> >(min);
}


TVM_DLL float  SINT32ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<32, true> > (in).to_double();
  return static_cast<ac_int<32, true> > (custom_datatype);
}

TVM_DLL uint32_t FloatToSINT32(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<32, true> > (in);
}

TVM_DLL uint32_t SINT32Max(uint32_t a, uint32_t b) {
  // max
  ac_int<32, true> acustom = Uint32ToAC<ac_int<32, true> > (a);
  ac_int<32, true> bcustom = Uint32ToAC<ac_int<32, true> > (b);
  return ACToUint32<ac_int<32, true> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t SINT32Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<32, true> acustom = Uint32ToAC<ac_int<32, true> > (a);
  ac_int<32, true> bcustom = Uint32ToAC<ac_int<32, true> > (b);
  return ACToUint32<ac_int<32, true> > (acustom + bcustom);
}

TVM_DLL uint32_t SINT32Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<32, true> acustom = Uint32ToAC<ac_int<32, true> > (a);
  ac_int<32, true> bcustom = Uint32ToAC<ac_int<32, true> > (b);
  return ACToUint32<ac_int<32, true> > (acustom - bcustom);
}

TVM_DLL uint32_t SINT32Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<32, true> acustom = Uint32ToAC<ac_int<32, true> > (a);
  ac_int<32, true> bcustom = Uint32ToAC<ac_int<32, true> > (b);
  return ACToUint32<ac_int<32, true> > (acustom * bcustom);
}

TVM_DLL uint32_t SINT32Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<32, true> acustom = Uint32ToAC<ac_int<32, true> > (a);
  ac_int<32, true> bcustom = Uint32ToAC<ac_int<32, true> > (b);
  return ACToUint32<ac_int<32, true> > (acustom / bcustom);
}


}

// Generated code for dtype: UINT32

extern "C" {

TVM_DLL uint32_t MinUINT32() {
  // return minimum representable value
  ac_int<32, false> min;
  min.set_val<AC_VAL_MIN>();
  return ACToUint32<ac_int<32, false> >(min);
}


TVM_DLL float  UINT32ToFloat(uint32_t in) {
  float custom_datatype = Uint32ToAC<ac_int<32, false> > (in).to_double();
  return static_cast<ac_int<32, false> > (custom_datatype);
}

TVM_DLL uint32_t FloatToUINT32(float in) {
  // cast from float to custom datatype
  return ACToUint32<ac_int<32, false> > (in);
}

TVM_DLL uint32_t UINT32Max(uint32_t a, uint32_t b) {
  // max
  ac_int<32, false> acustom = Uint32ToAC<ac_int<32, false> > (a);
  ac_int<32, false> bcustom = Uint32ToAC<ac_int<32, false> > (b);
  return ACToUint32<ac_int<32, false> > (acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t UINT32Add(uint32_t a, uint32_t b) {
  // add operation
  ac_int<32, false> acustom = Uint32ToAC<ac_int<32, false> > (a);
  ac_int<32, false> bcustom = Uint32ToAC<ac_int<32, false> > (b);
  return ACToUint32<ac_int<32, false> > (acustom + bcustom);
}

TVM_DLL uint32_t UINT32Sub(uint32_t a, uint32_t b) {
  // subtract
  ac_int<32, false> acustom = Uint32ToAC<ac_int<32, false> > (a);
  ac_int<32, false> bcustom = Uint32ToAC<ac_int<32, false> > (b);
  return ACToUint32<ac_int<32, false> > (acustom - bcustom);
}

TVM_DLL uint32_t UINT32Mul(uint32_t a, uint32_t b) {
  // multiply
  ac_int<32, false> acustom = Uint32ToAC<ac_int<32, false> > (a);
  ac_int<32, false> bcustom = Uint32ToAC<ac_int<32, false> > (b);
  return ACToUint32<ac_int<32, false> > (acustom * bcustom);
}

TVM_DLL uint32_t UINT32Div(uint32_t a, uint32_t b) {
  // divide
  ac_int<32, false> acustom = Uint32ToAC<ac_int<32, false> > (a);
  ac_int<32, false> bcustom = Uint32ToAC<ac_int<32, false> > (b);
  return ACToUint32<ac_int<32, false> > (acustom / bcustom);
}


}
