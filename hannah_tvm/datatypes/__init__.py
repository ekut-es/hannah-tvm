#
# Copyright (c) 2022 University of Tübingen.
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
import ctypes
import os

import numpy as np
import tvm
from tvm import relay

from .config import BITS, OPS, SIGNED

dir_path = os.path.dirname(os.path.realpath(__file__))
ctypes.CDLL(os.path.join(dir_path, "libac_types.so"), ctypes.RTLD_GLOBAL)


typeid = 150
for bits in BITS:
    for sign in SIGNED:
        typename = "S" if sign else "U"
        typename += "INT"
        typename += str(bits)

        tvm.target.datatype.register(typename, typeid)

        tvm.target.datatype.register_op(
            tvm.target.datatype.create_lower_func(
                {
                    (32, 32): f"FloatTo{typename}",  # cast from float32 to SINT6
                }
            ),
            "Cast",
            "llvm",
            "float",
            typename,
        )

        tvm.target.datatype.register_op(
            tvm.target.datatype.create_lower_func({(32, 32): f"{typename}ToFloat"}),
            "Cast",
            "llvm",
            typename,
            "float",
        )

        tvm.target.datatype.register_min_func(
            tvm.target.datatype.create_min_lower_func({32: f"Min{typename}"}, typename),
            typename,
        )

        tvm.target.datatype.register_op(
            tvm.target.datatype.lower_ite,
            "Call",
            "llvm",
            typename,
            intrinsic_name="tir.if_then_else",
        )

        tvm.target.datatype.register_op(
            tvm.target.datatype.lower_call_pure_extern,
            "Call",
            "llvm",
            typename,
            intrinsic_name="tir.call_pure_extern",
        )
        for op in OPS:
            tvm.target.datatype.register_op(
                tvm.target.datatype.create_lower_func({32: f"{typename}{op}"}),
                op,
                "llvm",
                typename,
            )

        typeid += 1
