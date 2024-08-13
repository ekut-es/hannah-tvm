#
# Copyright (c) 2024 hannah-tvm contributors.
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
import numpy as np
import pytest
import tvm
from tvm import relay

from hannah_tvm import datatypes


@pytest.mark.parametrize(
    "dtype",
    [
        "SINT2",
        "SINT3",
        "SINT4",
        "SINT5",
        "SINT6",
        "SINT7",
        "SINT8",
        "SINT9",
        "SINT10",
        "SINT12",
        "SINT14",
        "SINT16",
        "UINT2",
        "UINT3",
        "UINT4",
        "UINT5",
        "UINT6",
        "UINT7",
        "UINT8",
        "UINT9",
        "UINT10",
        "UINT12",
        "UINT14",
        "UINT16",
    ],
)
def test_simple(dtype: str):
    try:
        with tvm.transform.PassContext(config={"tir.disable_vectorize": True}):
            x = relay.var("x", shape=(3,), dtype="float32")
            y = relay.var("y", shape=(3,), dtype="float32")
            x_myfloat = relay.cast(x, dtype=f"custom[{dtype}]32")
            y_myfloat = relay.cast(y, dtype=f"custom[{dtype}]32")
            z_myfloat = relay.max(x_myfloat + y_myfloat * x_myfloat, axis=0)
            z = relay.cast(z_myfloat, dtype="float32")

            program = relay.Function([x, y], z)
            module = tvm.IRModule.from_expr(program)
            module = relay.transform.InferType()(module)

            x_input = np.random.rand(3).astype("float32") * 10
            y_input = np.random.rand(3).astype("float32") * 10
            print("x: {}".format(x_input))
            print("y: {}".format(y_input))

            with tvm.transform.PassContext(
                config={"tir.disable_vectorize": True, "tir.disable_assert": True}
            ):
                z_output_myfloat = relay.create_executor(
                    "graph", mod=module
                ).evaluate()(x_input, y_input)
            print("z: {}".format(z_output_myfloat))
    except tvm.TVMError as e:
        # Print last line of error
        print(str(e).split("\n")[-1])
        raise e


if __name__ == "__main__":
    test_simple()
