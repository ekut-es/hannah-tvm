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
import tvm.relay


@tvm.relay.transform.function_pass(opt_level=0)
class LegalizeQuantizedTypes(tvm.relay.expr_functor.ExprMutator):
    def __init__(self):
        super().__init__()

        self.dtype_map = {}
        for i in range(1, 9):
            self.dtype_map[f"int{i}"] = "int8"
        for i in range(9, 17):
            self.dtype_map[f"int{i}"] = "int16"
        for i in range(17, 33):
            self.dtype_map[f"int{i}"] = "int32"
        for i in range(33, 65):
            self.dtype_map[f"int{i}"] = "int64"

        for i in range(1, 9):
            self.dtype_map[f"uint{i}"] = "uint8"
        for i in range(9, 17):
            self.dtype_map[f"uint{i}"] = "uint16"
        for i in range(17, 33):
            self.dtype_map[f"uint{i}"] = "uint32"
        for i in range(33, 65):
            self.dtype_map[f"uint{i}"] = "uint64"

    def transform_function(self, func, mod, ctx):
        return self.visit(func)

    def visit_constant(self, const):
        if const.data.dtype in self.dtype_map:
            new_dtype = self.dtype_map[const.data.dtype]
            if const.data.dtype != new_dtype:
                return const.astype(new_dtype)
        return const

    def visit_function(self, fn):
        new_params = []
        binds = {}
        for param in fn.params:
            # Get the parameter's type annotation.
            var_type = param.type_annotation
            if isinstance(var_type, tvm.ir.TensorType):
                dtype = var_type.dtype

            # See if we want to replace dtype.
            if dtype in self.dtype_map:
                new_dtype = self.dtype_map[dtype]
            else:
                new_dtype = var_type.dtype

            # Generate new variable.
            new_param = tvm.relay.expr.var(
                param.name_hint, shape=var_type.shape, dtype=new_dtype
            )

            new_params.append(new_param)
            binds[param] = new_param

        new_body = self.visit(fn.body)
        # Rewrite the body to use new parameters.
        new_body = tvm.relay.expr.bind(new_body, binds)

        # Construct the updated function and return.
        return tvm.relay.Function(
            new_params,
            new_body,
            # You could change the return type, if you use None it will re-infer.
            None,
            type_params=fn.type_params,
            attrs=fn.attrs,
        )

    def visit_call(self, call):
        new_args = [self.visit(arg) for arg in call.args]
        new_attrs = call.attrs
        new_fn = self.visit(call.op)
        new_call = tvm.relay.Call(
            new_fn, new_args, new_attrs, call.type_args, call.span
        )

        if call.op.name == "nn.conv1d":
            out_dtype = call.attrs.out_dtype
            new_attrs = dict(call.attrs)
            new_attrs["out_dtype"] = self.dtype_map[out_dtype]
            new_call = tvm.relay.nn.conv1d(*new_args, **new_attrs)
        elif call.op.name == "nn.conv2d":
            out_dtype = call.attrs.out_dtype
            new_attrs = dict(call.attrs)
            new_attrs["out_dtype"] = self.dtype_map[out_dtype]
            new_call = tvm.relay.nn.conv2d(*new_args, **new_attrs)
        elif call.op.name == "nn.conv3d":
            out_dtype = call.attrs.out_dtype
            new_attrs = dict(call.attrs)
            new_attrs["out_dtype"] = self.dtype_map[out_dtype]
            new_call = tvm.relay.nn.conv3d(*new_args, **new_attrs)
        elif call.op.name == "nn.dense":
            out_dtype = call.attrs.out_dtype
            new_attrs = dict(call.attrs)
            new_attrs["out_dtype"] = self.dtype_map[out_dtype]
            new_call = tvm.relay.nn.dense(*new_args, **new_attrs)
        elif call.op.name == "qnn.requantize":
            out_dtype = call.attrs.out_dtype
            new_attrs = dict(call.attrs)
            new_attrs["out_dtype"] = self.dtype_map[out_dtype]
            new_call = tvm.relay.qnn.op.requantize(*new_args, **new_attrs)
        elif call.op.name == "cast":
            out_dtype = call.attrs.dtype
            new_attrs = dict(call.attrs)
            new_attrs["dtype"] = self.dtype_map[out_dtype]
            new_call = tvm.relay.cast(*new_args, **new_attrs)

        return new_call
