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
import copy
import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch.fx
import tvm
import tvm.relay as relay
from hannah.models.factory import pooling, qat, qconfig
from matplotlib import use
from torch.ao.nn import quantized as nnq
from torch.ao.nn.intrinsic import quantized as nni

logger = logging.getLogger("__name__")


def legalize_var_name(s):
    s = s.replace(".", "_")

    return s


@dataclass
class TensorMetadata:
    shape: List[int]
    bits: int

    # Quantization info
    scale: Optional[float] = None
    dtype: Optional[str] = None
    zero_point: Optional[float] = None

    @property
    def relay_dtype(self):
        return f"{self.dtype}{self.bits}"


def parse_dtype(dtype: str):
    if dtype.startswith("int"):
        type = "int"
        bits = int(dtype[3:])
    elif dtype.startswith("uint"):
        type = "uint"
        bits = int(dtype[4:])
    elif dtype.startswith("float"):
        type = "float"
        bits = int(dtype[5:])
    else:
        raise Exception(f"Unhandled dtype: {dtype}")
    return type, bits


def get_integer_range(dtype) -> Optional[Tuple[int, int]]:
    type, bits = parse_dtype(dtype)
    if type == "int":
        return -(2 ** (bits - 1)), --(2 ** (bits - 1)) - 1
    elif type == "uint":
        return 0, 2 ** (bits - 1)
    else:
        return None


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
            if const.data.dtype != self.dtype_map[const.data.dtype]:
                return const.astype(self.dtype_map[const.data.dtype])
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
                dtype = self.dtype_map[dtype]
            else:
                dtype = var_type.dtype

            # Generate new variable.
            new_param = tvm.relay.expr.var(
                param.name_hint, shape=var_type.shape, dtype=dtype
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

        out_dtype = None
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

        if out_dtype is not None:
            range = get_integer_range(out_dtype)
            if range is not None:
                new_call = tvm.relay.clip(new_call, range[0], range[1])

        return new_call


class QuantizationTracer(torch.fx.Tracer):
    LEAF_MODULES = [
        qat.Conv1d,
        qat.Conv2d,
        qat.ConvBn1d,
        qat.ConvBn2d,
        qat.ConvBnReLU1d,
        qat.ConvBnReLU2d,
        qat.ConvReLU1d,
        qat.ConvReLU2d,
        qat.Linear,
        qat.LinearReLU,
        qconfig.STEQuantize,
    ]

    def is_leaf_module(self, module, module_qualified_name):
        for leaf_cls in self.LEAF_MODULES:
            if isinstance(module, leaf_cls):
                return True

        return super().is_leaf_module(module, module_qualified_name)


class RelayConverter(torch.fx.Interpreter):
    def __init__(
        self,
        graph_module,
        input_dtype="int8",
        input_scale=1 / (2**7),
        accumulator_dtype="int20",
    ):
        super().__init__(graph_module)
        self.accumulator_dtype = accumulator_dtype
        self.input_dtype = input_dtype
        self.input_scale = input_scale

        if relay is None:
            raise Exception(
                "TVM does not seem to be installed, please make sure that 'import tvm.relay works'"
            )

        self.tvm_mod = None
        self.modules = {}
        for name, module in graph_module.named_modules():
            self.modules[name] = module

        self.outputs = {}
        self.tensor_info: Dict["str", TensorMetadata] = {}
        self.func_args = []
        self.returns = []
        self.params = {}

        self.module_map = {
            qat.Conv1d: self._handle_qat_conv,
            qat.Conv2d: self._handle_qat_conv,
            qat.ConvBn1d: self._handle_qat_conv,
            qat.ConvBn2d: self._handle_qat_conv,
            qat.ConvBnReLU1d: self._handle_qat_conv,
            qat.ConvBnReLU2d: self._handle_qat_conv,
            qat.ConvReLU1d: self._handle_qat_conv,
            qat.ConvReLU2d: self._handle_qat_conv,
            qat.Linear: self._handle_qat_linear,
            qat.LinearReLU: self._handle_qat_linear,
            qconfig.STEQuantize: self._handle_requantize,
            torch.nn.ReLU: self._handle_relu,
            torch.nn.Dropout: self._handle_identity,
            torch.nn.Identity: self._handle_identity,
            torch.nn.Flatten: self._handle_flatten,
            nni.ConvReLU2d: self._handle_nni_conv,
        }

    def _gen_requantize(
        self,
        input,
        input_scale,
        input_dtype,
        output_scale,
        output_dtype,
        use_rescale=False,
        axis=-1,
        rounding="UPWARD",
    ):
        assert input_dtype.startswith("int")
        assert output_dtype.startswith("int")
        if output_dtype == input_dtype and output_scale == input_scale:
            return input

        input_bits = int(input_dtype[3:])
        output_bits = int(output_dtype[3:])

        output = input
        if use_rescale:
            output = tvm.relay.qnn.op.requantize(
                output,
                tvm.relay.const(input_scale, dtype="float32"),
                tvm.relay.const(0, dtype="int32"),
                tvm.relay.const(output_scale, dtype="float32"),
                tvm.relay.const(0, dtype="int32"),
                axis=axis,
                rounding=rounding,
                out_dtype=output_dtype,
            )
        else:
            rescale = input_scale / output_scale
            rescale_shift = int(math.log2(rescale))
            accumulator_dtype = (
                input_dtype if input_bits > output_bits else output_dtype
            )

            if output_bits > input_bits:
                output = relay.cast(output, output_dtype)
            if rescale != 1.0:
                if 2**rescale_shift == rescale:
                    if rescale_shift > 0:
                        output = tvm.relay.left_shift(
                            output,
                            tvm.relay.cast(
                                tvm.relay.const(rescale_shift), dtype=accumulator_dtype
                            ),
                        )
                    else:
                        output = tvm.relay.right_shift(
                            output,
                            tvm.relay.cast(
                                tvm.relay.const(abs(rescale_shift)),
                                dtype=accumulator_dtype,
                            ),
                        )
                else:
                    output = tvm.relay.multiply(
                        output,
                        tvm.relay.cast(
                            tvm.relay.const(int(rescale)), dtype=accumulator_dtype
                        ),
                    )
            if input_bits != output_bits:
                output = relay.cast(output, output_dtype)
        return output

    def _handle_nni_conv(self, node, module, result):
        inputs = list(node.all_input_nodes)
        data = self.outputs[inputs[0].name]
        weight = module.weight()
        weight_int = weight.int_repr().detach().numpy()

        output_scale = module.scale

        if result.dtype == torch.qint8:
            output_dtype = "int8"
        elif result.dtype == torch.qint32:
            output_dtype = "int32"
        elif result.dtype == torch.quint8:
            output_dtype = "uint8"
        elif result.dtype == torch.quint32:
            output_dtype = "uint32"

        output_zero_point = module.zero_point
        output_shape = result.shape
        if module.bias() is not None:
            raise Exception("Bias is not supported")

        return tvm.relay.qnn.op.conv2d(
            data, weight, output_scale, output_zero_point, output_dtype, output_shape
        )

    def _handle_flatten(self, node, module, result):
        inputs = list(node.all_input_nodes)
        assert len(inputs) == 1
        data = self.outputs[inputs[0].name]
        flatten = tvm.relay.nn.batch_flatten(data)
        self.outputs[node.name] = flatten
        output_metadata = copy.deepcopy(self.tensor_info[inputs[0].name])
        output_metadata.shape = result.shape
        self.tensor_info[node.name] = output_metadata

    def _handle_identity(self, node, module, result):
        inputs = list(node.all_input_nodes)
        assert len(inputs) == 1
        data = self.outputs[inputs[0].name]
        self.outputs[node.name] = data
        self.tensor_info[node.name] = self.tensor_info[inputs[0].name]
        return None

    def _handle_qat_linear(self, node, module, result):
        weight = module.weight
        bias = module.bias

        if hasattr(module, "bn"):
            weight, bias = torch.nn.utils.fuse_conv_bn_weights(
                module.weight,
                module.bias,
                module.bn.running_mean,
                module.bn.running_var,
                module.bn.eps,
                module.bn.weight,
                module.bn.bias,
            )

        inputs = list(node.all_input_nodes)
        data = self.outputs[inputs[0].name]
        input_info = self.tensor_info[inputs[0].name]
        input_bits = input_info.bits
        input_dtype = input_info.relay_dtype
        input_scale = input_info.scale

        quant_weight = module.weight_fake_quant.quantize(weight)
        quant_bias = module.bias_fake_quant.quantize(bias) if bias is not None else None
        weight_dtype = f"int{module.weight_fake_quant.bits}"
        weight_scale = module.weight_fake_quant.quantization_function.scale

        weight_name = legalize_var_name(f"{node.name}.weight")
        weight = tvm.relay.Var(
            weight_name, tvm.relay.TensorType(quant_weight.shape, dtype=weight_dtype)
        )
        self.params[weight_name] = tvm.nd.array(
            (quant_weight).detach().numpy().astype("byte")
        )
        if bias is not None:
            bias_dtype = f"int{module.bias_fake_quant.bits}"
            bias_scale = module.bias_fake_quant.quantization_function.scale
            bias_name = legalize_var_name(f"{node.name}.bias")
            bias = tvm.relay.Var(
                bias_name, tvm.relay.TensorType(quant_bias.shape, dtype=bias_dtype)
            )
            self.params[bias_name] = tvm.nd.array(
                (quant_bias).detach().numpy().astype("byte")
            )

        inputs = list(node.all_input_nodes)
        data = self.outputs[inputs[0].name]

        linear_out = tvm.relay.nn.dense(
            data, weight, out_dtype=self.accumulator_dtype
        )  # FIXME use proper out dtype

        accumulator_scale = weight_scale * input_scale

        if bias is not None:
            if bias_scale >= accumulator_scale:
                bias = self._gen_requantize(
                    bias,
                    bias_scale,
                    bias_dtype,
                    accumulator_scale,
                    self.accumulator_dtype,
                    use_rescale=True,
                    axis=0,
                )
            elif bias_scale < accumulator_scale:
                linear_out = self._gen_requantize(
                    linear_out,
                    accumulator_scale,
                    self.accumulator_dtype,
                    bias_scale,
                    self.accumulator_dtype,
                    use_rescale=True,
                    axis=0,
                )
                bias = relay.cast(bias, self.accumulator_dtype)
                accumulator_scale = bias_scale

            linear_out = tvm.relay.nn.bias_add(linear_out, bias)

        if isinstance(module, qat.LinearReLU):
            linear_out = tvm.relay.nn.relu(linear_out)

        if hasattr(module.activation_post_process, "bits") and module.out_quant:
            output_dtype = f"int{module.activation_post_process.bits}"
            output_scale = module.activation_post_process.quantization_function.scale
        else:
            output_dtype = self.accumulator_dtype
            output_scale = accumulator_scale

        # Calculate shift factors
        linear_out = self._gen_requantize(
            linear_out,
            accumulator_scale,
            self.accumulator_dtype,
            output_scale,
            output_dtype,
            use_rescale=True,
        )

        self.outputs[node.name] = linear_out
        dtype, bits = parse_dtype(output_dtype)
        self.tensor_info[node.name] = TensorMetadata(
            shape=result.shape, dtype=dtype, bits=bits, scale=output_scale, zero_point=0
        )

    def _handle_qat_conv(self, node, module, result):
        weight = module.weight
        bias = module.bias

        if hasattr(module, "bn"):
            weight, bias = torch.nn.utils.fuse_conv_bn_weights(
                module.weight,
                module.bias,
                module.bn.running_mean,
                module.bn.running_var,
                module.bn.eps,
                module.bn.weight,
                module.bn.bias,
            )

        padding = tuple(module.padding)
        stride = tuple(module.stride)
        dilation = tuple(module.dilation)
        groups = module.groups
        out_channels = module.out_channels

        quant_weight = module.weight_fake_quant.quantize(weight)
        quant_bias = module.bias_fake_quant.quantize(bias) if bias is not None else None

        weight_dtype = f"int{module.weight_fake_quant.bits}"
        weight_scale = module.weight_fake_quant.quantization_function.scale

        inputs = list(node.all_input_nodes)
        data = self.outputs[inputs[0].name]
        input_info = self.tensor_info[inputs[0].name]
        input_bits = input_info.bits
        input_dtype = input_info.relay_dtype
        input_scale = input_info.scale

        weight_name = legalize_var_name(f"{node.name}.weight")
        weight = tvm.relay.Var(
            weight_name, tvm.relay.TensorType(quant_weight.shape, dtype=weight_dtype)
        )
        self.params[weight_name] = tvm.nd.array(
            (quant_weight).detach().numpy().astype("byte")
        )
        if bias is not None:
            bias_dtype = f"int{module.bias_fake_quant.bits}"
            bias_scale = module.bias_fake_quant.quantization_function.scale
            bias_name = legalize_var_name(f"{node.name}.bias")
            bias = tvm.relay.Var(
                bias_name, tvm.relay.TensorType(quant_bias.shape, dtype=bias_dtype)
            )
            self.params[bias_name] = tvm.nd.array(
                (quant_bias).detach().numpy().astype("byte")
            )

        if quant_weight.dim() == 3:
            conv_out = tvm.relay.nn.conv1d(
                data,
                weight,
                strides=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                channels=out_channels,
                kernel_size=quant_weight.size(2),
                data_layout="NCW",
                kernel_layout="OIW",
                out_dtype=self.accumulator_dtype,
            )  # FIXME use proper out dtype
        elif quant_weight.dim() == 4:
            conv_out = tvm.relay.nn.conv2d(
                data,
                weight,
                strides=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                channels=out_channels,
                kernel_size=(quant_weight.size(2), quant_weight.size(3)),
                data_layout="NCHW",
                kernel_layout="OIHW",
                out_dtype=self.accumulator_dtype,
            )
        else:
            raise Exception(
                f"Quantized weights of dimension {quant_weight.dim()} are not supported"
            )

        # print("conv_out:", conv_out)
        accumulator_scale = weight_scale * input_scale

        if bias is not None:
            if bias_scale >= accumulator_scale:
                bias = self._gen_requantize(
                    bias,
                    bias_scale,
                    bias_dtype,
                    accumulator_scale,
                    self.accumulator_dtype,
                    use_rescale=True,
                    axis=0,
                )
            elif bias_scale < accumulator_scale:
                conv_out = self._gen_requantize(
                    conv_out,
                    accumulator_scale,
                    self.accumulator_dtype,
                    bias_scale,
                    self.accumulator_dtype,
                    use_rescale=True,
                    axis=0,
                )
                bias = relay.cast(bias, self.accumulator_dtype)
                accumulator_scale = bias_scale

                # print("conv_out", conv_out)
                # print("bias", bias)

            conv_out = tvm.relay.nn.bias_add(conv_out, bias)

        if (
            isinstance(module, qat.ConvBnReLU1d)
            or isinstance(module, qat.ConvBnReLU2d)
            or isinstance(module, qat.ConvReLU1d)
            or isinstance(module, qat.ConvReLU2d)
        ):
            conv_out = tvm.relay.nn.relu(conv_out)

        if (
            hasattr(module.activation_post_process, "bits")
            and getattr(module, "out_quant", True) is True
        ):
            output_dtype = f"int{module.activation_post_process.bits}"
            output_scale = module.activation_post_process.quantization_function.scale
        else:
            output_dtype = self.accumulator_dtype
            output_scale = accumulator_scale

        # Calculate shift factors
        conv_out = self._gen_requantize(
            conv_out,
            accumulator_scale,
            self.accumulator_dtype,
            output_scale,
            output_dtype,
            use_rescale=True,
        )

        self.outputs[node.name] = conv_out
        dtype, bits = parse_dtype(output_dtype)
        self.tensor_info[node.name] = TensorMetadata(
            shape=result.shape, dtype=dtype, bits=bits, scale=output_scale, zero_point=0
        )

    def _handle_relu(self, node, module, result):
        inputs = list(node.all_input_nodes)
        assert len(inputs) == 1
        data = self.outputs[inputs[0].name]
        relu = tvm.relay.nn.relu(data)
        self.outputs[node.name] = relu
        output_metadata = copy.deepcopy(self.tensor_info[inputs[0].name])
        self.tensor_info[node.name] = output_metadata

    def _handle_requantize(self, node, module, result):
        inputs = list(node.all_input_nodes)
        assert len(inputs) == 1
        input = self.outputs[inputs[0].name]
        input_info = self.tensor_info[inputs[0].name]

        input_scale = input_info.scale
        input_dtype = input_info.relay_dtype

        if isinstance(module, qat.Identity):
            requantize_mod = module.activation_post_process
        else:
            requantize_mod = module

        output_scale = requantize_mod.quantization_function.scale
        output_dtype = requantize_mod.dtype
        output_bits = requantize_mod.bits

        output_metadata = copy.deepcopy(self.tensor_info[inputs[0].name])
        output_metadata.bits = output_bits
        output_metadata.dtype = output_dtype
        output_metadata.scale = output_scale

        requantize = self._gen_requantize(
            input,
            input_scale,
            input_dtype,
            output_metadata.scale,
            output_metadata.relay_dtype,
            use_rescale=True,
            rounding="UPWARD",
        )
        self.outputs[node.name] = requantize
        self.tensor_info[node.name] = output_metadata

    def _handle_getattr(self, node, result):
        target = self.fetch_attr(node.target)

        if torch.is_tensor(target):
            var = relay.var(
                node.target, relay.TensorType(result.shape, dtype=self.input_dtype)
            )
            self.outputs[node.target] = var
            dtype, bits = parse_dtype(self.input_dtype)
            self.tensor_info[node.target] = TensorMetadata(
                shape=result.shape, dtype=dtype, bits=bits, scale=self.input_scale
            )
            self.func_args.append(var)
        else:
            raise Exception(f"Unhandled target: {target}")

    def _handle_module(self, node, result):
        module = self.modules[node.target]
        if type(module) in self.module_map:
            self.module_map[type(module)](node, module, result)
        else:
            # breakpoint()
            raise Exception(f"Support for module: {module} is not supported")

    def _handle_placeholder(self, node, result):
        var = relay.var(
            node.name, relay.TensorType(result.shape, dtype=self.input_dtype)
        )
        self.outputs[node.name] = var
        dtype, bits = parse_dtype(self.input_dtype)
        self.tensor_info[node.name] = TensorMetadata(
            shape=result.shape, dtype=dtype, bits=bits, scale=self.input_scale
        )
        self.func_args.append(var)

    def _handle_output(self, node, result):
        inputs = list(node.all_input_nodes)

        for input in inputs:
            self.returns.append(self.outputs[input.name])

    def _handle_function(self, node, result):
        target = node.target

        if target.__name__ == "add":
            inputs = list(node.all_input_nodes)
            assert len(inputs) == 2
            lhs = self.outputs[inputs[0].name]
            rhs = self.outputs[inputs[1].name]
            lhs_data = self.tensor_info[inputs[0].name]
            rhs_data = self.tensor_info[inputs[1].name]
            assert lhs_data.dtype == rhs_data.dtype
            output_dtype = lhs_data.dtype
            output_bits = max(lhs_data.bits, rhs_data.bits)
            output_scale = min(lhs_data.scale, rhs_data.scale)

            lhs = self._gen_requantize(
                lhs,
                lhs_data.scale,
                f"{lhs_data.dtype}{lhs_data.bits}",
                output_scale,
                f"{output_dtype}{output_bits}",
                axis=1,
                use_rescale=True,
            )
            rhs = self._gen_requantize(
                rhs,
                rhs_data.scale,
                f"{rhs_data.dtype}{rhs_data.bits}",
                output_scale,
                f"{output_dtype}{output_bits}",
                axis=1,
                use_rescale=True,
            )

            add = tvm.relay.add(lhs, rhs)
            self.outputs[node.name] = add
            self.tensor_info[node.name] = TensorMetadata(
                shape=result.shape,
                bits=output_bits,
                scale=output_scale,
                zero_point=0,
                dtype=output_dtype,
            )
        elif target.__name__ == "sum":
            inputs = list(node.all_input_nodes)
            assert len(inputs) == 1
            data = self.outputs[inputs[0].name]
            sum = tvm.relay.sum(data, axis=2, keepdims=True)
            self.outputs[node.name] = sum
            output_info = copy.deepcopy(self.tensor_info[inputs[0].name])
            self.tensor_info[node.name] = output_info
        elif target.__name__ == "truediv":
            inputs = list(node.all_input_nodes)
            assert len(inputs) == 1
            input_info = self.tensor_info[inputs[0].name]
            divider = node.args[1]

            data = self.outputs[inputs[0].name]
            div = data / tvm.relay.cast(
                tvm.relay.cast(tvm.relay.const(divider), input_info.relay_dtype),
                input_info.relay_dtype,
            )
            print(input_info)
            print(div)
            self.outputs[node.name] = div
            output_info = copy.deepcopy(self.tensor_info[inputs[0].name])
            self.tensor_info[node.name] = output_info

        elif target == torch.quantize_per_tensor:
            inputs = list(node.all_input_nodes)
            self.outputs[node.name] = self.outputs[node.args[0].name]
            output_info = copy.deepcopy(self.tensor_info[inputs[0].name])
            self.tensor_info[node.name] = output_info

        else:
            raise Exception(f"Unandled function {target}")

    def run_node(self, node):
        result = super().run_node(node)

        if node.op == "call_module":
            result_metadata = self._handle_module(node, result)
        elif node.op == "call_function":
            result_metadata = self._handle_function(node, result)
        elif node.op == "output":
            result_metadata = self._handle_output(node, result)
        elif node.op == "placeholder":
            result_metadata = self._handle_placeholder(node, result)
        elif node.op == "get_attr":
            result_metadata = self._handle_getattr(node, result)
        else:
            raise Exception(f"Node {node} with op {node.op} is not supported")

        return result

    def propagate(self, *args):
        return super().run(*args)

    def run(self, input):
        tvm_mod = tvm.IRModule()

        super().run(input)

        ret = (
            self.returns[0] if len(self.returns) == 1 else tvm.relay.Tuple(self.returns)
        )
        free_vars = relay.analysis.free_vars(ret)
        function = relay.Function(free_vars, ret)
        tvm_mod["main"] = function

        tvm_mod = tvm.relay.transform.InferType()(tvm_mod)

        return tvm_mod, self.params
