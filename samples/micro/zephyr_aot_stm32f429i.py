# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
microTVM with TFLite Models
===========================
**Author**: `Tom Gall <https://github.com/tom-gall>`_

This tutorial is an introduction to working with microTVM and a TFLite
model with Relay.
"""

######################################################################
# .. note::
#     If you want to run this tutorial on the microTVM Reference VM, download the Jupyter
#     notebook using the link at the bottom of this page and save it into the TVM directory. Then:
#
#     #. Login to the reference VM with a modified ``vagrant ssh`` command:
#
#         ``$ vagrant ssh -- -L8888:localhost:8888``
#
#     #. Install jupyter:  ``pip install jupyterlab``
#     #. ``cd`` to the TVM directory.
#     #. Install tflite: poetry install -E importer-tflite
#     #. Launch Jupyter Notebook: ``jupyter notebook``
#     #. Copy the localhost URL displayed, and paste it into your browser.
#     #. Navigate to saved Jupyter Notebook (``.ipynb`` file).
#
#
# Setup
# -----
#
# Install TFLite
# ^^^^^^^^^^^^^^
#
# To get started, TFLite package needs to be installed as prerequisite. You can do this in two ways:
#
# 1. Install tflite with ``pip``
#
#     .. code-block:: bash
#
#       pip install tflite=2.1.0 --user
#
# 2. Generate the TFLite package yourself. The steps are the following:
#
#     Get the flatc compiler.
#     Please refer to https://github.com/google/flatbuffers for details
#     and make sure it is properly installed.
#
#     .. code-block:: bash
#
#       flatc --version
#
#     Get the TFLite schema.
#
#     .. code-block:: bash
#
#       wget https://raw.githubusercontent.com/tensorflow/tensorflow/r1.13/tensorflow/lite/schema/schema.fbs
#
#     Generate TFLite package.
#
#     .. code-block:: bash
#
#       flatc --python schema.fbs
#
#     Add the current folder (which contains generated tflite module) to PYTHONPATH.
#
#     .. code-block:: bash
#
#       export PYTHONPATH=${PYTHONPATH:+$PYTHONPATH:}$(pwd)
#
# To validate that the TFLite package was installed successfully, ``python -c "import tflite"``
#
# Install Zephyr (physical hardware only)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# When running this tutorial with a host simulation (the default), you can use the host ``gcc`` to
# build a firmware image that simulates the device. When compiling to run on physical hardware, you
# need to install a *toolchain* plus some target-specific dependencies. microTVM allows you to
# supply any compiler and runtime that can launch the TVM RPC server, but to get started, this
# tutorial relies on the Zephyr RTOS to provide these pieces.
#
# You can install Zephyr by following the
# `Installation Instructions <https://docs.zephyrproject.org/latest/getting_started/index.html>`_.
#
# Aside: Recreating your own Pre-Trained TFLite model
#  The tutorial downloads a pretrained TFLite model. When working with microcontrollers
#  you need to be mindful these are highly resource constrained devices as such standard
#  models like MobileNet may not fit into their modest memory.
#
#  For this tutorial, we'll make use of one of the TF Micro example models.
#
#  If you wish to replicate the training steps see:
#  https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/micro/examples/hello_world/train
#
#    .. note::
#
#      If you accidentally download the example pretrained model from:
#
#      ``wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/micro/hello_world_2020_04_13.zip``
#
#      this will fail due to an unimplemented opcode (114)
#
# Load and prepare the Pre-Trained Model
# --------------------------------------
#
# Load the pretrained TFLite model from a file in your current
# directory into a buffer

import logging
import os
import pathlib
import sys

import numpy as np
import tvm
import tvm.micro as micro
from tvm import relay
from tvm.contrib import graph_executor, utils
from tvm.contrib.download import download_testdata


def _create_header_file(tensor_name, npy_data, output_path):
    """
    This method generates a header file containing the data contained in the numpy array provided.
    It is used to capture the tensor data (for both inputs and expected outputs).
    """
    file_path = pathlib.Path(f"{output_path}/" + tensor_name).resolve()
    # create header file
    raw_path = file_path.with_suffix(".h").resolve()
    with open(raw_path, "w") as header_file:
        header_file.write("#include <stddef.h>\n")
        header_file.write("#include <stdint.h>\n")
        header_file.write("#include <dlpack/dlpack.h>\n")
        header_file.write(f"const size_t {tensor_name}_len = {npy_data.size};\n")

        if npy_data.dtype == "int8":
            header_file.write(f"int8_t {tensor_name}[] =")
        elif npy_data.dtype == "int32":
            header_file.write(f"int32_t {tensor_name}[] = ")
        elif npy_data.dtype == "uint8":
            header_file.write(f"uint8_t {tensor_name}[] = ")
        elif npy_data.dtype == "float32":
            header_file.write(f"float {tensor_name}[] = ")

        header_file.write("{")
        for i in np.ndindex(npy_data.shape):
            header_file.write(f"{npy_data[i]}, ")
        header_file.write("};\n\n")


model_url = "https://people.linaro.org/~tom.gall/sine_model.tflite"
model_file = "sine_model.tflite"
model_path = download_testdata(model_url, model_file, module="data")

tflite_model_buf = open(model_path, "rb").read()

######################################################################
# Using the buffer, transform into a tflite model python object
try:
    import tflite

    tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
except AttributeError:
    import tflite.Model

    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

######################################################################
# Print out the version of the model
version = tflite_model.Version()
print("Model Version: " + str(version))

######################################################################
# Parse the python model object to convert it into a relay module
# and weights.
# It is important to note that the input tensor name must match what
# is contained in the model.
#
# If you are unsure what that might be, this can be discovered by using
# the ``visualize.py`` script within the Tensorflow project.
# See `How do I inspect a .tflite file? <https://www.tensorflow.org/lite/guide/faq>`_

input_tensor = "dense_4_input"
input_shape = (1,)
input_dtype = "float32"

mod, params = relay.frontend.from_tflite(
    tflite_model,
    shape_dict={input_tensor: input_shape},
    dtype_dict={input_tensor: input_dtype},
)

######################################################################
# Defining the target
# -------------------
#
# Now we create a build config for relay, turning off two options and then calling relay.build which
# will result in a C source file for the selected TARGET. When running on a simulated target of the
# same architecture as the host (where this Python script is executed) choose "host" below for the
# TARGET and a proper board/VM to run it (Zephyr will create the right QEMU VM based on BOARD. In
# the example below the x86 arch is selected and a x86 VM is picked up accordingly:
#
# TARGET = tvm.target.target.micro("host")
# BOARD = "qemu_x86"
#
# Compiling for physical hardware
#  When running on physical hardware, choose a TARGET and a BOARD that describe the hardware. The
#  STM32F746 Nucleo target and board is chosen in the example below. Another option would be to
#  choose the STM32F746 Discovery board instead. Since that board has the same MCU as the Nucleo
#  board but a couple of wirings and configs differ, it's necessary to select the "stm32f746g_disco"
#  board to generated the right firmware image.
#
TARGET = tvm.target.target.micro(
    "stm32f4xx", options=["-link-params=1", "--executor=aot"]
)
BOARD = "stm32f429i_disc1"  # "nucleo_f746zg" # or "stm32f746g_disco#"
#
#  For some boards, Zephyr runs them emulated by default, using QEMU. For example, below is the
#  TARGET and BOARD used to build a microTVM firmware for the mps2-an521 board. Since that board
#  runs emulated by default on Zephyr the suffix "-qemu" is added to the board name to inform
#  microTVM that the QEMU transporter must be used to communicate with the board. If the board name
#  already has the prefix "qemu_", like "qemu_x86", then it's not necessary to add that suffix.
#
#  TARGET = tvm.target.target.micro("mps2_an521")
#  BOARD = "mps2_an521-qemu"

######################################################################
# Now, compile the model for the target:

with tvm.transform.PassContext(
    opt_level=3,
    config={"tir.disable_vectorize": True},
    disabled_pass=["FuseOps", "AlterOpLayout"],
):
    lowered = relay.build(mod, target_host=TARGET, target=TARGET, params=params)


# Compiling for a host simulated device
# -------------------------------------
#
# First, compile a static microTVM runtime for the targeted device. In this case, the host simulated
# device is used.
# compiler = tvm.micro.DefaultCompiler(target=TARGET)
# opts = tvm.micro.default_options(
#    os.path.join(tvm.micro.get_standalone_crt_dir(), "template", "host")
# )

# Compiling for physical hardware (or an emulated board, like the mps_an521)
# --------------------------------------------------------------------------
#  For physical hardware, comment out the previous section selecting TARGET and BOARD and use this
#  compiler definition instead of the one above.
#
import subprocess

from tvm.micro.contrib import zephyr

repo_root = subprocess.check_output(
    ["git", "rev-parse", "--show-toplevel"], encoding="utf-8"
).strip()
project_dir = os.path.join(
    repo_root, "external", "tvm", "apps", "microtvm", "zephyr", "aot_demo"
)

sample = np.array([0.5], dtype="float32")
output_shape = (1,)

model_files_path = os.path.join(project_dir, "include")
_create_header_file(("input_data"), sample, model_files_path)
_create_header_file(
    "output_data", np.zeros(shape=output_shape, dtype="float32"), model_files_path
)

compiler = zephyr.ZephyrCompiler(
    project_dir=project_dir,
    board=BOARD,
    zephyr_toolchain_variant="zephyr",
    env_vars={"ZEPHYR_RUNTIME": "ZEPHYR-AOT"},
)

logging.basicConfig(level="DEBUG")
workspace = tvm.micro.Workspace(debug=True)
opts = tvm.micro.default_options(f"{project_dir}/crt")
opts["bin_opts"]["include_dirs"].append(os.path.join(project_dir, "include"))
opts["lib_opts"]["include_dirs"].append(os.path.join(project_dir, "include"))
micro_binary = tvm.micro.build_static_runtime(
    workspace,
    compiler,
    lowered.lib,
    opts,
    executor="aot",
    extra_libs=[tvm.micro.get_standalone_crt_lib("memory")],
)

flasher = compiler.flasher()
transport = flasher.flash(micro_binary)
transport.open()
transport.write(b"start\n", timeout_sec=5)

print("Results:")
while True:
    res = transport.read(1, timeout_sec=10)
    for c in res:
        sys.stdout.write(chr(c))
