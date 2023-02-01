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
"""
.. _tutorial-tensor-expr-get-started:

Working with Operators Using Tensor Expression
==============================================
**Author**: `Tianqi Chen <https://tqchen.github.io>`_

In this tutorial we will turn our attention to how TVM works with Tensor
Expression (TE) to define tensor computations and apply loop optimizations. TE
describes tensor computations in a pure functional language (that is each
expression has no side effects). When viewed in context of the TVM as a whole,
Relay describes a computation as a set of operators, and each of these
operators can be represented as a TE expression where each TE expression takes
input tensors and produces an output tensor.

This is an introductory tutorial to the Tensor Expression language in TVM. TVM
uses a domain specific tensor expression for efficient kernel construction. We
will demonstrate the basic workflow with two examples of using the tensor expression
language. The first example introduces TE and scheduling with vector
addition. The second expands on these concepts with a step-by-step optimization
of a matrix multiplication with TE. This matrix multiplication example will
serve as the comparative basis for future tutorials covering more advanced
features of TVM.
"""

# sphinx_gallery_start_ignore
from tvm import testing

testing.utils.install_request_hook(depth=3)
# sphinx_gallery_end_ignore

################################################################################
# Example 1: Writing and Scheduling Vector Addition in TE for CPU
# ---------------------------------------------------------------
#
# Let's look at an example in Python in which we will implement a TE for
# vector addition, followed by a schedule targeted towards a CPU.
# We begin by initializing a TVM environment.

import numpy as np
import tvm
import tvm.testing
from tvm import te

################################################################################
# You will get better performance if you can identify the CPU you are targeting
# and specify it. If you're using LLVM, you can get this information from the
# command ``llc --version`` to get the CPU type, and you can check
# ``/proc/cpuinfo`` for additional extensions that your processor might
# support. For example, you can use ``llvm -mcpu=skylake-avx512`` for CPUs with
# AVX-512 instructions.

tgt = tvm.target.Target(target="llvm -mcpu=skylake", host="llvm -mcpu=skylake")

################################################################################
# Describing the Vector Computation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We describe a vector addition computation. TVM adopts tensor semantics, with
# each intermediate result represented as a multi-dimensional array. The user
# needs to describe the computation rule that generates the tensors. We first
# define a symbolic variable ``n`` to represent the shape. We then define two
# placeholder Tensors, ``A`` and ``B``, with given shape ``(n,)``. We then
# describe the result tensor ``C``, with a ``compute`` operation. The
# ``compute`` defines a computation, with the output conforming to the
# specified tensor shape and the computation to be performed at each position
# in the tensor defined by the lambda function. Note that while ``n`` is a
# variable, it defines a consistent shape between the ``A``, ``B`` and ``C``
# tensors. Remember, no actual computation happens during this phase, as we
# are only declaring how the computation should be done.

n = te.var("n")
A = te.placeholder((n,), name="A")
B = te.placeholder((n,), name="B")
C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")

################################################################################
# .. admonition:: Lambda Functions
#
#   The second argument to the ``te.compute`` method is the function that
#   performs the computation. In this example, we're using an anonymous function,
#   also known as a ``lambda`` function, to define the computation, in this case
#   addition on the ``i``\th element of ``A`` and ``B``.

################################################################################
# Create a Default Schedule for the Computation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# While the above lines describe the computation rule, we can compute ``C`` in
# many different ways to fit different devices. For a tensor with multiple
# axes, you can choose which axis to iterate over first, or computations can be
# split across different threads. TVM requires that the user to provide a
# schedule, which is a description of how the computation should be performed.
# Scheduling operations within TE can change loop orders, split computations
# across different threads, and group blocks of data together, amongst other
# operations. An important concept behind schedules is that they only describe
# how the computation is performed, so different schedules for the same TE will
# produce the same result.
#
# TVM allows you to create a naive schedule that will compute ``C`` in by
# iterating in row major order.
#
# .. code-block:: c
#
#   for (int i = 0; i < n; ++i) {
#     C[i] = A[i] + B[i];
#   }

s = te.create_schedule(C.op)

######################################################################
# Compile and Evaluate the Default Schedule
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# With the TE expression and a schedule, we can produce runnable code for our
# target language and architecture, in this case LLVM and a CPU. We provide
# TVM with the schedule, a list of the TE expressions that are in the schedule,
# the target and host, and the name of the function we are producing. The result
# of the output is a type-erased function that can be called directly from Python.
#
# In the following line, we use ``tvm.build`` to create a function. The build
# function takes the schedule, the desired signature of the function (including
# the inputs and outputs) as well as target language we want to compile to.

print(tvm.lower(s, [A, B, C], simple_mode=True))


fadd = tvm.build(s, [A, B, C], tgt, name="myadd")

################################################################################
# Let's run the function, and compare the output to the same computation in
# numpy. The compiled TVM function exposes a concise C API that can be invoked
# from any language. We begin by creating a device, which is a device (CPU in this
# example) that TVM can compile the schedule to. In this case the device is an
# LLVM CPU target. We can then initialize the tensors in our device and
# perform the custom addition operation. To verify that the computation is
# correct, we can compare the result of the output of the c tensor to the same
# computation performed by numpy.

dev = tvm.device(tgt.kind.name, 0)

n = 1024
a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
fadd(a, b, c)
tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())

################################################################################
# To get a comparison of how fast this version is compared to numpy, create a
# helper function to run a profile of the TVM generated code.
import timeit

np_repeat = 100
np_running_time = timeit.timeit(
    setup="import numpy\n"
    "n = 32768*1000\n"
    'dtype = "float32"\n'
    "a = numpy.random.rand(n, 1).astype(dtype)\n"
    "b = numpy.random.rand(n, 1).astype(dtype)\n",
    stmt="answer = a + b",
    number=np_repeat,
)
print("Numpy running time: %f" % (np_running_time / np_repeat))


def evaluate_addition(func, target, optimization, log):
    dev = tvm.device(target.kind.name, 0)
    n = 32768 * 100
    a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
    b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
    c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)

    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    mean_time = evaluator(a, b, c).mean
    print("%s: %f" % (optimization, mean_time))

    log.append((optimization, mean_time))


log = [("numpy", np_running_time / np_repeat)]
evaluate_addition(fadd, tgt, "naive", log=log)

################################################################################
# Updating the Schedule to Use Parallelism
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Now that we've illustrated the fundamentals of TE, let's go deeper into what
# schedules do, and how they can be used to optimize tensor expressions for
# different architectures. A schedule is a series of steps that are applied to
# an expression to transform it in a number of different ways. When a schedule
# is applied to an expression in TE, the inputs and outputs remain the same,
# but when compiled the implementation of the expression can change. This
# tensor addition, in the default schedule, is run serially but is easy to
# parallelize across all of the processor threads. We can apply the parallel
# schedule operation to our computation.

s[C].parallel(C.op.axis[0])

################################################################################
# The ``tvm.lower`` command will generate the Intermediate Representation (IR)
# of the TE, with the corresponding schedule. By lowering the expression as we
# apply different schedule operations, we can see the effect of scheduling on
# the ordering of the computation. We use the flag ``simple_mode=True`` to
# return a readable C-style statement.

print(tvm.lower(s, [A, B, C], simple_mode=True))

################################################################################
# It's now possible for TVM to run these blocks on independent threads. Let's
# compile and run this new schedule with the parallel operation applied:

fadd_parallel = tvm.build(s, [A, B, C], tgt, name="myadd_parallel")
fadd_parallel(a, b, c)

tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())

evaluate_addition(fadd_parallel, tgt, "parallel", log=log)

################################################################################
# Updating the Schedule to Use Vectorization
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Modern CPUs also have the ability to perform SIMD operations on floating
# point values, and we can apply another schedule to our computation expression
# to take advantage of this. Accomplishing this requires multiple steps: first
# we have to split the schedule into inner and outer loops using the split
# scheduling primitive. The inner loops can use vectorization to use SIMD
# instructions using the vectorize scheduling primitive, then the outer loops
# can be parallelized using the parallel scheduling primitive. Choose the split
# factor to be the number of threads on your CPU.

# Recreate the schedule, since we modified it with the parallel operation in
# the previous example
# n = te.var("n")
n = 32768 * 100
A = te.placeholder((n,), name="A")
B = te.placeholder((n,), name="B")
C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")

s = te.create_schedule(C.op)

# This factor should be chosen to match the number of threads appropriate for
# your CPU. This will vary depending on architecture, but a good rule is
# setting this factor to equal the number of available CPU cores.
factor = 8

outer, inner = s[C].split(C.op.axis[0], factor=factor)
s[C].parallel(outer)
s[C].vectorize(inner)

fadd_vector = tvm.build(s, [A, B, C], tgt, name="myadd_parallel")

print(fadd_vector.get_source())

evaluate_addition(fadd_vector, tgt, "vector", log=log)

print(tvm.lower(s, [A, B, C], simple_mode=True))

################################################################################
# Comparing the Different Schedules
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We can now compare the different schedules

baseline = log[0][1]
print(
    "%s\t%s\t%s" % ("Operator".rjust(20), "Timing".rjust(20), "Performance".rjust(20))
)
for result in log:
    print(
        "%s\t%s\t%s"
        % (
            result[0].rjust(20),
            str(result[1]).rjust(20),
            str(result[1] / baseline).rjust(20),
        )
    )


################################################################################
# .. admonition:: Code Specialization
#
#   As you may have noticed, the declarations of ``A``, ``B`` and ``C`` all
#   take the same shape argument, ``n``. TVM will take advantage of this to
#   pass only a single shape argument to the kernel, as you will find in the
#   printed device code. This is one form of specialization.
#
#   On the host side, TVM will automatically generate check code that checks
#   the constraints in the parameters. So if you pass arrays with different
#   shapes into fadd, an error will be raised.
#
#   We can do more specializations. For example, we can write :code:`n =
#   tvm.runtime.convert(1024)` instead of :code:`n = te.var("n")`, in the
#   computation declaration. The generated function will only take vectors with
#   length 1024.

################################################################################
# We've defined, scheduled, and compiled a vector addition operator, which we
# were then able to execute on the TVM runtime. We can save the operator as a
# library, which we can then load later using the TVM runtime.
