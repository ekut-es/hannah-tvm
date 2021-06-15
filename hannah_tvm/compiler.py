import copy
import glob
import os

import tvm
import tvm.micro


def get_compiler_options(config):
    opts = tvm.micro.default_options(
        os.path.join(tvm.micro.get_standalone_crt_dir(), "template", "host")
    )

    for option in opts.values():
        if not "libs" in option:
            option["libs"] = []
        for flag, values in option.items():
            values.extend(config[flag])

    from pprint import pprint

    pprint(opts)

    return opts


class Compiler_Ext(tvm.micro.DefaultCompiler):
    def __init__(self, target, prefix=None, opts=None):
        super().__init__(target)

        self.prefix = prefix
        self.opts = opts

    def _autodetect_toolchain_prefix(self, target):
        if self.prefix is not None:
            return self.prefix

        super()._autodetect_toolchain_prefix(target)

    def _defaults_from_target(self, target):
        if self.opts is not None:
            return copy.deepcopy(self.opts)

        opts = super(Compiler_Ext, self)._defaults_from_target(target)
        return opts

    # TODO: move upstream
    def binary(self, output, objects, options=None, link_main=True, main_options=None):
        assert self.target is not None, (
            "must specify target= to constructor, or compile sources which specify the target "
            "first"
        )

        args = [self._autodetect_toolchain_prefix(self.target) + "g++"]
        args.extend(self._defaults_from_target(self.target))
        if options is not None:
            args.extend(options.get("ldflags", []))

            for include_dir in options.get("include_dirs", []):
                args.extend(["-I", include_dir])

        output_filename = os.path.basename(output)
        output_abspath = os.path.join(output, output_filename)
        args.extend(["-g", "-o", output_abspath])

        if link_main:
            host_main_srcs = glob.glob(
                os.path.join(
                    tvm.micro.get_standalone_crt_dir(), "template", "host", "*.cc"
                )
            )
            if main_options:
                main_lib = self.library(
                    os.path.join(output, "host"), host_main_srcs, main_options
                )
                for lib_name in main_lib.library_files:
                    args.append(main_lib.abspath(lib_name))
            else:
                args.extend(host_main_srcs)

        for obj in objects:
            for lib_name in obj.library_files:
                args.append(obj.abspath(lib_name))

        extra_libs = options.get("libs", [])
        args.extend(extra_libs)

        tvm.micro.compiler.run_cmd(args)
        return tvm.micro.MicroBinary(output, output_filename, [])
