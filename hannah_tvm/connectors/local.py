from tvm import autotvm, auto_scheduler
from .core import BoardConnector, TaskConnector, BuildArtifactHandle


@dataclass
class LocalBuildArtifactHandle(BuildArtifactHandle):
    lib: Any


class LocalTaskConnector(TaskConnector):
    def __init__(self, board_config, tracker_port):
        self._board_config = board_config
        self._target = None
        self.auto_scheduler_ctx = None

    def setup(self):
        self._target = tvm.target.Target(
            self._board_config.target, host=self._board_config.target_host
        )

    def target(self):
        return self._target

    def runner(self, tuner=None):
        if tuner == "autotvm":
            runner = autotvm.LocalRunner(enable_cpu_cache_flush=True)
        elif tuner == "auto_scheduler":
            if self.auto_scheduler_ctx is None:
                self.auto_scheduler_ctx = auto_scheduler.LocalRPCMeasureContext()
            runner = self.auto_scheduler_ctx.runner
        return runner

    def builder(self, tuner=None):
        if tuner == "autotvm":
            builder = autotvm.LocalBuilder(build_func="default")
        elif tuner == "auto_scheduler":
            builder = "local"
        return builder

    def upload(self, lib):
        # Export library
        tmp = tvm.contrib.utils.tempdir()
        filename = "net.tar"
        lib.export_library(tmp.relpath(filename))

        # Upload module to device
        logger.info("Upload...")
        remote = auto_scheduler.utils.request_remote(
            self._board_config.name, "localhost", self._tracker_port, timeout=10000
        )

        remote.upload(tmp.relpath(filename))

        rlib = remote.load_module(filename)
        logger.info("Upload finished")
        return AutomateBuildArtifactHandle(remote, rlib, lib)

    def measure(self, remote_handle, inputs):
        dev = self._remote_dev(remote_handle.remote)
        rlib = remote_handle.rlib
        lib = remote_handle.lib
        module = tvm.contrib.graph_executor.GraphModule(rlib["default"](dev))
        logger.info("Set inputs")
        for name, val in inputs.items():
            data_tvm = tvm.nd.array(val)
            module.set_input(name, data_tvm)

        # Evaluate on Graph Executor
        logger.info("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", dev, repeat=10, min_repeat_ms=500)
        prof_res = np.array(ftimer().results) * 1e6  # convert to microsecond

        return prof_res

    def profile(self, remote_handle, inputs):
        dev = self._remote_dev(remote_handle.remote)
        # Use debug Executor to get per operator runtime
        rlib = remote_handle.rlib
        lib = remote_handle.lib
        debug_module = tvm.contrib.debugger.debug_executor.GraphModuleDebug(
            rlib["debug_create"]("default", dev), [dev], lib.get_graph_json(), None
        )
        for name, val in inputs.items():
            data_tvm = tvm.nd.array(val)
            debug_module.set_input(name, data_tvm)
        debug_profile = debug_module.profile()

        return debug_profile

    def teardown(self):
        if self.auto_scheduler_ctx is not None:
            self.auto_scheduler_ctx = None

    def _remote_dev(self, remote):
        target = self.target()
        if str(target.kind) == "cuda":
            dev = remote.cuda()
        else:
            dev = remote.cpu()
        return dev


class AutomateBoardConnector(BoardConnector):
    def __init__(self, board_config):
        self._board_config = board_config

        self._tracker_port: Union[int, None] = None
        self._tracker_conn = None
        self._tracker = None

        self._server_process: Union[AutomateServer, None] = None

    def setup(self):
        pass

    def task_connector(self):
        pass

    def is_alive(self):
        return True

    def reset(self):
        pass

    def teardown(self):
        pass

    def boards_available(self):
        return 1

    def _start_tracker(self) -> int:
        pass

    def _start_server(self):
        pass
