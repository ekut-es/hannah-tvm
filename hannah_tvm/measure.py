import atexit
import logging
import multiprocessing
import time


import tvm.auto_scheduler as auto_scheduler
import tvm.rpc

try:
    from automate.config import AutomateConfig
    from automate.context import AutomateContext

    automate_available = True
except ModuleNotFoundError:
    automate_available = False

logger = logging.getLogger("hannah_tvm.measure")

_automate_context = None
_automate_measure_contexts = []


@atexit.register
def cleanup():
    logging.info("Cleaning up measurement contexts")
    for rpc in _automate_measure_contexts:
        context.finish()


class ServerProcess(multiprocessing.Process):
    def __init__(self, board_name, rundir, tracker_port):
        super().__init__()
        self.board_name = board_name
        self.rundir = rundir
        self.tracker_port = tracker_port
        self._pconn, self._cconn = multiprocessing.Pipe()

    def run(self):
        print("Running server")
        rundir = self.rundir
        tracker_port = self.tracker_port
        name = self.board_name
        automate_config = AutomateConfig()
        automate_context = AutomateContext(automate_config)
        board_connection = automate_context.board(name).connect()

        with board_connection.forward_remote(9000, tracker_port):
            with board_connection.forward_local(9090, 9090):
                promise = board_connection.run(
                    f"python3.6 -m tvm.exec.rpc_server --key {name} --host localhost --port=9090 --tracker=localhost:9000",
                    env={"PYTHONPATH": str(rundir)},
                    shell=True,
                    warn=True,
                    pty=True,
                    asynchronous=True,
                )

                while (
                    not promise.runner.process_is_finished
                    and not promise.runner.has_dead_threads
                ):
                    if self._cconn.poll():
                        msg = self._cconn.recv()
                        if msg == "exit":
                            promise.runner.send_interrupt(
                                Exception("Could not send interrupt")
                            )
                    time.sleep(0.5)

                result = promise.join()
                if not result:
                    board_connection.close()
                    self._cconn.send(str(result))

    def running(self):
        if self._pconn.poll():
            error = self._pconn.recv()
            raise Exception(str(error))

        try:
            conn = tvm.rpc.connect_tracker("localhost", self.tracker_port)
            queue_info = conn.summary()["queue_info"]
            if self.board_name in queue_info:
                if queue_info[self.board_name]["free"] > 0:
                    return True
            conn.close()

        except Exception as e:
            print(str(e))
            pass
        return False

    def finish(self):
        self._pconn.send("exit")
        while True:
            try:
                if not self.running:
                    return
                time.sleep(0.5)
            except Exception:
                pass


class AutomateRPCMeasureContext:
    """A context wrapper for running RPCRunner locally.
    This will launch a local RPC Tracker and local RPC Server.

    Parameters
    ----------
    priority : int = 1
        The priority of this run request, larger is more prior.
    n_parallel : int = 1
        The number of tasks run in parallel.
    timeout : int = 10
        The timeout limit (in second) for each run.
        This is used in a wrapper of the multiprocessing.Process.join().
    number : int = 3
        The number of times to run the generated code for taking average.
        We call these runs as one `repeat` of measurement.
    repeat : int = 1
        The number of times to repeat the measurement.
        In total, the generated code will be run (1 + number x repeat) times,
        where the first "1" is warm up and will be discarded.
        The returned result contains `repeat` costs,
        each of which is an average of `number` costs.
    min_repeat_ms : int = 0
        The minimum duration of one `repeat` in milliseconds.
        By default, one `repeat` contains `number` runs. If this parameter is set,
        the parameters `number` will be dynamically adjusted to meet the
        minimum duration requirement of one `repeat`.
        i.e., When the run time of one `repeat` falls below this time, the `number` parameter
        will be automatically increased.
    cooldown_interval : float = 0.0
        The cool down interval between two measurements.
    enable_cpu_cache_flush: bool = False
        Whether to flush cache on CPU between repeated measurements.
        Flushing cache can make the measured latency of one operator closer to
        its actual latency during end-to-end inference.
        To make this option effective, the argument `number` should also be set to 1.
        This is only has effect on CPU task.
    """

    def __init__(
        self,
        board_config,
        priority=1,
        n_parallel=1,
        timeout=10,
        number=3,
        repeat=1,
        min_repeat_ms=0,
        cooldown_interval=0.0,
        enable_cpu_cache_flush=False,
    ):
        if not automate_available:
            raise Exception(
                "automate remote execution framework is not installed please install with poetry install -E automate"
            )

        host = "0.0.0.0"
        self.board_config = board_config
        device_key = board_config.name

        global _automate_context
        if _automate_context is None:
            from automate.config import AutomateConfig
            from automate.context import AutomateContext

            automate_config = AutomateConfig()
            _automate_context = AutomateContext(automate_config)

        self.board = None
        self.board_connection = None
        self.tracker = None
        self.server_process = None

        self.board = _automate_context.board(board_config.name)
        self.board.lock()
        self.board_connection = self.board.connect()
        for command in self.board_config.setup:
            self.board_connection.run(command)

        from tvm.rpc.tracker import Tracker

        self.tracker = Tracker(host, port=9000, port_end=9090, silent=False)
        time.sleep(1.0)

        if board_config.rebuild_runtime:
            self._build_runtime()
        try:
            self.server_process = ServerProcess(
                self.board.name,
                str(self.board.rundir / "tvm/python"),
                self.tracker.port,
            )
            self.server_process.start()
            print("waiting for server")
            while not self.server_process.running():
                time.sleep(1.0)
                print(".", end="")
            print("")

        except Exception as e:
            logger.warn("Could not start server process rebuilding device runtime")
            logger.warn(
                "Use board.rebuild_runtime=true to rebuild tvm runtime for target board"
            )
            raise Exception("Could not start server process")

        conn = tvm.rpc.connect_tracker("localhost", self.tracker.port)
        logger.info("%s", conn.text_summary())
        conn.close()

        self.runner = auto_scheduler.RPCRunner(
            device_key,
            host,
            self.tracker.port,
            priority,
            n_parallel,
            timeout,
            number,
            repeat,
            min_repeat_ms,
            cooldown_interval,
            enable_cpu_cache_flush,
        )

    def finish(self):
        logging.info("Finishing remote measurement context")
        # Terminate the tracker process
        if self.tracker is not None:
            if hasattr(self.tracker, "proc"):
                self.tracker.terminate()

        if self.board_connection is not None:
            self.board_connection.close()

        if self.board and self.board_config.teardown:
            with self.board.connect() as conn:
                for command in self.board_config.teardown:
                    conn.run(command)

        if self.board:
            self.board.unlock()

        time.sleep(0.5)

    def _build_runtime(self):
        connection = self.board_connection
        with connection.cd(self.board.rundir):
            connection.run("rm -rf tvm")
            connection.run(
                "git clone https://github.com/apache/tvm.git --depth 1 --recursive"
            )
            with connection.cd("tvm"):
                connection.run("cp cmake/config.cmake .")
                if self.board_config.opencl:
                    connection.run(
                        'sed -i "s/USE_OPENCL OFF/USE_OPENCL ON/" config.cmake'
                    )
                if self.board_config.cuda:
                    connection.run('sed -i "s/USE_CUDA OFF/USE_CUDA ON/" config.cmake')
                connection.run("make runtime -j4")

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.finish()
