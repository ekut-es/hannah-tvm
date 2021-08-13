import atexit
import logging
import multiprocessing
import time
from tqdm.std import tqdm


import tvm.auto_scheduler as auto_scheduler
import tvm.rpc
import tvm.rpc.tracker

try:
    from automate.config import AutomateConfig
    from automate.context import AutomateContext
    from automate.utils.network import find_local_port, find_remote_port

    automate_available = True
except ModuleNotFoundError:
    automate_available = False

logger = logging.getLogger("hannah_tvm.measure")

_automate_context = None
_automate_measure_contexts = []


@atexit.register
def cleanup():
    logging.info("Cleaning up measurement contexts")
    for context in _automate_measure_contexts:
        context.finish()


class ServerProcess(multiprocessing.Process):
    def __init__(self, board_config, tracker_port):
        super().__init__()
        self.board_config = board_config

        self.tracker_port = tracker_port
        self._pconn, self._cconn = multiprocessing.Pipe()

    def run(self):
        name = self.board_config.name
        automate_config = AutomateConfig()
        automate_context = AutomateContext(automate_config)

        board = automate_context.board(name)
        python_path = str(board.rundir / "tvm" / "python")
        tracker_port = self.tracker_port

        try:

            with board.connect() as board_connection:

                if self.board_config.rebuild_runtime:
                    self._build_runtime(board_connection, board)

                logger.info("Running setup commands")
                for setup in self.board_config.setup:
                    board_connection.run(setup)

                logger.info(
                    "forwarding remote port %d to local port %i", 9000, tracker_port
                )
                with board_connection.forward_remote(9000, tracker_port):
                    local_port = find_local_port(9091, 90199)
                    logger.info(
                        "forwarding local port %d to remote port %i",
                        local_port,
                        local_port,
                    )
                    with board_connection.forward_local(local_port, local_port):
                        logger.info("Starting remote server")
                        promise = board_connection.run(
                            f"python3.6 -m tvm.exec.rpc_server --key {name} --host localhost --port={local_port} --port-end={local_port+1} --tracker=localhost:9000",
                            env={"PYTHONPATH": str(python_path)},
                            shell=True,
                            warn=True,
                            pty=True,
                            asynchronous=True,
                        )

                        killed = False
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
                                    killed = True
                            time.sleep(2.0)

                        result = promise.join()

                        if (not result) and (not killed):
                            logger.info("Result %s", str(result))
                            self._cconn.send(str(result))

                logger.info("Running teardown commands")
                for teardown in self.board_config.teardown:
                    board_connection.run(teardown)

                logger.info("Server has finished")
        except Exception as e:
            logger.critical("Could not start server process")
            logger.critical(str(e))

    def _build_runtime(self, connection, board):
        with connection.cd(board.rundir):
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

    @property
    def running(self):
        if not self.is_alive():
            return False

        return True

    def finish(self):
        self._pconn.send("exit")
        while True:
            try:
                if not self.running:
                    return
                time.sleep(0.5)
            except Exception as e:
                print(str(e))


class PrintPBarInfo(tvm.auto_scheduler.task_scheduler.TaskSchedulerCallback):
    """The callback that prints a table of current progress."""

    def __init__(self, results):
        self.results = results

    def post_tune(self, task_scheduler: tvm.auto_scheduler.TaskScheduler, task_id):

        # overall info
        if all(cost < 1e9 for cost in task_scheduler.best_costs):
            total_latency_str = "%.3f" % (task_scheduler.cur_score.value * 1e3)
        else:
            total_latency_str = "-"

        logger.info(
            "Estimated total latency: %s ms\tTrials: %d\tUsed time : %.0f s\tNext ID: %d\t"
            % (
                total_latency_str,
                task_scheduler.ct,
                time.time() - task_scheduler.tic,
                task_id,
            )
        )
