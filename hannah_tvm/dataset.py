import hashlib
import logging
import pathlib
import pickle

from tvm.auto_scheduler.measure_record import dump_record_to_string

from .utils import RelayVisualizer

logger = logging.getLogger(__name__)


def clean_file_name(x):
    x = str(x)
    x = x.replace(" ", "")
    x = x.replace('"', "")
    x = x.replace("'", "")
    return x


class PerformanceDataset:
    def __init__(self, board: str, target: str) -> None:
        self.board = str(board)
        self.target = str(target)
        self._base_dir = self.database_file = (
            pathlib.Path(__file__).parent.resolve() / ".." / "dataset"
        )

    def _build_hash_path(self, hash: str, category: str, suffix: str):
        num_splits = 3
        split_length = 2
        hash_path = self._base_dir / category
        hash_path = hash_path / self.board / self.target
        assert len(hash) > num_splits * split_length

        for i in range(num_splits):
            hash_part = hash[:split_length]
            hash = hash[split_length:]

            hash_path = hash_path / hash_part

        hash_path = hash_path / hash

        hash_path = hash_path.with_suffix(suffix)
        return hash_path

    def add_program(self, network_name, relay_mod, params):
        logger.info("Adding relay program: %s", network_name)

        base_path = self._base_dir / "network_info" / self.board / network_name
        base_path.parent.mkdir(exist_ok=True, parents=True)

        plotter = RelayVisualizer()
        plotter = plotter.render(relay_mod, params, base_path)

        relay_txt = relay_mod.astext().encode("utf-8")
        network_path = base_path.with_suffix(".relay.txt")
        with network_path.open("wb") as out_file:
            out_file.write(relay_txt)

        network_pkl_path = base_path.with_suffix(".relay.pkl")
        with network_pkl_path.open("wb") as out_file:
            pickle.dump(relay_mod, out_file)

    def add_tasks(self, scheduler, network_name, tasks, task_weights=None):
        task_info_filename = (
            self._base_dir / "task_info" / self.board / scheduler / network_name
        )
        task_info_filename = task_info_filename.with_suffix(".pkl")
        task_info_filename.parent.mkdir(exist_ok=True, parents=True)

        with task_info_filename.open("wb") as f:
            pickle.dump((tasks, task_weights), f)

    def add_tuning_results(self, scheduler, results):
        base_folder = self._base_dir / "tuning_results" / self.board / scheduler
        base_folder.mkdir(exist_ok=True, parents=True)
        for result in results:
            inp, res = result
            workload_key = inp.task.workload_key
            base_filename = clean_file_name(f"{inp.task.workload_key}_{self.target}")
            target_file = base_folder / base_filename
            target_file = target_file.with_suffix(".json")

            logger.info("Logging results to %s", str(target_file))

            with target_file.open("a+") as f:
                target_str = dump_record_to_string(inp, res) + "\n"
                f.write(target_str)

    def add_measurement(self, network_name, profile_results, debug_results):
        logger.info("Adding Measurement result")
        result_path = self._base_dir / "network_results" / self.board / network_name
        result_path.parent.mkdir(exist_ok=True, parents=True)
