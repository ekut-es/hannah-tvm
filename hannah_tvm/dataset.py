import hashlib
import itertools
import json
import logging
import pathlib
import pickle
from collections import OrderedDict, namedtuple
from typing import Any, Dict, Iterable, List, Optional
from unittest import result

import numpy as np
import pandas as pd
from tvm import auto_scheduler, autotvm
from tvm.auto_scheduler.measure_record import dump_record_to_string

from .utils import RelayVisualizer

logger = logging.getLogger(__name__)

_BASE_DIR = pathlib.Path(__file__).parent.resolve() / ".." / "dataset"

NetworkResult = namedtuple(
    "NetworkResult",
    ["board", "target", "model", "tuner", "measurement", "relay", "tir_primfuncs"],
)


def clean_file_name(x):
    x = str(x)
    x = x.replace(" ", "")
    x = x.replace('"', "")
    x = x.replace("'", "")
    x = x.replace(".", "_")
    return x


class PerformanceDataset:
    def __init__(self, board: str, target: str) -> None:
        self.board = str(board)
        self.target = str(target)
        self._base_dir = _BASE_DIR

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
        base_folder = self._get_tuning_results_dir(scheduler)
        if scheduler == "auto_scheduler":
            print(results)
            for inp, res in results:

                workload_key = inp.task.workload_key
                base_filename = clean_file_name(f"{workload_key}_{self.target}")
                target_file = base_folder / base_filename
                target_file = target_file.with_suffix(".json")

                logger.debug("Logging results to %s", str(target_file))

                with target_file.open("a+") as f:
                    target_str = dump_record_to_string(inp, res)
                    f.write(target_str)

        elif scheduler == "autotvm":
            wkl_dict = OrderedDict()
            for inp, res in results:
                str_key = self._gen_autotvm_task_key(inp.task)
                base_filename = clean_file_name(str_key)
                target_file = base_folder / base_filename
                target_file = target_file.with_suffix(".json")

                with open(target_file, "a+") as fout:
                    fout.write(autotvm.record.encode(inp, res) + "\n")

    def load_tuning_results(
        self, scheduler, tasks: Optional[Iterable[auto_scheduler.SearchTask]] = None
    ):
        base_folder = self._get_tuning_results_dir(scheduler)
        res_iterator = []
        if scheduler == "auto_scheduler":
            task_files = []
            if task_files is not None:
                for task in tasks:
                    workload_key = task.workload_key
                    base_filename = clean_file_name(f"{workload_key}_{self.target}")
                    target_file = base_folder / base_filename
                    target_file = target_file.with_suffix(".json")

                    if target_file.exists():
                        task_files.append(target_file)

            readers = []
            for task_file in task_files:
                logger.debug("Creating reader for file: %s", str(task_file))
                reader = auto_scheduler.RecordReader(str(task_file))

                readers.append(reader)

            res_iterator = list(itertools.chain(*readers))
        elif scheduler == "autotvm":
            generators = []
            for inp in tasks:
                str_key = self._gen_autotvm_task_key(inp)
                base_filename = clean_file_name(str_key)
                target_file = base_folder / base_filename
                target_file = target_file.with_suffix(".json")
                if target_file.exists():
                    gen = autotvm.record.load_from_file(target_file)
                    generators.append(gen)
            res_iterator = list(itertools.chain(*generators))
        return res_iterator

    def _gen_autotvm_task_key(self, task):
        str_key = "_".join(
            [task.name, str(task.args), str(task.kwargs), str(self.target)]
        )
        return str_key

    def add_measurement_network(self, scheduler, network_name, relay_module):
        logger.info("Adding target relay")
        result_path = (
            self._base_dir
            / "network_results"
            / self.board
            / scheduler
            / f"{network_name}_{str(self.target)}.relay.pkl"
        )
        result_path.parent.mkdir(exist_ok=True, parents=True)
        with result_path.open("wb") as result_file:
            pickle.dump(relay_module, result_file)

    def add_measurement_primfuncs(self, scheduler, network_name, primfuncs):
        logger.info("Adding target relay")
        result_path = (
            self._base_dir
            / "network_results"
            / self.board
            / scheduler
            / f"{network_name}_{str(self.target)}.primfuncs.pkl"
        )
        result_path.parent.mkdir(exist_ok=True, parents=True)
        with result_path.open("wb") as result_file:
            pickle.dump(primfuncs, result_file)

    def add_measurement(self, scheduler, network_name, results: Dict[str, Any]):
        logger.info("Adding Measurement result")
        result_path = (
            self._base_dir
            / "network_results"
            / self.board
            / scheduler
            / f"{network_name}_{str(self.target)}.json"
        )
        result_path.parent.mkdir(exist_ok=True, parents=True)
        with result_path.open("w") as result_file:
            json.dump(results, result_file)

    def _get_tuning_results_dir(self, scheduler):
        base_folder = self._base_dir / "tuning_results" / self.board / scheduler
        base_folder.mkdir(exist_ok=True, parents=True)
        return base_folder


class DatasetFull:
    def __init__(self):
        self._base_dir = _BASE_DIR

    def measurements(self) -> pd.DataFrame:
        base_folder = self._base_dir / "network_results"
        measurements = []
        for result_file in base_folder.glob("*/*/*.json"):
            parts = result_file.parts
            model_name = parts[-1].split(".")[0]
            model_name, target_name = (
                "_".join(model_name.split("_")[:-1]),
                model_name.split("_")[-1],
            )
            scheduler_name = parts[-2]
            board_name = parts[-3]

            result = {}
            result["Model"] = model_name
            result["Board"] = board_name
            result["Tuner"] = scheduler_name
            result["Target"] = target_name

            with result_file.open() as result_stream:
                record = json.load(result_stream)
                result["Duration (us)"] = np.mean(record["Duration (us)"])
                result["Duration StdDev"] = np.std(record["Duration (us)"])
                result["Duration PtP"] = np.ptp(record["Duration (us)"])

            measurements.append(result)

        df = pd.DataFrame.from_records(measurements)

        df = df.sort_values(["Board", "Model", "Tuner"])

        return df

    def network_results(self) -> List[NetworkResult]:
        base_folder = self._base_dir / "network_results"
        measurements = []
        for result_file in base_folder.glob("*/*/*.json"):
            parts = result_file.parts
            model_name = parts[-1].split(".")[0]
            model_name, target_name = (
                "_".join(model_name.split("_")[:-1]),
                model_name.split("_")[-1],
            )
            scheduler_name = parts[-2]
            board_name = parts[-3]

            with result_file.open() as result_stream:
                record = json.load(result_stream)

            relay_file = result_file.with_suffix(".relay.pkl")
            tir_file = result_file.with_suffix(".primfuncs.pkl")

            relay = None
            tir = None

            if relay_file.exists():
                with relay_file.open("rb") as f:
                    relay = pickle.load(f)

            if tir_file.exists():
                with tir_file.open("rb") as f:
                    tir = pickle.load(f)

            result = NetworkResult(
                board_name, target_name, model_name, scheduler_name, record, relay, tir
            )
            measurements.append(result)

        return measurements
