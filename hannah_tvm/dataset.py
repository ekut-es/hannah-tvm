import logging
import pathlib
import hashlib

from multiprocessing import Lock

logger = logging.getLogger(__name__)


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

    def add_program(self, network_name, relay_mod):
        logger.info("Adding relay program: %s", network_name)

        relay_txt = relay_mod.astext().encode("utf-8")

        network_path = self._base_dir / "network_info" / self.board / network_name
        network_path = network_path.with_suffix(".relay.txt")
        network_path.parent.mkdir(exist_ok=True, parents=True)

        with network_path.open("wb") as out_file:
            out_file.write(relay_txt)

    def add_measurement(self, network_name, profile_results, debug_results):
        logger.info("Adding Measurement result")
        result_path = self._base_dir / "network_results" / self.board / network_name
        result_path.parent.mkdir(exist_ok=True, parents=True)
