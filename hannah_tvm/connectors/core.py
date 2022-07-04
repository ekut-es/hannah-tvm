from abc import ABC, abstractmethod, abstractproperty
from typing import List, Literal


class BuildArtifactHandle:
    """Connector specific handle for build artifact and sometimes corresponding remote connection"""

    pass


class TaskConnector(ABC):
    @abstractmethod
    def setup(self):
        """Setup board for task configuration"""
        pass

    @abstractmethod
    def target(self):
        """return tvm target for this board"""
        pass

    @abstractmethod
    def builder(self, tuner=None):
        """return target specific builder for selected tuning method"""
        pass

    @abstractmethod
    def runner(self, tuner=None):
        """return target specific runner for selected tuning method"""
        pass

    @abstractmethod
    def upload(self, mod):
        """Export the target library to and upload/flash for remote"""
        pass

    @abstractmethod
    def measure(self, handle, inputs):
        """Measure a full neural network"""
        pass

    @abstractmethod
    def profile(self, handle, inputs):
        """Run per layer profiling of full neural network"""
        pass

    @abstractmethod
    def teardown(self):
        """Teardown task called at the end of each task executiion"""
        pass


class BoardConnector(ABC):
    def supported_tuners() -> List[Literal["autotvm", "auto_scheduler"]]:
        return ["autotvm", "auto_scheduler"]

    @abstractmethod
    def setup(self):
        """Called for each board connection"""
        pass

    @abstractmethod
    def task_connector(self) -> TaskConnector:
        """Return task connector for target board"""
        pass

    @abstractmethod
    def is_alive(self) -> bool:
        """Return true if the board connector is still alive"""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the board connection"""
        pass

    @abstractmethod
    def teardown(self):
        """Teardown board for measurements called at the end of each task"""
        pass

    @abstractmethod
    def boards_available(self) -> int:
        """Get number of available boards not currently used by a tuning task"""
        pass
