from dataclasses import dataclass
from .core import BoardConnector, TaskConnector, BuildArtifactHandle


@dataclass
class MicroTVMBuildArtifact(BuildArtifactHandle):
    pass


class MicroTVMTaskConnector(TaskConnector):
    pass


class MicroTVMBoardConnector(BoardConnector):
    pass
