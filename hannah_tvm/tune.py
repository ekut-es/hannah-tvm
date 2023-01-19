#
# Copyright (c) 2023 University of Tübingen.
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

import logging

import hydra
from omegaconf import OmegaConf

from .experiment_scheduler import TuningExperimentScheduler

logger = logging.getLogger(__name__)


@hydra.main(config_name="config", config_path="conf", version_base="1.2")
def main(config):
    logging.captureWarnings(True)
    logger.info(OmegaConf.to_yaml(config))

    scheduler = TuningExperimentScheduler(config)
    scheduler.run()


if __name__ == "__main__":
    main()
