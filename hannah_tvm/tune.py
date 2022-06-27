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
