import logging
import hydra


from .experiment_scheduler import TuningExperimentScheduler

logger = logging.getLogger(__name__)


@hydra.main(config_name="config", config_path="conf")
def main(config):
    scheduler = TuningExperimentScheduler(config)
    scheduler.run()


if __name__ == "__main__":
    main()
