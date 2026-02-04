import hydra
from omegaconf import DictConfig

from tasks.generate_csv import run


@hydra.main(version_base="1.3", config_path="../config", config_name="generate_csv")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
