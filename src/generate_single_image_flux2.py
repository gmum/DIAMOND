import hydra
from omegaconf import DictConfig

from tasks.single_image_flux2 import run


@hydra.main(version_base="1.3", config_path="../config", config_name="single_image_flux2")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
