import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from learn2clean.configs import register_all_configs


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="tutorials/09_sequential_sb3_ppo",
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    os.environ["PROJECT_ROOT"] = str(Path(__file__).resolve().parents[2])
    register_all_configs()
    main()
