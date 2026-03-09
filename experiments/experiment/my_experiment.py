import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from experiments.tools.load_and_split_data import load_and_split_data
from learn2clean.configs import register_all_configs


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="experiment/my_experiment",
)
def main(cfg: DictConfig) -> None:
    yaml_str = OmegaConf.to_yaml(cfg)
    print(yaml_str)
    print(load_and_split_data(cfg))
    # TODO write experiment code here


if __name__ == "__main__":
    os.environ["PROJECT_ROOT"] = str(Path(__file__).resolve().parents[2])
    register_all_configs()
    main()
