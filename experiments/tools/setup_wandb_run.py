import wandb
from omegaconf import DictConfig, OmegaConf
from wandb.sdk.wandb_run import Run


def setup_wandb_run(cfg: DictConfig) -> Run:
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    return wandb.init(
        name=cfg.experiment.name,
        config=config_dict,
        **cfg.wandb,
    )
