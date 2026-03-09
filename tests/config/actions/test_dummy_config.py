from hydra.utils import instantiate
from omegaconf import OmegaConf

from learn2clean.actions import DummyAdd
from learn2clean.configs.actions.dummy import DummyAddConfig


def test_dummy_config_instantiates_correctly():
    cfg = OmegaConf.structured(DummyAddConfig)
    action = instantiate(cfg)

    assert isinstance(
        action, DummyAdd
    ), f"Config must be DummyAdd, got : {type(action)}"
    assert action.name == "DummyAdd"
    if hasattr(action, "increment"):
        assert action.increment == 1
    elif hasattr(action, "params"):
        assert action.params.get("increment") == 1
