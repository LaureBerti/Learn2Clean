from hydra.utils import instantiate
from omegaconf import OmegaConf

from learn2clean.actions import (
    IQROutlierCleaner,
    LocalOutlierFactorCleaner,
    ZScoreOutlierCleaner,
)
from learn2clean.configs.actions.cleaning.outlier_detection import (
    IQROutlierCleanerConfig,
    LocalOutlierFactorCleanerConfig,
    ZScoreOutlierCleanerConfig,
)


class TestOutlierDetectionConfigs:
    def test_iqr_cleaner_config(self):
        cfg = OmegaConf.structured(IQROutlierCleanerConfig)
        action = instantiate(cfg)
        assert isinstance(action, IQROutlierCleaner)
        assert action.name == "IQROutlierCleaner"
        assert action.params.get("factor") == 1.5
        assert action.params.get("method") == "mask"
        cfg_override = OmegaConf.structured(
            IQROutlierCleanerConfig(factor=3.0, method="drop", numeric_only=False)
        )
        action_override = instantiate(cfg_override)

        assert action_override.params.get("factor") == 3.0
        assert action_override.params.get("method") == "drop"
        assert action_override.params.get("numeric_only") is False

    def test_lof_cleaner_config(self):
        cfg = OmegaConf.structured(LocalOutlierFactorCleanerConfig)
        action = instantiate(cfg)
        assert isinstance(action, LocalOutlierFactorCleaner)
        assert action.name == "LocalOutlierFactorCleaner"
        assert action.params.get("n_neighbors") == 20
        assert action.params.get("contamination") == "auto"
        cfg_override = OmegaConf.structured(
            LocalOutlierFactorCleanerConfig(
                n_neighbors=50, contamination=0.1, method="drop"
            )
        )
        action_override = instantiate(cfg_override)
        assert action_override.params.get("n_neighbors") == 50
        assert action_override.params.get("contamination") == 0.1
        assert action_override.params.get("method") == "drop"

    def test_zscore_cleaner_config(self):
        cfg = OmegaConf.structured(ZScoreOutlierCleanerConfig)
        action = instantiate(cfg)
        assert isinstance(action, ZScoreOutlierCleaner)
        assert action.name == "ZScoreOutlierCleaner"
        assert action.params.get("threshold") == 3.0
        cfg_override = OmegaConf.structured(
            ZScoreOutlierCleanerConfig(
                threshold=2.5, method="clip", columns=["salary", "age"]
            )
        )
        action_override = instantiate(cfg_override)
        assert action_override.params.get("threshold") == 2.5
        assert action_override.params.get("method") == "clip"
        assert action_override.columns == ["salary", "age"]

    def test_outlier_invalid_method(self):
        cfg = OmegaConf.structured(IQROutlierCleanerConfig(method="invalid_method"))
        action = instantiate(cfg)
        assert action.params.get("method") == "invalid_method"
