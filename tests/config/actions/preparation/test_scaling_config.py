from hydra.utils import instantiate
from omegaconf import OmegaConf

from learn2clean.actions import (
    DecimalScaler,
    Log10Scaler,
    MinMaxScaler,
    QuantileScaler,
    ZScoreScaler,
)
from learn2clean.configs.actions.preparation.scaling import (
    DecimalScalerConfig,
    Log10ScalerConfig,
    MinMaxScalerConfig,
    QuantileScalerConfig,
    ZScoreScalerConfig,
)


class TestScalingConfigs:
    def test_decimal_scaler_config(self):
        cfg = OmegaConf.structured(DecimalScalerConfig)
        action = instantiate(cfg)

        assert isinstance(action, DecimalScaler)
        assert action.name == "DecimalScaler"

    def test_log10_scaler_config(self):
        cfg = OmegaConf.structured(Log10ScalerConfig)
        action = instantiate(cfg)
        assert isinstance(action, Log10Scaler)
        assert action.params.get("shift_epsilon") == 1e-6
        cfg_override = OmegaConf.structured(Log10ScalerConfig(shift_epsilon=1.0))
        action_override = instantiate(cfg_override)
        assert action_override.params.get("shift_epsilon") == 1.0

    def test_min_max_scaler_config(self):
        cfg = OmegaConf.structured(MinMaxScalerConfig)
        action = instantiate(cfg)
        assert isinstance(action, MinMaxScaler)

    def test_quantile_scaler_config(self):
        cfg = OmegaConf.structured(QuantileScalerConfig)
        action = instantiate(cfg)
        assert isinstance(action, QuantileScaler)
        assert action.params.get("n_quantiles") == 1000
        assert action.params.get("output_distribution") == "uniform"
        cfg_override = OmegaConf.structured(
            QuantileScalerConfig(
                n_quantiles=10, output_distribution="normal", random_state=42
            )
        )
        action_override = instantiate(cfg_override)
        assert action_override.params.get("n_quantiles") == 10
        assert action_override.params.get("output_distribution") == "normal"
        assert action_override.params.get("random_state") == 42

    def test_zscore_scaler_config(self):
        cfg = OmegaConf.structured(ZScoreScalerConfig)
        action = instantiate(cfg)
        assert isinstance(action, ZScoreScaler)
        assert action.params.get("with_mean") is True
        assert action.params.get("with_std") is True
        cfg_override = OmegaConf.structured(
            ZScoreScalerConfig(with_mean=False, with_std=True)
        )
        action_override = instantiate(cfg_override)
        assert action_override.params.get("with_mean") is False
        assert action_override.params.get("with_std") is True

    def test_scaling_columns_selection(self):
        cfg = OmegaConf.structured(
            ZScoreScalerConfig(columns=["salary", "height"], exclude=["id"])
        )
        action = instantiate(cfg)
        assert action.columns == ["salary", "height"]
        assert action.exclude == ["id"]
