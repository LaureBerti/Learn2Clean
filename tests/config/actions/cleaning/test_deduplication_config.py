import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf

from learn2clean.actions import (
    ExactDeduplicator,
    ApproximateDeduplicator,
    JaccardSimilarityDeduplicator,
)
from learn2clean.configs.actions.cleaning.deduplication import (
    ExactDeduplicatorConfig,
    ApproximateDeduplicatorConfig,
    JaccardSimilarityDeduplicatorConfig,
)


class TestDeduplicationConfigs:
    def test_exact_deduplicator_config(self):
        cfg = OmegaConf.structured(ExactDeduplicatorConfig)
        action = instantiate(cfg)

        assert isinstance(action, ExactDeduplicator)
        assert action.name == "ExactDeduplicator"
        assert action.params.get("keep") == "first"
        cfg_override = OmegaConf.structured(
            ExactDeduplicatorConfig(
                keep="last", columns=["id", "email"], name="MyExactDedup"
            )
        )
        action_override = instantiate(cfg_override)
        assert action_override.name == "MyExactDedup"
        assert action_override.columns == ["id", "email"]
        assert action_override.params.get("keep") == "last"

    def test_approximate_deduplicator_config(self):
        cfg = OmegaConf.structured(ApproximateDeduplicatorConfig)
        action = instantiate(cfg)
        assert isinstance(action, ApproximateDeduplicator)
        assert action.name == "ApproximateDeduplicator"
        assert action.params.get("threshold") == 0.95
        assert action.params.get("comparison_type") == "jarowinkler"
        cfg_override = OmegaConf.structured(
            ApproximateDeduplicatorConfig(
                threshold=0.8, comparison_type="levenshtein", max_rows_limit=500
            )
        )
        action_override = instantiate(cfg_override)
        assert action_override.params.get("threshold") == 0.8
        assert action_override.params.get("comparison_type") == "levenshtein"
        assert action_override.params.get("max_rows_limit") == 500

    def test_jaccard_deduplicator_config(self):
        cfg = OmegaConf.structured(JaccardSimilarityDeduplicatorConfig)
        action = instantiate(cfg)
        assert isinstance(action, JaccardSimilarityDeduplicator)
        assert action.name == "JaccardSimilarityDeduplicator"
        assert action.params.get("threshold") == 0.8
        assert action.params.get("num_perm") == 128
        assert action.params.get("combine_columns") is True
        cfg_override = OmegaConf.structured(
            JaccardSimilarityDeduplicatorConfig(
                threshold=0.5, num_perm=256, combine_columns=False, keep="last"
            )
        )
        action_override = instantiate(cfg_override)
        assert action_override.params.get("threshold") == 0.5
        assert action_override.params.get("num_perm") == 256
        assert action_override.params.get("combine_columns") is False
        assert action_override.params.get("keep") == "last"
