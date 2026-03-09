from dataclasses import dataclass
from typing import Any

from hydra.core.config_store import ConfigStore

from ..base import ActionConfig


@dataclass
class ExactDeduplicatorConfig(ActionConfig):
    name: str = "ExactDeduplicator"
    _target_: str = "learn2clean.actions.ExactDeduplicator"
    keep: Any = "first"


@dataclass
class ApproximateDeduplicatorConfig(ActionConfig):
    name: str = "ApproximateDeduplicator"
    _target_: str = "learn2clean.actions.ApproximateDeduplicator"
    threshold: float = 0.95
    comparison_type: str = "jarowinkler"  # "levenshtein", "jarowinkler", etc.
    keep: Any = "first"
    max_rows_limit: int = 10000


@dataclass
class JaccardSimilarityDeduplicatorConfig(ActionConfig):
    name: str = "JaccardSimilarityDeduplicator"
    _target_: str = (
        "learn2clean.actions.cleaning.deduplication.jaccard_similarity_deduplicator.JaccardSimilarityDeduplicator"
    )
    threshold: float = 0.8
    num_perm: int = 128
    combine_columns: bool = True
    keep: Any = "first"


def register_deduplication_configs(cs: ConfigStore) -> None:
    group = "action/deduplication"
    cs.store(group=group, name="exact", node=ExactDeduplicatorConfig)
    cs.store(group=group, name="approximate", node=ApproximateDeduplicatorConfig)
    cs.store(group=group, name="jaccard", node=JaccardSimilarityDeduplicatorConfig)
