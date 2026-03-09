import numpy as np
import pandas as pd
import pytest
from gymnasium import spaces

from learn2clean.observers.data_stats_observer import DataStatsObserver


class TestDataStatsObserver:
    """Test suite for the DataStatsObserver class."""

    @pytest.fixture
    def observer(self):
        return DataStatsObserver(n_actions=3)

    @pytest.fixture
    def sample_df(self):
        # 3 rows, 2 cols (1 num, 1 cat), 1 NaN
        return pd.DataFrame({"A": [1.0, 2.0, np.nan], "B": ["x", "y", "z"]})

    def test_initialization_validation(self):
        """Verifies n_actions check."""
        obs = DataStatsObserver(n_actions=0)
        with pytest.raises(ValueError, match="n_actions must be positive"):
            obs.get_observation_space()

    def test_observation_structure(self, observer):
        """Verifies keys and shapes of the space."""
        space = observer.get_observation_space()
        assert isinstance(space, spaces.Dict)
        assert space["dataset_stats"].shape == (5,)
        assert space["action_history"].shape == (3,)

    def test_values_correctness(self, observer, sample_df):
        """Verifies that statistics are calculated correctly."""
        history = np.array([0, 1, 0])
        obs = observer.observe(sample_df, None, action_history=history)

        # Stats expected: [Rows=3, Cols=2, Nulls=1, Num=1, Cat=1]
        expected_stats = np.array([3, 2, 1, 1, 1], dtype=np.float32)

        np.testing.assert_array_equal(obs["dataset_stats"], expected_stats)
        np.testing.assert_array_equal(obs["action_history"], history.astype(np.float32))

    def test_fallback_no_history(self, observer, sample_df):
        """Verifies behavior when history is None."""
        obs = observer.observe(sample_df, None, action_history=None)
        assert np.all(obs["action_history"] == 0.0)
