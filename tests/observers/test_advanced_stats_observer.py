import numpy as np
import pandas as pd
import pytest
from gymnasium import spaces

from learn2clean.observers.advanced_stats_observer import AdvancedStatsObserver


class TestAdvancedStatsObserver:
    """Test suite for the AdvancedStatsObserver class."""

    @pytest.fixture
    def observer(self):
        """Returns an observer initialized with 3 actions."""
        return AdvancedStatsObserver(n_actions=3)

    @pytest.fixture
    def numeric_df(self):
        """
        Creates a DataFrame with specific statistical properties.
        - Col A: Symmetric (Skew ~ 0)
        - Col B: Skewed (1, 1, 10)
        - Col C: Zeros (for sparsity)
        """
        return pd.DataFrame(
            {
                "A": [1, 2, 3, 2, 1],  # Symmetric
                "B": [1, 1, 1, 1, 10],  # Highly skewed / High Kurtosis
                "C": [0, 0, 0, 5, 5],  # 60% zeros
            }
        )

    def test_initialization_check(self):
        """Verifies that n_actions must be positive."""
        obs = AdvancedStatsObserver(n_actions=0)
        with pytest.raises(ValueError, match="n_actions must be positive"):
            obs.get_observation_space()

    def test_observation_space_structure(self, observer):
        """Verifies the shape and keys of the observation space."""
        space = observer.get_observation_space()

        assert isinstance(space, spaces.Dict)
        assert "advanced_stats" in space.spaces
        assert "action_history" in space.spaces

        assert space["advanced_stats"].shape == (5,)
        assert space["action_history"].shape == (3,)

    def test_stats_calculation(self, observer, numeric_df):
        """Verifies that statistics are calculated (non-zero) for numeric data."""
        obs = observer.observe(numeric_df, y=None)
        stats = obs["advanced_stats"]

        # 1. Skewness should be non-zero due to col B
        assert stats[0] != 0.0

        # 2. Kurtosis should be non-zero due to col B
        assert stats[1] != 0.0

        # 3. Correlation should be calculated (3 columns)
        assert stats[2] >= 0.0

        # 4. Sparsity: Total cells = 15. Zeros = 3 (in Col C). Ratio = 3/15 = 0.2
        expected_sparsity = 3 / 15
        assert np.isclose(stats[3], expected_sparsity)

        # 5. Label balance should be 0.0 (y is None)
        assert stats[4] == 0.0

    def test_perfect_correlation(self, observer):
        """Verifies correlation calculation for perfectly correlated features."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [2, 4, 6]})  # Corr = 1.0
        obs = observer.observe(df, y=None)

        # Index 2 is Mean Correlation
        assert np.isclose(obs["advanced_stats"][2], 1.0)

    def test_label_balance(self, observer):
        """Verifies label balance calculation for categorical targets."""
        df = pd.DataFrame({"A": [1, 2, 3]})

        # Target: 'a' appears 2/3, 'b' appears 1/3. Minority is 'b' -> 0.333...
        y = pd.Series(["a", "a", "b"])

        obs = observer.observe(df, y=y)
        balance = obs["advanced_stats"][4]

        assert np.isclose(balance, 1 / 3)

    def test_label_balance_regression(self, observer):
        """Verifies label balance is 0 for regression (numeric target)."""
        df = pd.DataFrame({"A": [1, 2, 3]})
        y = pd.Series([10.5, 20.1, 30.2])  # Numeric

        obs = observer.observe(df, y=y)
        assert obs["advanced_stats"][4] == 0.0

    def test_edge_case_no_numeric_cols(self, observer):
        """Verifies behavior when DataFrame has no numeric columns."""
        df = pd.DataFrame({"A": ["a", "b"], "B": ["x", "y"]})
        obs = observer.observe(df, y=None)

        stats = obs["advanced_stats"]
        # Should return all zeros
        np.testing.assert_array_equal(stats, np.zeros(5, dtype=np.float32))

    def test_history_handling(self, observer, numeric_df):
        """Verifies action history processing."""
        history = np.array([1, 0, 1])
        obs = observer.observe(numeric_df, y=None, action_history=history)

        np.testing.assert_array_equal(obs["action_history"], history.astype(np.float32))

    def test_history_fallback(self, observer, numeric_df):
        """Verifies fallback to zeros if history is None."""
        obs = observer.observe(numeric_df, y=None, action_history=None)

        np.testing.assert_array_equal(
            obs["action_history"], np.zeros(3, dtype=np.float32)
        )
