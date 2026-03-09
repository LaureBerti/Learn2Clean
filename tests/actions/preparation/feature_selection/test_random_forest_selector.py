import pandas as pd
from sklearn.datasets import make_classification

from learn2clean.actions import RandomForestSelector
from tests.utils.actions.assert_action_pipeline import assert_action_pipeline


def test_random_forest_selector_max_features() -> None:
    X_data, y_data = make_classification(
        n_samples=50,
        n_features=5,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        random_state=42,
    )

    column_names = [f"f{i}" for i in range(5)]
    df = pd.DataFrame(X_data, columns=column_names)

    rf_selector = RandomForestSelector(
        max_features=1,
        n_estimators=10,
        threshold=-1.0,
        random_state=42,
    )
    expected_output_data = {"f0": df["f0"].tolist()}

    assert_action_pipeline(
        rf_selector,
        df.to_dict("list"),
        expected_output_data,
        df.to_dict("list"),
        y_data.tolist(),
    )
