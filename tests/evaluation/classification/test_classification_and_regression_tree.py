from learn2clean.loaders.csv_loader import CSVLoader
from learn2clean.evaluation.classification.classification_and_regression_tree import (
    CART,
)


def test_cart_accuracy():
    dataset = CSVLoader(dataset="titanic.csv").load()

    evaluator = CART(
        target_col="Survived",
        test_size=0.2,
        random_state=42,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
    )

    accuracy = evaluator.evaluate(dataset)
    print("CART accuracy:", accuracy)

    assert isinstance(accuracy, float)
    assert 0.0 <= accuracy <= 1.0

    assert accuracy > 0.5
