from learn2clean.loaders.csv_loader import CSVLoader
from learn2clean.evaluation.classification.naive_bayes import NaiveBayesEvaluator


def test_naive_bayes_accuracy():
    dataset = CSVLoader(dataset="titanic.csv").load()

    # Instantiate evaluator
    evaluator = NaiveBayesEvaluator(
        target_col="Survived", test_size=0.2, random_state=42
    )

    # Compute accuracy
    accuracy = evaluator.evaluate(dataset)
    print("Naive Bayes Accuracy =", accuracy)

    # Assertions
    assert isinstance(accuracy, float)
    assert 0.0 <= accuracy <= 1.0

    # Optional: ensure classification is better than random guess
    assert accuracy > 0.5
