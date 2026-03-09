from learn2clean.loaders.csv_loader import CSVLoader
from learn2clean.evaluation.classification.multinomial_naive_bayes import (
    MultinomialNaiveBayes,
)


def test_multinomial_nb_accuracy():
    dataset = CSVLoader(dataset="titanic.csv").load()

    evaluator = MultinomialNaiveBayes(
        target_col="Survived",
        test_size=0.2,
        random_state=42,
        alpha=0.5,
    )

    accuracy = evaluator.evaluate(dataset)
    print("Multinomial Naive Bayes Accuracy =", accuracy)

    assert isinstance(accuracy, float)
    assert 0.0 <= accuracy <= 1.0

    assert accuracy > 0.5
