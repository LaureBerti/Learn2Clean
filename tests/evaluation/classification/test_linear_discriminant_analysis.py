from learn2clean.loaders.csv_loader import CSVLoader
from learn2clean.evaluation.classification.linear_discriminant_analysis import LDA


def test_lda_accuracy():
    dataset = CSVLoader(dataset="titanic.csv").load()

    evaluator = LDA(target_col="Survived", test_size=0.2, random_state=42)

    accuracy = evaluator.evaluate(dataset)
    print(accuracy)

    assert isinstance(accuracy, float)
    assert 0.0 <= accuracy <= 1.0
    assert accuracy > 0.5
