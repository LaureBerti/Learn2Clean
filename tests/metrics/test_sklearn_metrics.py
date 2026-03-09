from sklearn.metrics import get_scorer, get_scorer_names


def test_list_sklearn_metrics():
    names = get_scorer_names()
    print("\n")
    for name in names:
        print(f"- {name}")
        scorer = get_scorer(name)
