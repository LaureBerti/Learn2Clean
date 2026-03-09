import numpy as np
import pandas as pd

from learn2clean.actions import RandomImputer, MeanImputer


def test_random_imputation_preserves_variance() -> None:
    """
    Demonstrates that Random Imputation preserves the variance of the data,
    whereas Mean Imputation shrinks it.
    """
    # 1. Create a dataset with high variance
    np.random.seed(42)
    data = np.random.normal(loc=10, scale=5, size=1000)  # Mean=10, Std=5
    df = pd.DataFrame({"A": data})

    # 2. Introduce 50% missing values
    df_missing = df.copy()
    mask = np.random.choice([True, False], size=len(df), p=[0.5, 0.5])
    df_missing.loc[mask, "A"] = np.nan

    # Calculate original variance (approx 25)
    original_var = df["A"].var()

    # 3. Apply Mean Imputation
    mean_imp = MeanImputer()
    df_mean = mean_imp.fit(df_missing).transform(df_missing)
    mean_var = df_mean["A"].var()

    # 4. Apply Random Imputation
    rand_imp = RandomImputer(random_state=42)
    df_rand = rand_imp.fit(df_missing).transform(df_missing)
    rand_var = df_rand["A"].var()

    print(f"\nOriginal Var: {original_var:.2f}")
    print(f"Mean Imp Var: {mean_var:.2f} (Expected to drop significantly)")
    print(f"Random Imp Var: {rand_var:.2f} (Expected to be close to original)")

    # Assertions
    # Mean imputation replaces 50% of data with a constant -> Variance drops by roughly half
    assert mean_var < original_var * 0.7

    # Random imputation replaces data with same distribution -> Variance stays close
    assert abs(rand_var - original_var) < original_var * 0.2
