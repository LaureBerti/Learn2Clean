import os
from pathlib import Path

import pandas as pd
import pytest
from dotenv import load_dotenv

# .env file must contains
# KAGGLE_USERNAME=<user>
# KAGGLE_KEY=<key>
load_dotenv()

from learn2clean.loaders import KaggleLoader

TEST_DATASET_ID = "sapal6/the-testcase-dataset"
TARGET_FILE_NAME = "Test_cases.csv"

has_kaggle_creds = "KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ


@pytest.mark.skipif(
    not has_kaggle_creds,
    reason="KAGGLE_USERNAME and KAGGLE_KEY required for integration test.",
)
@pytest.mark.kaggle_integration
def test_kaggle_loader_integration(tmp_path: Path):
    """
    Smoke test: Tries to actually download a small dataset from Kaggle.
    Requires internet connection and valid API credentials.
    """
    print(f"\n[Smoke Test] Downloading {TEST_DATASET_ID} to {tmp_path}...")

    # 1. Initialization
    loader = KaggleLoader(
        dataset_id=TEST_DATASET_ID,
        download_dir=str(tmp_path),
        filename=TARGET_FILE_NAME,
    )

    df = loader.load()

    assert isinstance(df, pd.DataFrame), "The result should be a pandas DataFrame"
    assert not df.empty, "The downloaded dataset should not be empty"

    expected_file = tmp_path / TARGET_FILE_NAME
    assert expected_file.exists(), "The CSV file should persist on disk"

    assert df.shape[0] > 0

    print(f"\n[Success] Downloaded {df.shape[0]} rows and {df.shape[1]} columns.")
