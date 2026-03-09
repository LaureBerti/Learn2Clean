"""
CSVLoader module for Learn2Clean.

This module provides a concrete implementation of the DatasetLoader
for loading datasets from CSV files. It defines the CSVLoader class
which reads a CSV file from a specified path and returns a pandas
DataFrame. It handles basic error checking for file existence and
parsing issues.
"""

from pathlib import Path
from typing import Any

import pandas as pd

from learn2clean.loaders.base import DatasetLoader


class CSVLoader(DatasetLoader):
    """
    A concrete implementation of the DatasetLoader for loading CSV files.

    This class reads a CSV file from a specified path and returns its content
    as a pandas DataFrame. It extends the base `DatasetLoader` to provide
    CSV-specific loading logic and error handling.
    """

    path: Path
    read_csv_kwargs: dict[str, Any]

    # Calculate Project Root and Data Root once
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
    PROJECT_DATA_ROOT = PROJECT_ROOT / "data"

    def __init__(
        self,
        target_col: str | None = None,
        file_path: str | None = None,
        dataset: str | None = None,
        read_csv_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the CSVLoader.

        Enforces mutually exclusive arguments: exactly one of 'file_path' or 'dataset'
        must be provided.

        Parameters
        ----------
        file_path : str, optional
            The absolute or relative path to the CSV file.
        dataset : str, optional
            The name of the file inside the project's 'data' directory (e.g., 'titanic.csv').
        read_csv_kwargs : dict, optional
            Dictionary of arguments passed directly to `pandas.read_csv`.
        """
        super().__init__(target_col=target_col)

        # 1. Validation: Mutually Exclusive Arguments (XOR logic)
        if (file_path is None) == (dataset is None):
            raise ValueError(
                "Configuration Error: You must provide exactly one of 'file_path' OR 'dataset'. "
                f"Received: file_path={file_path}, dataset={dataset}"
            )

        # 2. Path Resolution
        if file_path:
            # Case A: Explicit path
            self.path = Path(file_path).expanduser().resolve()
            self.log_context = f"path='{self.path}'"

        else:
            # Case B: 'Shortcut' using project data root
            # Note: Using __file__ can be brittle if packaged/installed differently.
            # Ideally, passing absolute paths via Hydra's ${hydra:runtime.cwd} is preferred.
            project_root = Path(__file__).resolve().parent.parent.parent.parent
            data_root = project_root / "data"

            if not data_root.exists():
                raise FileNotFoundError(
                    f"Project data root not found at expected location: {data_root}"
                )

            # dataset is not None here due to check #1
            assert dataset is not None
            self.path = (data_root / dataset).resolve()
            self.log_context = f"dataset='{dataset}'"

        # 3. Kwargs handling
        self.read_csv_kwargs = read_csv_kwargs or {}

        self.log.info(f"CSVLoader initialized ({self.log_context})")

    def load(self) -> pd.DataFrame:
        """
        Load the dataset from the CSV file.

        Returns
        -------
        pd.DataFrame
            The loaded data.
        """
        if not self.path.exists():
            self.log.error(f"File not found: {self.path}")
            raise FileNotFoundError(f"CSV file missing at: {self.path}")

        self.log.info(f"Loading CSV from {self.path.name}...")

        try:
            # We unpack the dictionary (**dict) here, not in __init__
            df = pd.read_csv(self.path, **self.read_csv_kwargs)

            self.log.info(
                f"Loaded {len(df)} rows, {len(df.columns)} cols from {self.path.name}"
            )
            return df

        except Exception as e:
            self.log.error(f"Failed to parse CSV: {e}")
            raise

    def get_source_info(self) -> dict[str, Any]:
        return {
            "source_type": "csv_file",
            "path": str(self.path),
            "read_kwargs": self.read_csv_kwargs,
        }
