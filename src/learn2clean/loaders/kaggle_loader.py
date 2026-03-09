from pathlib import Path
from typing import Any

import pandas as pd

from learn2clean.loaders.base import DatasetLoader


class KaggleLoader(DatasetLoader):
    """
    Concrete implementation of DatasetLoader for fetching datasets from Kaggle.

    This loader handles authentication via the Kaggle API, downloads the specified
    dataset to a local directory, extracts it, and loads a specific CSV file
    into a pandas DataFrame.

    Prerequisites:
        - A valid 'kaggle.json' in ~/.kaggle/ or KAGGLE_USERNAME/KAGGLE_KEY env vars.
    """

    def __init__(
        self,
        dataset_id: str,
        download_dir: str,
        filename: str,
        target_col: str | None = None,
        read_csv_kwargs: dict[str, Any] | None = None,
    ):
        """
        Initialize the KaggleLoader.

        Parameters
        ----------
        dataset_id : str
            The Kaggle dataset identifier (e.g., 'zillow/zecon').
        download_dir : str
            The local directory path where the dataset will be downloaded and extracted.
        filename : str
            The specific name of the CSV file to load from the downloaded dataset.
        read_csv_kwargs : dict, optional
            Additional keyword arguments passed directly to `pandas.read_csv`
            (e.g., sep, encoding, header).
        """
        super().__init__(target_col=target_col)

        self.dataset_id = dataset_id
        # Resolve path immediately to handle relative paths or user expansion (~)
        self.download_dir = Path(download_dir).expanduser().resolve()
        self.filename = filename
        self.read_csv_kwargs = read_csv_kwargs or {}

        self.log.info(
            f"KaggleLoader initialized. Dataset: '{self.dataset_id}', "
            f"Target File: '{self.filename}', Download Dir: '{self.download_dir}'"
        )

    def load(self) -> pd.DataFrame:
        """
        Downloads the dataset from Kaggle, extracts it, and loads the target CSV file.

        Returns
        -------
        pd.DataFrame
            The loaded data.

        Raises
        ------
        RuntimeError
            If Kaggle authentication or download fails.
        FileNotFoundError
            If the target file is not found after download.
        TypeError
            If the loaded object is not a DataFrame.
        """
        from kaggle.api.kaggle_api_extended import KaggleApi

        # 1. Ensure the download directory exists
        if not self.download_dir.exists():
            self.log.info(f"Creating download directory: {self.download_dir}")
            self.download_dir.mkdir(parents=True, exist_ok=True)

        # 2. Authenticate
        try:
            api = KaggleApi()
            api.authenticate()
        except Exception as e:
            self.log.error(f"Kaggle API Authentication failed: {e}")
            raise RuntimeError(
                "Failed to authenticate with Kaggle. Ensure ~/.kaggle/kaggle.json exists "
                "or environment variables are set."
            ) from e

        # 3. Download
        self.log.info(
            f"Downloading dataset '{self.dataset_id}' to '{self.download_dir}'..."
        )
        try:
            download_path_str = str(self.download_dir)
            # unzip=True handles extraction automatically.
            # quiet=False allows seeing progress in logs/stdout if configured.
            api.dataset_download_files(
                dataset=self.dataset_id, path=download_path_str, unzip=True, quiet=False
            )
            self.log.info("Download and extraction complete.")
        except Exception as e:
            self.log.error(f"Kaggle API error during download: {e}")
            raise RuntimeError(
                f"Failed to download dataset '{self.dataset_id}' from Kaggle."
            ) from e

        # 4. Locate File
        final_file_path = self.download_dir / self.filename

        if not final_file_path.exists():
            self.log.error(
                f"Required file '{self.filename}' not found at: {final_file_path}."
            )
            raise FileNotFoundError(
                f"File '{self.filename}' missing after downloading '{self.dataset_id}'. "
                f"Check if the filename is correct."
            )

        # 5. Load CSV
        self.log.info(f"Loading CSV file: {final_file_path.name}")
        try:
            df = pd.read_csv(final_file_path, **self.read_csv_kwargs)

            # Ensure strict return type (read_csv can return TextFileReader if chunksize is set)
            if not isinstance(df, pd.DataFrame):
                raise TypeError(
                    f"Expected pandas DataFrame, got {type(df)}. "
                    "Ensure 'chunksize' is not set in read_csv_kwargs."
                )

            self.log.info(
                f"Successfully loaded {len(df)} rows and {len(df.columns)} columns."
            )
            return df

        except Exception as e:
            self.log.error(f"Error reading CSV file '{self.filename}': {e}")
            # Re-raise nicely formatted
            raise RuntimeError(f"Failed to parse CSV file '{final_file_path}'") from e

    def get_source_info(self) -> dict[str, Any]:
        """
        Returns metadata about the Kaggle dataset source.
        """
        final_path = self.download_dir / self.filename
        return {
            "source_type": "kaggle_dataset",
            "dataset_id": self.dataset_id,
            "filename": self.filename,
            "download_dir": str(self.download_dir),
            "local_path": str(final_path),
            "read_csv_kwargs": self.read_csv_kwargs,
        }
