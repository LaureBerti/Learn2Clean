from typing import Any, ClassVar

import pandas as pd
from pandera import Check, Column, DataFrameSchema, errors

from ...data_frame_action import DataFrameAction

SchemaConfig = dict[str, Any]


class PanderaSchemaValidator(DataFrameAction):
    """
    Implements Inconsistency Detection (CC) and Pattern Checking (PC) using Pandera.

    Violating rows are identified and removed based on a configurable schema
    that defines column-level constraints and patterns.
    """

    DEFAULT_PARAMS: ClassVar[dict[str, Any]] = {
        "schema_config": {
            "columns": {
                # Example structure:
                # "Age": {"dtype": "int", "checks": [{"greater_than_or_equal_to": 18}]},
                # "PostalCode": {"dtype": "str", "checks": [{"str_matches": r"^\d{5}$"}]},
            }
        },
        "remove_inconsistent_rows": True,
    }

    def _convert_config_to_schema(self, config: SchemaConfig) -> DataFrameSchema:
        """
        Converts a dictionary configuration into a Pandera DataFrameSchema object.

        This method parses the simplified dictionary format to build complex
        Pandera checks dynamically.
        """
        schema_kwargs = {}
        column_configs = config.get("columns", {})

        pandera_columns = {}

        for col_name, col_config in column_configs.items():
            checks_list: list[Check] = []
            raw_checks = col_config.pop("checks", [])

            for check_def in raw_checks:
                # Iterate over checks (e.g., {"greater_than_or_equal_to": 18})
                for check_name, check_value in check_def.items():
                    try:
                        # Handle regex specifically as 'str_matches' in newer Pandera versions
                        # or allow direct mapping to Check attributes.
                        if check_name == "regex":
                            checks_list.append(Check.str_matches(check_value))
                        elif hasattr(Check, check_name):
                            check_method = getattr(Check, check_name)
                            checks_list.append(check_method(check_value))
                        else:
                            self.log_warning(
                                f"Unknown Pandera check '{check_name}' for column '{col_name}'."
                            )
                    except Exception as e:
                        self.log_error(
                            f"Error creating check '{check_name}' for '{col_name}': {e}"
                        )

            # Define the Column
            pandera_columns[col_name] = Column(
                dtype=col_config.get("dtype", None),
                checks=checks_list,
                nullable=col_config.get("nullable", False),
                required=col_config.get("required", True),
                coerce=True,  # Try to convert types if possible
            )

        schema_kwargs["columns"] = pandera_columns
        # Allow extra columns by default so we don't drop data not mentioned in schema
        return DataFrameSchema(**schema_kwargs, strict=False)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validates the DataFrame against the Pandera schema and removes inconsistent rows.

        Returns:
            pd.DataFrame: The cleaned DataFrame with inconsistent rows removed (if configured).
        """
        # Ensure we work on a copy to avoid side effects
        df_copy = df.copy()

        schema_config: SchemaConfig = self.params.get(
            "schema_config", self.DEFAULT_PARAMS["schema_config"]
        )

        if not schema_config.get("columns"):
            self.log_warning(
                "No column constraints provided in 'schema_config'. Skipping validation."
            )
            return df_copy

        try:
            schema = self._convert_config_to_schema(schema_config)
        except Exception as e:
            self.log_error(f"Failed to build Pandera schema: {e}")
            return df_copy

        self.log_info("Starting Pandera validation...")

        try:
            # lazy=True ensures we catch ALL errors, not just the first one
            schema.validate(df_copy, lazy=True, inplace=True)
            self.log_info("Validation successful: No inconsistencies found.")
            return df_copy

        except errors.SchemaErrors as e:
            failure_cases = e.failure_cases
            unique_failures = len(failure_cases)
            self.log_warning(f"Found {unique_failures} schema violations.")

            if not self.params.get("remove_inconsistent_rows"):
                self.log_info(
                    "remove_inconsistent_rows is False. Returning original data with warnings."
                )
                return df_copy

            # Identify rows to drop. failure_cases['index'] contains the row indices.
            # We must filter out non-index failures (like column missing errors)
            rows_to_drop = failure_cases[failure_cases["index"].notna()][
                "index"
            ].unique()

            num_dropped = len(rows_to_drop)
            df_cleaned = df_copy.drop(index=rows_to_drop, errors="ignore")

            self.log_info(
                f"Removed {num_dropped} inconsistent rows based on schema constraints."
            )
            return df_cleaned

        except Exception as e:
            self.log_error(f"Unexpected error during validation: {e}")
            return df_copy
