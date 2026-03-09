from learn2clean.actions import PanderaSchemaValidator
from tests.utils.actions.assert_action_pipeline import assert_action_pipeline


def test_pandera_checker_valid_data():
    """Test that valid data passes through unchanged."""
    assert_action_pipeline(
        PanderaSchemaValidator(
            schema_config={
                "schema_config": {
                    "columns": {
                        "age": {
                            "dtype": "int",
                            "checks": [{"greater_than_or_equal_to": 18}],
                        }
                    }
                }
            }
        ),
        {"age": [25, 30, 40], "email": ["a@b.com", "c@d.com", "e@f.com"]},
        {"age": [25, 30, 40], "email": ["a@b.com", "c@d.com", "e@f.com"]},
    )


def test_pandera_checker_removes_inconsistent_rows():
    """Test that valid data passes through unchanged."""
    assert_action_pipeline(
        PanderaSchemaValidator(
            schema_config={
                "columns": {
                    "age": {
                        "dtype": "int",
                        "checks": [{"greater_than_or_equal_to": 18}],
                    }
                }
            }
        ),
        {"age": [25, 10, 40], "email": ["a@b.com", "c@d.com", "e@f.com"]},
        {"age": [25, 40], "email": ["a@b.com", "e@f.com"]},
    )


def test_pandera_checker_bad_config_logging(caplog):
    """Test resilience against bad configuration."""
    assert_action_pipeline(
        PanderaSchemaValidator(
            schema_config={
                "columns": {"val": {"checks": [{"non_existent_check": 100}]}}
            }
        ),
        {"val": [1, 2, 3]},
        {"val": [1, 2, 3]},
    )
    assert "Unknown Pandera check" in caplog.text


def test_pandera_cc_pc_removes_inconsistent_rows():
    """
    Test PanderaConsistencyChecking using both constraint (CC) and pattern (PC) checks.
    """
    assert_action_pipeline(
        PanderaSchemaValidator(
            schema_config={
                "columns": {
                    # CC: Age must be >= 18
                    "Age": {
                        "dtype": "int64",
                        "checks": [{"greater_than_or_equal_to": 18}],
                    },
                    # PC: Code must be exactly 4 characters
                    "Code": {
                        "dtype": "object",
                        "checks": [{"regex": r"^[A-Z]\d{3}$"}],  # Ex: A123, B456
                    },
                    # Constraint simple
                    "Status": {"dtype": "object", "checks": [{"isin": ["Ok", "Fail"]}]},
                }
            }
        ),
        {
            "ID": [10, 11, 12, 13, 14],
            "Age": [25, 17, 30, 5, 50],  # Inconsistent CC: Rows 11 (17) and 13 (5) < 18
            "Code": [
                "A123",
                "B456",
                "C789",
                "D000",
                "E99",
            ],  # Inconsistent PC: Row 14 ('E99') != 4 chars
            "Status": ["Ok", "Ok", "Fail", "Ok", "Ok"],
        },
        {
            "ID": [10, 12],
            "Age": [25, 30],
            "Code": [
                "A123",
                "C789",
            ],
            "Status": ["Ok", "Fail"],
        },
    )


def test_pandera_cc_removal():
    """Test that the action identifies errors but doesn't remove rows if removal is disabled."""
    assert_action_pipeline(
        PanderaSchemaValidator(
            schema_config={
                "columns": {
                    "Age": {
                        "dtype": "int64",
                        "checks": [{"greater_than_or_equal_to": 18}],
                    }
                }
            },
            remove_inconsistent_rows=True,
        ),
        {"Age": [20, 30]},
        {"Age": [20, 30]},
    )


def test_pandera_cc_no_removal():
    """Test that the action identifies errors but doesn't remove rows if removal is disabled."""
    assert_action_pipeline(
        PanderaSchemaValidator(
            schema_config={
                "columns": {
                    "Age": {
                        "dtype": "int64",
                        "checks": [{"greater_than_or_equal_to": 18}],
                    }
                }
            },
            remove_inconsistent_rows=False,
        ),
        {"Age": [20, 15, 30]},
        {"Age": [20, 15, 30]},
    )
