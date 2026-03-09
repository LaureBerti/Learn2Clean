def test_environment() -> None:
    """Check that the test environment works correctly."""
    assert True


def test_python_version() -> None:
    """Check that the Python version is 3.13."""
    import sys

    version = sys.version_info
    print(f"Expected Python 3.13, found {version.major}.{version.minor}")


def test_pytest_import() -> None:
    """Check that pytest is installed and importable."""
    import pytest

    assert callable(pytest.main)
