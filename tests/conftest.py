"""Pytest configuration and shared fixtures for semantic_frame tests.

This module provides common fixtures used across multiple test files.
Fixtures are organized by category for easier discovery.
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest

# =============================================================================
# Random Seed Fixtures
# =============================================================================


@pytest.fixture
def fixed_seed():
    """Set a fixed random seed for reproducible tests."""
    np.random.seed(42)
    yield 42


# =============================================================================
# Sample Data Fixtures - NumPy Arrays
# =============================================================================


@pytest.fixture
def constant_data():
    """Constant data (all same values) - useful for edge case testing."""
    return np.array([5.0] * 20)


@pytest.fixture
def linear_rising_data():
    """Linearly rising data for trend testing."""
    return np.linspace(0, 100, 50)


@pytest.fixture
def linear_falling_data():
    """Linearly falling data for trend testing."""
    return np.linspace(100, 0, 50)


@pytest.fixture
def normal_data(fixed_seed):
    """Normal distribution data."""
    return np.random.normal(100, 10, 100)


@pytest.fixture
def normal_data_with_outlier(normal_data):
    """Normal data with a clear outlier."""
    data = normal_data.copy()
    data[50] = 300  # Clear outlier
    return data


@pytest.fixture
def seasonal_data():
    """Data with clear seasonal/cyclic pattern."""
    x = np.linspace(0, 4 * np.pi, 100)
    return 50 + 20 * np.sin(x)


@pytest.fixture
def step_change_data():
    """Data with a clear step change in the middle."""
    before = np.full(30, 100.0)
    after = np.full(30, 150.0)
    return np.concatenate([before, after])


@pytest.fixture
def high_volatility_data(fixed_seed):
    """Data with high variance/volatility."""
    return np.random.uniform(0, 100, 50)


@pytest.fixture
def low_volatility_data(fixed_seed):
    """Data with low variance/volatility."""
    return np.random.normal(100, 1, 50)


@pytest.fixture
def data_with_nans():
    """Data containing NaN values (20% missing)."""
    return np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, np.nan, 8.0, 9.0, 10.0])


@pytest.fixture
def data_with_inf():
    """Data containing Inf values."""
    return np.array([1.0, 2.0, np.inf, 4.0, 5.0])


# =============================================================================
# Sample Data Fixtures - Pandas
# =============================================================================


@pytest.fixture
def pandas_series(normal_data):
    """Normal data as pandas Series."""
    return pd.Series(normal_data, name="values")


@pytest.fixture
def pandas_dataframe_numeric():
    """Simple pandas DataFrame with numeric columns."""
    return pd.DataFrame(
        {
            "col_a": [1, 2, 3, 4, 5],
            "col_b": [5, 4, 3, 2, 1],
            "col_c": [10, 10, 10, 10, 10],
        }
    )


@pytest.fixture
def pandas_dataframe_mixed():
    """Pandas DataFrame with mixed column types."""
    return pd.DataFrame(
        {
            "numeric": [1, 2, 3, 4, 5],
            "string": ["a", "b", "c", "d", "e"],
            "datetime": pd.date_range("2024-01-01", periods=5),
            "float": [1.1, 2.2, 3.3, 4.4, 5.5],
        }
    )


@pytest.fixture
def pandas_dataframe_correlated():
    """DataFrame with correlated columns."""
    return pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "y_positive": [2, 4, 6, 8, 10],  # Perfect positive correlation
            "y_negative": [5, 4, 3, 2, 1],  # Perfect negative correlation
        }
    )


# =============================================================================
# Sample Data Fixtures - Polars
# =============================================================================


@pytest.fixture
def polars_series(normal_data):
    """Normal data as polars Series."""
    return pl.Series("values", normal_data)


@pytest.fixture
def polars_dataframe_numeric():
    """Simple polars DataFrame with numeric columns."""
    return pl.DataFrame(
        {
            "col_a": [1, 2, 3, 4, 5],
            "col_b": [5, 4, 3, 2, 1],
            "col_c": [10, 10, 10, 10, 10],
        }
    )


@pytest.fixture
def polars_series_with_nulls():
    """Polars Series with null values."""
    return pl.Series("data", [1.0, 2.0, None, 4.0, 5.0])


# =============================================================================
# Edge Case Data Fixtures
# =============================================================================


@pytest.fixture
def empty_array():
    """Empty numpy array."""
    return np.array([])


@pytest.fixture
def single_value():
    """Single value array."""
    return np.array([42.0])


@pytest.fixture
def two_values():
    """Two value array (minimum for trends)."""
    return np.array([1.0, 2.0])


@pytest.fixture
def all_nan_data():
    """Array of all NaN values."""
    return np.array([np.nan, np.nan, np.nan])


@pytest.fixture
def all_inf_data():
    """Array of all Inf values."""
    return np.array([np.inf, np.inf, -np.inf])


@pytest.fixture
def extreme_values():
    """Array with extreme numerical values."""
    return np.array([1e-10, 1e10, 1e-10, 1e10])


# =============================================================================
# Parametrized Test Data
# =============================================================================


@pytest.fixture(params=["pandas", "numpy", "polars", "list"])
def data_type(request, normal_data):
    """Parametrized fixture providing same data in different types."""
    if request.param == "pandas":
        return pd.Series(normal_data)
    elif request.param == "numpy":
        return normal_data
    elif request.param == "polars":
        return pl.Series("values", normal_data)
    else:  # list
        return normal_data.tolist()


@pytest.fixture(params=["text", "json", "full"])
def output_format(request):
    """Parametrized fixture for output formats."""
    return request.param


# =============================================================================
# Result Type Fixtures
# =============================================================================


@pytest.fixture
def sample_anomaly_info():
    """Sample AnomalyInfo for testing."""
    from semantic_frame.interfaces.json_schema import AnomalyInfo

    return AnomalyInfo(index=5, value=100.0, z_score=3.5)


@pytest.fixture
def sample_series_profile():
    """Sample SeriesProfile for testing."""
    from semantic_frame.interfaces.json_schema import SeriesProfile

    return SeriesProfile(
        count=100,
        mean=50.0,
        median=49.5,
        std=10.0,
        min_val=20.0,
        max_val=80.0,
        missing_pct=5.0,
    )


# =============================================================================
# Mock Fixtures
# =============================================================================


@pytest.fixture
def mock_api_key():
    """Mock API key for testing."""
    return "test-api-key-12345"


# =============================================================================
# Pytest Configuration
# =============================================================================


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "benchmark: marks benchmark tests")
    config.addinivalue_line(
        "markers", "integration: marks integration tests requiring external services"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically add markers based on test location."""
    for item in items:
        # Mark benchmark tests
        if "benchmark" in str(item.fspath):
            item.add_marker(pytest.mark.benchmark)
