"""
Tests for the data validator function
"""
import pandas as pd
import pytest

from ml_alpha.data.validator import validate_data
from ml_alpha.config import MIN_FEATURE_ROWS


@pytest.fixture
def valid_data():
    """Fixture for a valid DataFrame with the required structure and content."""
    dates = pd.date_range(start="2020-01-01", periods=100, freq="B")  # 100 business days
    return pd.DataFrame({
        "close": [100.0] * 100,
        "volume": [1_000_000] * 100,
    }, index=dates)


def test_validate_data_valid(valid_data):
    """Test that valid data passes validation."""
    validate_data(valid_data)


def test_validate_data_empty():
    """Test that empty DataFrame fails validation."""
    with pytest.raises(ValueError, match=r"Data must not be empty\."):
        validate_data(pd.DataFrame())

def test_validate_data_non_datetime_index(valid_data):
    """Test that non-DatetimeIndex fails validation."""
    df = valid_data.copy()
    df.index = range(len(df))  # Change index to integer
    with pytest.raises(ValueError, match=r"Data index must be a DatetimeIndex\."):
        validate_data(df)

def test_validate_data_duplicate_index(valid_data):
    """Test that duplicate timestamps in index fail validation."""
    df = valid_data.copy()
    df = pd.concat([df, df.iloc[:10]])  # Add duplicate rows
    with pytest.raises(ValueError, match=r"Data index contains duplicate timestamps\."):
        validate_data(df)

def test_validate_data_non_monotonic_index(valid_data):
    """Test that non-monotonic index fails validation."""
    df = valid_data.copy()
    df = df.iloc[::-1]  # Reverse the order to make it non-monotonic
    with pytest.raises(ValueError, match=r"Data index must be monotonically increasing"):
        validate_data(df)

def test_validate_data_missing_close(valid_data):
    """Test that missing values in 'close' column fail validation."""
    df = valid_data.copy()
    df.loc[df.index[0], "close"] = None
    with pytest.raises(ValueError, match=r"'close' column contains missing values\."):
        validate_data(df)

def test_validate_data_negative_close(valid_data):
    """Test that zero or negative values in 'close' column fail validation."""
    df = valid_data.copy()
    df.loc[df.index[0], "close"] = -100.0  # Introduce negative value in 'close'
    with pytest.raises(ValueError, match=r"'close' column contains zero or negative values\."):
        validate_data(df)

def test_validate_data_zero_close(valid_data):
    """Test that zero values in 'close' column fail validation."""
    df = valid_data.copy()
    df.loc[df.index[0], "close"] = 0.0  # Introduce zero value in 'close'
    with pytest.raises(ValueError, match=r"'close' column contains zero or negative values\."):
        validate_data(df)

def test_validate_data_missing_volume(valid_data):
    """Test that missing values in 'volume' column fail validation."""
    df = valid_data.copy()
    df.loc[df.index[0], "volume"] = None  # Introduce NaN in 'volume'
    with pytest.raises(ValueError, match=r"'volume' column contains missing values\."):
        validate_data(df)

def test_validate_data_negative_volume(valid_data):
    """Test that negative values in 'volume' column fail validation."""
    df = valid_data.copy()
    df.loc[df.index[0], "volume"] = -1_000_000  # Introduce negative value in 'volume'
    with pytest.raises(ValueError, match=r"'volume' column contains negative values\."):
        validate_data(df)

def test_validate_data_insufficient_rows(valid_data):
    """Test that having fewer than MIN_FEATURE_ROWS rows fails validation."""
    df = valid_data.head(MIN_FEATURE_ROWS - 1)  # Create DataFrame with too few rows
    with pytest.raises(ValueError, match=rf"Data must contain at least {MIN_FEATURE_ROWS} rows"):
        validate_data(df)

def test_validate_data_missing_close_column(valid_data):
    """Test that missing 'close' column fails validation."""
    df = valid_data.copy()
    df.drop(columns=["close"], inplace=True)  # Remove 'close' column
    with pytest.raises(ValueError, match=r"Data must contain 'close' column\."):
        validate_data(df)

def test_validate_data_missing_volume_column(valid_data):
    """Test that missing 'volume' column fails validation."""
    df = valid_data.copy()
    df.drop(columns=["volume"], inplace=True)  # Remove 'volume' column
    with pytest.raises(ValueError, match=r"Data must contain 'volume' column\."):
        validate_data(df)