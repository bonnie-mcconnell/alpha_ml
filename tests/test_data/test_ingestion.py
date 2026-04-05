"""
Tests for the data ingestion module.
"""
import pytest
import pandas as pd

from ml_alpha.data.ingestion import load_data


def test_load_data_valid():
    """Test loading data with valid parameters."""
    ticker = "SPY"
    start_date = "2020-01-01"
    data = load_data(ticker, start_date)
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert isinstance(data.index, pd.DatetimeIndex)


def test_correct_columns():
    """Test that the loaded data has the correct columns."""
    ticker = "SPY"
    start_date = "2020-01-01"
    data = load_data(ticker, start_date)
    assert set(data.columns) == {"close", "volume"}
    assert data.index.min() >= pd.to_datetime(start_date)
    assert data["close"].notnull().all()
    assert (data["close"] > 0).all()
    assert data["volume"].notnull().all()


def test_load_data_invalid_ticker():
    """Test loading data with an invalid ticker."""
    ticker = "INVALID_TICKER"
    start_date = "2020-01-01"
    with pytest.raises(ValueError):
        load_data(ticker, start_date)


def test_load_data_invalid_start_date():
    """Test loading data with an invalid start date."""
    ticker = "SPY"
    start_date = "2000-301-100"
    with pytest.raises(ValueError):
        load_data(ticker, start_date)


def test_load_data_empty_data():
    """Test loading data for a future date range that results in no data."""
    ticker = "SPY"
    with pytest.raises(ValueError):
        load_data(ticker, "2099-01-01")