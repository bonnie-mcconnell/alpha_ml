"""
Data validation functions for ingested data.
Doesn't transform data, only checks for required structure and content.
"""
import pandas as pd

from ml_alpha.config import MIN_FEATURE_ROWS


def validate_data(data: pd.DataFrame) -> None:
    """
    Validate the input data for backtesting.
    
    Checks:
    - DataFrame is not empty.
    - Index is a DatetimeIndex with no duplicates and is monotonically increasing.
    - Contains 'close' column.
    - Contains 'volume' column.
    - 'close' column has no missing values, is not empty, and contains no negative values.
    - 'volume' column has no missing values and contains no negative values.
    - DataFrame has at least MIN_FEATURE_ROWS rows to allow for feature construction after dropping NaNs.

    
    Args:
        data: DataFrame to validate.
    Raises:
        ValueError: If any validation check fails.
    """
    if data.empty:
        raise ValueError("Data must not be empty.")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data index must be a DatetimeIndex.")
    if data.index.has_duplicates:
        raise ValueError("Data index contains duplicate timestamps.")
    if not data.index.is_monotonic_increasing:
        raise ValueError("Data index must be monotonically increasing (no out-of-order dates).")
    if 'close' not in data.columns:
        raise ValueError("Data must contain 'close' column.")
    if 'volume' not in data.columns:
        raise ValueError("Data must contain 'volume' column.")
    if data['close'].isnull().any():
        raise ValueError("'close' column contains missing values.")
    if data['volume'].isnull().any():
        raise ValueError("'volume' column contains missing values.")
    if (data['close'] <= 0).any():
        raise ValueError("'close' column contains zero or negative values.")
    if (data['volume'] < 0).any():
        raise ValueError("'volume' column contains negative values.")
    if len(data) < MIN_FEATURE_ROWS:
        raise ValueError(f"Data must contain at least {MIN_FEATURE_ROWS} rows to construct features.")
    
