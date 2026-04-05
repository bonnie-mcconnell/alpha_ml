"""
Data ingestion module for loading the data for this project.
"""
import pandas as pd
import yfinance as yf


def load_data(ticker: str, start_date: str) -> pd.DataFrame:
    """
    Load the data from the source and return it as a DataFrame.

    Args:
        ticker (str): The ticker symbol to load data for.
        start_date (str): The start date for the data in YYYY-MM-DD format.


    Raises:        
        ValueError: If no data is found for the given ticker and start date.

    Returns:
        pd.DataFrame: A DataFrame containing the close price (adjusted close) and volume, indexed by date.
    """
    end_date = pd.Timestamp.today()
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)
    if data is None or data.empty:
        raise ValueError(f"No data found for ticker {ticker} starting from {start_date}.")
    data.columns = data.columns.get_level_values(0)
    data = data[["Adj Close", "Volume"]].rename(columns={"Adj Close": "close", "Volume": "volume"})
    return data
