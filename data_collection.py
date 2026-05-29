"""
Data collection and preprocessing for portfolio optimization analysis.

This module downloads historical price data from Yahoo Finance and prepares
it for portfolio analysis. We use index ETFs that are typical 401(k) offerings.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os

# Asset symbols representing Joe's portfolio components
ASSETS = {
    'VTI': 'Vanguard Total US Stock Market Index ETF',
    'VXUS': 'Vanguard Total International Stock Index ETF',
    'BND': 'Vanguard Total Bond Market Index ETF',
}

# Risk-free rate proxy (using 3-month Treasury Bill)
RISK_FREE_SYMBOL = '^IRX'  # 3-month Treasury Bill yield (annualized)


def _extract_price_field(downloaded, field='Adj Close'):
    """Return a DataFrame/Series for the requested price field across yfinance layouts."""
    if isinstance(downloaded, pd.DataFrame):
        if field in downloaded.columns:
            return downloaded[field]
        if isinstance(downloaded.columns, pd.MultiIndex):
            if field in downloaded.columns.get_level_values(0):
                return downloaded[field]
            if field in downloaded.columns.get_level_values(-1):
                return downloaded.xs(field, axis=1, level=-1)
        if 'Close' in downloaded.columns:
            return downloaded['Close']
    raise KeyError(field)


def download_price_data(symbols, end_date=None, years=5):
    """
    Download historical daily adjusted close prices from Yahoo Finance.

    Parameters:
    -----------
    symbols : list or dict
        List of ticker symbols or dict mapping symbols to descriptions
    end_date : str or datetime, optional
        End date for data collection (default: today)
    years : int
        Number of years of historical data to download (default: 5)

    Returns:
    --------
    pd.DataFrame
        DataFrame with dates as index and adjusted close prices as columns
    """
    if isinstance(symbols, dict):
        symbols = list(symbols.keys())

    if end_date is None:
        end_date = datetime.now()
    else:
        end_date = pd.to_datetime(end_date)

    start_date = end_date - timedelta(days=365 * years)

    print(f"Downloading {len(symbols)} assets from {start_date.date()} to {end_date.date()}...")

    # Download data
    raw = yf.download(symbols, start=start_date, end=end_date, progress=False, auto_adjust=False, group_by='column')
    data = _extract_price_field(raw, 'Adj Close')

    # Handle case where only one symbol was downloaded (returns Series instead of DataFrame)
    if isinstance(data, pd.Series):
        data = data.to_frame(name=symbols[0])

    # Remove any rows with NaN values
    data = data.dropna()

    if data.empty:
        raise ValueError("Downloaded price data is empty")

    print(f"Downloaded {len(data)} trading days of data")
    print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")

    return data


def download_risk_free_rate(end_date=None, years=5):
    """
    Download 3-month Treasury Bill yields from Yahoo Finance.
    Note: IRX provides annualized yields; we'll convert to daily returns.

    Parameters:
    -----------
    end_date : str or datetime, optional
        End date for data collection (default: today)
    years : int
        Number of years of data to download (default: 5)

    Returns:
    --------
    float
        Current annualized risk-free rate as decimal (e.g., 0.05 for 5%)
    """
    if end_date is None:
        end_date = datetime.now()
    else:
        end_date = pd.to_datetime(end_date)

    start_date = end_date - timedelta(days=365 * years)

    print(f"Downloading 3-month Treasury Bill rate...")
    raw = yf.download(RISK_FREE_SYMBOL, start=start_date, end=end_date, progress=False, auto_adjust=False)
    data = _extract_price_field(raw, 'Adj Close')

    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0]

    # Get the most recent rate
    current_rate = data.iloc[-1] / 100  # Convert percentage to decimal

    print(f"Current 3-month T-Bill rate: {current_rate * 100:.2f}%")

    return current_rate


def calculate_log_returns(prices):
    """
    Calculate log returns from price data.

    Log returns are used because they are additive across time periods
    and more appropriate for statistical analysis than simple returns.

    Parameters:
    -----------
    prices : pd.DataFrame
        DataFrame with asset prices

    Returns:
    --------
    pd.DataFrame
        DataFrame with log returns for each asset
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()
    return log_returns


def annualize_statistics(log_returns, periods_per_year=252):
    """
    Convert daily statistics to annual equivalents.

    Parameters:
    -----------
    log_returns : pd.DataFrame
        Daily log returns
    periods_per_year : int
        Trading days per year (default: 252)

    Returns:
    --------
    tuple
        (annual_returns, annual_volatility, correlation_matrix)
    """
    annual_returns = log_returns.mean() * periods_per_year
    annual_volatility = log_returns.std() * np.sqrt(periods_per_year)
    correlation_matrix = log_returns.corr()

    return annual_returns, annual_volatility, correlation_matrix


def save_data(prices, returns, output_dir='data'):
    """
    Save processed data to CSV files for reproducibility.
    """
    os.makedirs(output_dir, exist_ok=True)

    prices.to_csv(os.path.join(output_dir, 'prices.csv'))
    returns.to_csv(os.path.join(output_dir, 'log_returns.csv'))

    print(f"Data saved to {output_dir}/")


if __name__ == "__main__":
    # Download data
    prices = download_price_data(ASSETS)
    returns = calculate_log_returns(prices)

    # Calculate statistics
    annual_returns, annual_vol, corr = annualize_statistics(returns)

    # Download risk-free rate
    rf_rate = download_risk_free_rate()

    # Save data
    save_data(prices, returns)

    # Display summary
    print("\n" + "=" * 60)
    print("ANNUAL STATISTICS")
    print("=" * 60)
    print("\nExpected Annual Returns:")
    print(annual_returns)
    print("\nAnnual Volatility (Standard Deviation):")
    print(annual_vol)
    print("\nCorrelation Matrix:")
    print(corr)
    print(f"\nRisk-Free Rate: {rf_rate * 100:.2f}%")
