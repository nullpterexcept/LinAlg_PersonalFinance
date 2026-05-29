"""
Utility functions for data validation and simple finance math.
"""

import numpy as np
import pandas as pd


def regularized_inverse(Sigma, reg=0.0):
    """
    Robust inverse for covariance matrices. Adds small ridge if necessary and uses
    pseudo-inverse as fallback.
    """
    try:
        return np.linalg.inv(Sigma + reg * np.eye(Sigma.shape[0]))
    except np.linalg.LinAlgError:
        return np.linalg.pinv(Sigma + reg * np.eye(Sigma.shape[0]))


def validate_mu_sigma(mu, Sigma):
    """
    Ensure mu and Sigma have compatible shapes and no NaNs.

    Returns: (mu_array_or_None, Sigma_array)
    """
    Sigma = np.asarray(Sigma, dtype=float)
    if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError("Sigma must be a square (N,N) matrix")
    n = Sigma.shape[0]

    if mu is None:
        mu_arr = None
    else:
        mu_arr = np.asarray(mu, dtype=float)
        if mu_arr.shape[0] != n:
            raise ValueError("Length of mu must match Sigma dimensions")

    if np.isnan(Sigma).any():
        raise ValueError("Sigma contains NaN values")
    if mu_arr is not None and np.isnan(mu_arr).any():
        raise ValueError("mu contains NaN values")

    return mu_arr, Sigma


def prices_to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Convert price DataFrame to log returns, dropping NaNs.
    """
    lr = np.log(prices / prices.shift(1)).dropna()
    return lr


def annualize(mu_daily, sigma_daily, periods_per_year=252):
    """
    Annualize daily mean and std (both can be scalars or arrays)
    """
    mu_a = np.asarray(mu_daily) * periods_per_year
    sigma_a = np.asarray(sigma_daily) * np.sqrt(periods_per_year)
    return mu_a, sigma_a
