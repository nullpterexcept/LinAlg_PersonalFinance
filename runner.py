"""
Runner script: end-to-end example using the data_collection and optimization modules.
"""
import datetime
import numpy as np

from data_collection import ASSETS, download_price_data, download_risk_free_rate
from optimization import (
    efficient_frontier,
    resampled_tangency_portfolio,
    solve_tangency_portfolio,
    solve_min_variance,
)
from utils import prices_to_log_returns, annualize

INFLATION_ASSUMPTION = 0.03
RISK_AVERSION = 3.0


def compute_merton_cal_allocation(tangency_weights, tangency_return, tangency_variance, rf=0.0, risk_aversion=3.0,
                                  allow_leverage=False):
    """
    Compute the Merton share using the capital allocation line.
    """
    if risk_aversion <= 0:
        raise ValueError("risk_aversion must be positive")
    if tangency_variance <= 0:
        raise ValueError("tangency_variance must be positive")

    tangency_weights = np.asarray(tangency_weights, dtype=float)
    risky_share = float((tangency_return - rf) / (risk_aversion * tangency_variance))
    risky_share = max(0.0, risky_share)
    if not allow_leverage:
        risky_share = min(1.0, risky_share)

    weights = tangency_weights * risky_share
    safe_share = 1.0 - risky_share

    return {
        "weights": weights,
        "safe_share": safe_share,
        "risky_share": risky_share,
        "risky_sleeve": tangency_weights,
        "tangency_return": float(tangency_return),
        "tangency_variance": float(tangency_variance),
    }


def main():
    # Step 1: download prices
    prices = download_price_data(ASSETS, end_date=datetime.date(2026, 5, 25))

    # compute log returns
    log_returns = prices_to_log_returns(prices)

    # annualize stats
    mu_daily = log_returns.mean()
    sigma_daily = log_returns.std()
    mu_a, sigma_a = annualize(mu_daily, sigma_daily)

    # convert mu_a and Sigma matrix
    mu = np.asarray(mu_a, dtype=float)
    Sigma = log_returns.cov().values * 252  # annualized covariance

    # Efficient frontier. Won't be used here but good for visualization if wanted (not implemented in this script).
    frontier = efficient_frontier(mu, Sigma, n_points=50)

    rf = download_risk_free_rate(end_date=datetime.date(2026, 5, 25))
    market = solve_tangency_portfolio(mu, Sigma, rf=rf, allow_short=False)

    # min variance
    minv = solve_min_variance(mu, Sigma, allow_short=False)

    resampled_market = resampled_tangency_portfolio(log_returns, rf=rf, n_boot=500, seed=7, allow_short=False)
    robust_weights = resampled_market['weights']
    robust_return = float(mu.dot(robust_weights))
    robust_variance = float(robust_weights.dot(Sigma).dot(robust_weights))
    merton_cal = compute_merton_cal_allocation(
        robust_weights,
        robust_return,
        robust_variance,
        rf=rf,
        risk_aversion=RISK_AVERSION,
        allow_leverage=False,
    )
    practical = merton_cal['weights']
    practical_cash = merton_cal['safe_share']

    parts = [f"{name}: {w:.4f}" for name, w in zip(prices.columns, practical)]
    parts.append(f"CASH/T-Bill: {practical_cash:.4f}")
    print("Final allocation including risk-free asset:\n" + "\n".join(parts))


if __name__ == '__main__':
    main()
