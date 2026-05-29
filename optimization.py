"""
Portfolio optimization routines: minimum-variance, tangency (max Sharpe), and efficient frontier.
"""

import numpy as np
from scipy.optimize import minimize

from utils import regularized_inverse, validate_mu_sigma


def _volatility(weights, Sigma):
    return float(np.sqrt(max(0.0, weights.dot(Sigma).dot(weights))))


def _full_investment_constraint():
    return {'type': 'eq', 'fun': lambda w: float(np.sum(w) - 1.0)}


def solve_min_variance(mu, Sigma, bounds=None, allow_short=True, reg=1e-8, tol=1e-10):
    """
    Solve the minimum-variance portfolio with sum(weights)=1.

    Returns dict with keys: weights, volatility, return (if mu provided), success, message
    """
    mu, Sigma = validate_mu_sigma(mu, Sigma)
    n = Sigma.shape[0]
    Sigma_reg = Sigma + reg * np.eye(n)

    ones = np.ones(n)

    # Unconstrained closed-form solution
    if bounds is None and allow_short:
        inv = regularized_inverse(Sigma_reg)
        w = inv.dot(ones) / (ones.dot(inv).dot(ones))
        vol = _volatility(w, Sigma)
        ret = float(mu.dot(w)) if mu is not None else None
        return {"weights": w, "volatility": vol, "return": ret, "success": True, "message": "closed-form"}

    # Otherwise use SLSQP
    if bounds is None:
        if allow_short:
            bnds = [(-1.0, 1.0)] * n
        else:
            bnds = [(0.0, 1.0)] * n
    else:
        if isinstance(bounds, tuple) and len(bounds) == 2:
            bnds = [bounds] * n
        else:
            bnds = list(bounds)

    def obj(w):
        return float(w.dot(Sigma_reg).dot(w))

    cons = [_full_investment_constraint()]
    x0 = np.ones(n) / n

    res = minimize(obj, x0, method='SLSQP', bounds=bnds, constraints=cons, tol=tol)

    w = res.x
    vol = _volatility(w, Sigma)
    ret = float(mu.dot(w)) if mu is not None else None
    return {"weights": w, "volatility": vol, "return": ret, "success": res.success, "message": res.message}


def solve_tangency_portfolio(mu, Sigma, rf=0.0, bounds=None, allow_short=True, reg=1e-8, tol=1e-10):
    """
    Compute the tangency (maximum Sharpe) portfolio under full investment (sum(weights)=1).

    Returns dict with weights, volatility, return, sharpe, success, message
    """
    mu, Sigma = validate_mu_sigma(mu, Sigma)
    n = Sigma.shape[0]
    Sigma_reg = Sigma + reg * np.eye(n)

    excess = mu - rf

    ones = np.ones(n)

    # Closed-form when unconstrained
    if bounds is None and allow_short:
        inv = regularized_inverse(Sigma_reg)
        top = inv.dot(excess)
        denom = ones.dot(top)
        if abs(denom) < 1e-12:
            denom = np.sign(denom) * 1e-12 if denom != 0 else 1e-12
        w = top / denom
        vol = _volatility(w, Sigma)
        ret = float(mu.dot(w))
        sharpe = float((ret - rf) / vol) if vol > 0 else np.nan
        return {"weights": w, "volatility": vol, "return": ret, "sharpe": sharpe, "success": True, "message": "closed-form"}

    # Otherwise maximize Sharpe via SLSQP (minimize negative Sharpe)
    if bounds is None:
        if allow_short:
            bnds = [(-1.0, 1.0)] * n
        else:
            bnds = [(0.0, 1.0)] * n
    else:
        if isinstance(bounds, tuple) and len(bounds) == 2:
            bnds = [bounds] * n
        else:
            bnds = list(bounds)

    def neg_sharpe(w):
        ret = float(mu.dot(w))
        vol = float(np.sqrt(max(0.0, w.dot(Sigma_reg).dot(w))))
        # penalize zero vol
        if vol <= 0:
            return 1e6
        return -(ret - rf) / vol

    cons = [_full_investment_constraint()]
    x0 = np.ones(n) / n

    res = minimize(neg_sharpe, x0, method='SLSQP', bounds=bnds, constraints=cons, tol=tol)

    w = res.x
    vol = _volatility(w, Sigma)
    ret = float(mu.dot(w))
    sharpe = float((ret - rf) / vol) if vol > 0 else np.nan
    return {"weights": w, "volatility": vol, "return": ret, "sharpe": sharpe, "success": res.success, "message": res.message}


def efficient_frontier(mu, Sigma, n_points=50,reg=1e-8):
    """
    Compute an efficient frontier by solving min variance for target returns.

    Returns dict with arrays: returns, vols, weights (n_points x n)
    """
    mu, Sigma = validate_mu_sigma(mu, Sigma)
    n = Sigma.shape[0]

    min_ret = float(np.min(mu))
    max_ret = float(np.max(mu))
    if np.isclose(min_ret, max_ret):
        max_ret = min_ret + 1e-6

    target_returns = np.linspace(min_ret, max_ret, n_points)

    weights = np.zeros((n_points, n))
    vols = np.zeros(n_points)
    rets = np.zeros(n_points)

    Sigma_reg = Sigma + reg * np.eye(n)

    inv = regularized_inverse(Sigma_reg)
    ones = np.ones(n)

    g = inv.dot(mu)    # Sigma^{-1} mu
    h = inv.dot(ones)  # Sigma^{-1} 1

    A = float(ones.dot(g))
    B = float(mu.dot(g))
    C = float(ones.dot(h))
    D = float(B * C - A * A)
    if abs(D) < 1e-12:
        D = 1e-12 if D >= 0 else -1e-12

    for i, tr in enumerate(target_returns):
        lam = (C * tr - A) / D
        gam = (B - A * tr) / D
        w = lam * g + gam * h
        weights[i, :] = w
        rets[i] = float(mu.dot(w))
        vols[i] = _volatility(w, Sigma)

    return {"returns": rets, "vols": vols, "weights": weights}


def resampled_tangency_portfolio(log_returns, rf=0.0, n_boot=500, seed=None, allow_short=False):
    """
    Bootstrap the return history, solve a tangency portfolio in each resample,
    and average the weights.
    """
    rng = np.random.default_rng(seed)
    weights = []
    n_obs = len(log_returns)
    for _ in range(n_boot):
        sample = log_returns.iloc[rng.integers(0, n_obs, n_obs)]
        mu_b = sample.mean().values * 252
        Sigma_b = sample.cov().values * 252
        res = solve_tangency_portfolio(mu_b, Sigma_b, rf=rf, allow_short=allow_short)
        w = np.maximum(np.asarray(res["weights"], dtype=float), 0.0)
        if w.sum() > 0:
            weights.append(w / w.sum())

    if not weights:
        raise ValueError("No bootstrap tangency portfolios could be solved")

    weights = np.asarray(weights)
    avg = weights.mean(axis=0)
    avg = avg / avg.sum()
    return {
        "weights": avg,
        "bootstrap_weights": weights,
        "positive_frequency": (weights > 1e-4).mean(axis=0),
        "median_weights": np.median(weights, axis=0),
        "n_boot": int(len(weights)),
    }