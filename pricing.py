"""Option pricers (Black-Scholes, binomial tree, Monte Carlo), Greeks, and implied volatility.

Conventions: S spot, K strike, T years to expiry, r risk-free rate, q dividend yield,
sigma volatility (all continuously compounded, annualized). kind is "call" or "put".
"""
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def _d1_d2(S, K, T, r, sigma, q):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)


def black_scholes(S, K, T, r, sigma, kind="call", q=0.0):
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    if kind == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def binomial(S, K, T, r, sigma, kind="call", q=0.0, steps=500, american=False):
    """Cox-Ross-Rubinstein tree."""
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    payoff = (lambda s: np.maximum(s - K, 0)) if kind == "call" else (lambda s: np.maximum(K - s, 0))

    spots = S * u ** np.arange(steps, -1, -1) * d ** np.arange(0, steps + 1)
    values = payoff(spots)
    for i in range(steps - 1, -1, -1):
        values = disc * (p * values[:-1] + (1 - p) * values[1:])
        if american:
            spots = spots[:-1] / u
            values = np.maximum(values, payoff(spots))
    return values[0]


def monte_carlo(S, K, T, r, sigma, kind="call", q=0.0, paths=200_000, seed=0):
    """GBM terminal-price simulation with antithetic variates. Returns (price, standard error)."""
    z = np.random.default_rng(seed).standard_normal(paths // 2)
    z = np.concatenate([z, -z])
    ST = S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    payoff = np.maximum(ST - K, 0) if kind == "call" else np.maximum(K - ST, 0)
    # Average each antithetic pair first so the standard error reflects the pairing.
    pairs = np.exp(-r * T) * 0.5 * (payoff[: paths // 2] + payoff[paths // 2 :])
    return pairs.mean(), pairs.std(ddof=1) / np.sqrt(len(pairs))


def greeks(S, K, T, r, sigma, kind="call", q=0.0):
    """Analytic Black-Scholes Greeks. Vega per 1.00 of vol, theta per year (dV/dt)."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    eq, er, pdf = np.exp(-q * T), np.exp(-r * T), norm.pdf(d1)
    gamma = eq * pdf / (S * sigma * np.sqrt(T))
    vega = S * eq * pdf * np.sqrt(T)
    decay = -S * eq * pdf * sigma / (2 * np.sqrt(T))
    if kind == "call":
        delta = eq * norm.cdf(d1)
        theta = decay - r * K * er * norm.cdf(d2) + q * S * eq * norm.cdf(d1)
    else:
        delta = eq * (norm.cdf(d1) - 1)
        theta = decay + r * K * er * norm.cdf(-d2) - q * S * eq * norm.cdf(-d1)
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}


def greeks_fd(S, K, T, r, sigma, kind="call", q=0.0, h=1e-4):
    """Same Greeks by central finite differences on the Black-Scholes price."""
    f = lambda **kw: black_scholes(**{"S": S, "K": K, "T": T, "r": r, "sigma": sigma, "kind": kind, "q": q, **kw})
    dS = S * h
    return {
        "delta": (f(S=S + dS) - f(S=S - dS)) / (2 * dS),
        "gamma": (f(S=S + dS) - 2 * f() + f(S=S - dS)) / dS**2,
        "vega": (f(sigma=sigma + h) - f(sigma=sigma - h)) / (2 * h),
        "theta": -(f(T=T + h) - f(T=T - h)) / (2 * h),  # value change as time passes = -dV/dT
    }


def implied_vol(price, S, K, T, r, kind="call", q=0.0):
    """Volatility that reproduces `price` under Black-Scholes, or nan if price breaks no-arbitrage bounds."""
    fwd_s, fwd_k = S * np.exp(-q * T), K * np.exp(-r * T)
    lower = max(fwd_s - fwd_k, 0) if kind == "call" else max(fwd_k - fwd_s, 0)
    upper = fwd_s if kind == "call" else fwd_k
    if not lower < price < upper:
        return float("nan")
    return brentq(lambda v: black_scholes(S, K, T, r, v, kind, q) - price, 1e-6, 5.0, xtol=1e-10)
