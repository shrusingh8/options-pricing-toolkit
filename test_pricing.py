"""Sanity checks. Run: python -m pytest -q"""
import numpy as np
import pytest

from pricing import black_scholes, binomial, greeks, greeks_fd, implied_vol, monte_carlo

CASES = [  # S, K, T, r, sigma, q
    (100, 100, 1.0, 0.05, 0.20, 0.00),
    (100, 120, 0.25, 0.03, 0.35, 0.01),
    (100, 80, 2.0, 0.01, 0.15, 0.02),
]
KINDS = ["call", "put"]


@pytest.mark.parametrize("S,K,T,r,sigma,q", CASES)
@pytest.mark.parametrize("kind", KINDS)
def test_three_pricers_agree(S, K, T, r, sigma, q, kind):
    bs = black_scholes(S, K, T, r, sigma, kind, q)
    assert binomial(S, K, T, r, sigma, kind, q, steps=2000) == pytest.approx(bs, abs=0.01)
    mc, se = monte_carlo(S, K, T, r, sigma, kind, q)
    assert abs(mc - bs) < 4 * se


@pytest.mark.parametrize("S,K,T,r,sigma,q", CASES)
def test_put_call_parity(S, K, T, r, sigma, q):
    c, p = black_scholes(S, K, T, r, sigma, "call", q), black_scholes(S, K, T, r, sigma, "put", q)
    assert c - p == pytest.approx(S * np.exp(-q * T) - K * np.exp(-r * T), abs=1e-10)


@pytest.mark.parametrize("S,K,T,r,sigma,q", CASES)
def test_no_arbitrage_bounds(S, K, T, r, sigma, q):
    fs, fk = S * np.exp(-q * T), K * np.exp(-r * T)
    c, p = black_scholes(S, K, T, r, sigma, "call", q), black_scholes(S, K, T, r, sigma, "put", q)
    assert max(fs - fk, 0) <= c <= fs
    assert max(fk - fs, 0) <= p <= fk
    # American put is worth at least the European one and at least immediate exercise.
    am = binomial(S, K, T, r, sigma, "put", q, steps=1000, american=True)
    assert am >= p - 1e-3 and am >= K - S


def test_binomial_converges_to_black_scholes():
    S, K, T, r, sigma = 100, 105, 1.0, 0.05, 0.2
    bs = black_scholes(S, K, T, r, sigma)
    errors = [abs(binomial(S, K, T, r, sigma, steps=n) - bs) for n in (25, 100, 400, 1600)]
    assert errors[-1] < 0.005
    # CRR error shrinks roughly like 1/n (it oscillates, so compare the ends, not every step).
    assert errors[-1] < errors[0] / 10


@pytest.mark.parametrize("S,K,T,r,sigma,q", CASES)
@pytest.mark.parametrize("kind", KINDS)
def test_greeks_match_finite_differences(S, K, T, r, sigma, q, kind):
    a, n = greeks(S, K, T, r, sigma, kind, q), greeks_fd(S, K, T, r, sigma, kind, q)
    for g in a:
        assert a[g] == pytest.approx(n[g], rel=1e-4, abs=1e-6), g


@pytest.mark.parametrize("S,K,T,r,sigma,q", CASES)
@pytest.mark.parametrize("kind", KINDS)
def test_implied_vol_round_trip(S, K, T, r, sigma, q, kind):
    price = black_scholes(S, K, T, r, sigma, kind, q)
    assert implied_vol(price, S, K, T, r, kind, q) == pytest.approx(sigma, abs=1e-7)


def test_implied_vol_rejects_arbitrage_prices():
    assert np.isnan(implied_vol(150, 100, 100, 1, 0.05))  # call worth more than the stock
