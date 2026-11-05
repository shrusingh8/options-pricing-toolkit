"""Back out implied vols from live SPY option quotes and plot the smile.

Run: python smile.py [TICKER] [TARGET_DAYS]   -> writes smile.png and smile_data.csv
"""
import sys
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from pricing import implied_vol

ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
target_days = int(sys.argv[2]) if len(sys.argv) > 2 else 30

tk = yf.Ticker(ticker)
spot = tk.history(period="5d")["Close"].iloc[-1]
try:  # 13-week T-bill yield as the risk-free rate
    rate = yf.Ticker("^IRX").history(period="5d")["Close"].iloc[-1] / 100
except Exception:
    rate = 0.04

today = date.today()
expiry = min(tk.options, key=lambda e: abs((date.fromisoformat(e) - today).days - target_days))
T = (date.fromisoformat(expiry) - today).days / 365
chain = tk.option_chain(expiry)

# The quoted spot is often stale relative to the option quotes, and the dividend yield is a guess.
# Put-call parity fixes both: C - P = S*e^(-qT) - K*e^(-rT), so the options themselves tell us the
# dividend-adjusted spot. Use the median over the strikes nearest the money; then price with q = 0.
both = chain.calls.merge(chain.puts, on="strike", suffixes=("_c", "_p"))
both = both[(both.bid_c > 0) & (both.bid_p > 0)]
both = both.iloc[(both.strike - spot).abs().argsort()[:6]]
mid_c, mid_p = (both.bid_c + both.ask_c) / 2, (both.bid_p + both.ask_p) / 2
quoted_spot, spot = spot, float(np.median(mid_c - mid_p + both.strike * np.exp(-rate * T)))

# Out-of-the-money options only: they are the liquid side of each strike.
quotes = pd.concat([
    chain.puts[chain.puts.strike < spot].assign(kind="put"),
    chain.calls[chain.calls.strike >= spot].assign(kind="call"),
])
quotes = quotes[(quotes.bid > 0) & (quotes.ask > quotes.bid)]
quotes = quotes.assign(mid=(quotes.bid + quotes.ask) / 2)
quotes = quotes[abs(np.log(quotes.strike / spot)) < 0.25]  # drop far wings with junk quotes

quotes["iv"] = [implied_vol(m, spot, k, T, rate, kind)
                for m, k, kind in zip(quotes.mid, quotes.strike, quotes.kind)]
quotes = quotes.dropna(subset=["iv"])
quotes["moneyness"] = quotes.strike / spot
quotes[["strike", "kind", "bid", "ask", "mid", "iv", "moneyness"]].to_csv("smile_data.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 4.5))
for kind, color in [("put", "#d62728"), ("call", "#1f77b4")]:
    side = quotes[quotes.kind == kind]
    ax.scatter(side.moneyness, side.iv * 100, s=14, color=color, label=f"OTM {kind}s")
ax.axvline(1, color="gray", lw=0.8, ls="--")
ax.set(xlabel="Strike / spot (spot implied by put-call parity)", ylabel="Implied volatility (%)",
       title=f"{ticker} volatility smile, expiry {expiry} ({T * 365:.0f} days), priced {today}")
ax.legend(frameon=False)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("smile.png", dpi=150)
print(f"quoted spot={quoted_spot:.2f} parity spot={spot:.2f} r={rate:.3%} expiry={expiry} points={len(quotes)} -> smile.png")
