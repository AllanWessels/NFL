"""NFL in-game win-probability forecasting.

Hard rule: market data (betting lines, Kalshi prices) is NEVER a model input.
It is used only in evaluation, to measure edge. See docs/ADR/0001-no-market-inputs.md.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
