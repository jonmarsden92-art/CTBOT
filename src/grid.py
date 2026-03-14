"""
grid.py - Grid Trading Strategy
=================================
Places a grid of buy and sell orders at regular price intervals.

How it works:
  1. Detect current price and volatility
  2. Set buy levels below current price at regular intervals
  3. When price drops to a level, buy
  4. When price recovers by grid_spacing, sell for profit
  5. Repeat continuously - profits from every oscillation

Best in: sideways/ranging markets
Avoid in: strong trending markets

Example with $50 on BTC at $70,000:
  Grid levels: $69,300 / $68,600 / $67,900 / $67,200
  Spacing: 1% = $700 per level
  Size per level: $12.50
  Each bounce = ~1% profit
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

GRID_STATE_FILE  = Path("logs/grid_state.json")
GRID_LEVELS      = 4
GRID_SPACING_PCT = 0.012
GRID_BUDGET_PCT  = 0.20
MIN_GRID_USD     = 1.50
GRID_SYMBOLS     = ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "DOGE/USD"]


def load_grid_state() -> dict:
    if GRID_STATE_FILE.exists():
        try:
            return json.loads(GRID_STATE_FILE.read_text())
        except Exception:
            pass
    return {"grids": {}, "total_profit": 0.0, "trades": [], "last_run": None}


def save_grid_state(state: dict):
    GRID_STATE_FILE.parent.mkdir(exist_ok=True)
    state["trades"] = state["trades"][-200:]
    GRID_STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def should_run_grid(df: pd.DataFrame, regime: str) -> bool:
    try:
        if regime in ("crash",):
            return False
        close   = df["close"].astype(float)
        high    = df["high"].astype(float)
        low     = df["low"].astype(float)
        prev    = close.shift(1)
        tr      = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
        atr_pct = float(tr.rolling(14).mean().iloc[-1]) / float(close.iloc[-1])
        range24 = (float(high.iloc[-24:].max()) - float(low.iloc[-24:].min())) / float(close.iloc[-1]) if len(close) >= 24 else 0
        return atr_pct > 0.004 and range24 < 0.10
    except Exception:
        return False


def run_grid(api, all_bars: dict, cash: float, regime: str,
             grid_state: dict, state: dict) -> dict:
    grid_budget = cash * GRID_BUDGET_PCT
    per_level   = grid_budget / GRID_LEVELS

    if per_level < MIN_GRID_USD:
        return grid_state

    for symbol in GRID_SYMBOLS:
        if symbol not in all_bars:
            continue
        df    = all_bars[symbol]
        close = df["close"].astype(float)
        high  = df["high"].astype(float)
        low   = df["low"].astype(float)
        price = float(close.iloc[-1])

        if not should_run_grid(df, regime):
            continue

        prev    = close.shift(1)
        tr      = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
        atr_pct = float(tr.rolling(14).mean().iloc[-1]) / price
        spacing = max(GRID_SPACING_PCT, atr_pct * 0.8)

        sym_key = symbol.replace("/", "")

        if sym_key not in grid_state["grids"]:
            levels = [round(price * (1 - spacing * (i + 1)), 6) for i in range(GRID_LEVELS)]
            grid_state["grids"][sym_key] = {
                "levels": levels, "entries": {}, "active": True,
                "base_price": price, "created": datetime.now().isoformat(),
            }
            log.info("🔲 Grid created " + symbol + " levels=" + str([round(l, 2) for l in levels]))
            continue

        grid   = grid_state["grids"][sym_key]
        levels = grid["levels"]
        base   = grid.get("base_price", price)

        if abs(price - base) / base > 0.05:
            levels = [round(price * (1 - spacing * (i + 1)), 6) for i in range(GRID_LEVELS)]
            grid["levels"]    = levels
            grid["base_price"] = price
            grid["entries"]   = {}
            log.info("🔲 Grid reset " + symbol)

        prev_price = float(close.iloc[-2]) if len(close) >= 2 else price

        for i, level_price in enumerate(levels):
            key = str(i)

            if key in grid["entries"]:
                entry  = grid["entries"][key]["price"]
                qty    = grid["entries"][key]["qty"]
                target = entry * (1 + spacing)

                if price >= target:
                    try:
                        api.submit_order(symbol=symbol, qty=qty, side="sell",
                                         type="market", time_in_force="gtc")
                        profit = qty * (price - entry)
                        log.info("🔲 Grid SELL " + symbol + " level=" + str(i) +
                                 " profit=$" + str(round(profit, 4)))
                        del grid["entries"][key]
                        grid_state["total_profit"] += profit
                        state["wins"] = state.get("wins", 0) + 1
                        grid_state["trades"].append({
                            "symbol": sym_key, "side": "sell", "level": i,
                            "price": price, "entry": entry, "profit": profit,
                            "time": datetime.now().isoformat(),
                        })
                    except Exception as e:
                        log.error("Grid sell failed: " + str(e))
            else:
                if prev_price > level_price >= price:
                    qty = per_level / price
                    qty = round(qty, 8) if price > 10000 else round(qty, 6) if price > 1 else round(qty, 2)
                    if qty * price < MIN_GRID_USD:
                        continue
                    try:
                        api.submit_order(symbol=symbol, qty=qty, side="buy",
                                         type="market", time_in_force="gtc")
                        grid["entries"][key] = {"price": price, "qty": qty,
                                                "time": datetime.now().isoformat()}
                        log.info("🔲 Grid BUY " + symbol + " level=" + str(i) +
                                 " @ $" + str(round(price, 4)))
                        grid_state["trades"].append({
                            "symbol": sym_key, "side": "buy", "level": i,
                            "price": price, "qty": qty,
                            "time": datetime.now().isoformat(),
                        })
                    except Exception as e:
                        log.error("Grid buy failed: " + str(e))

    grid_state["last_run"] = datetime.now().isoformat()
    if grid_state["total_profit"] != 0:
        log.info("🔲 Grid total profit: $" + str(round(grid_state["total_profit"], 4)))
    return grid_state
