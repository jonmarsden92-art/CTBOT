"""
correlation.py - Cross-Asset Correlation Analysis
===================================================
BTC is the market leader. When BTC moves, alts follow.

This module:
  1. Detects BTC momentum and trend
  2. Calculates each coin's correlation to BTC
  3. Identifies coins that are lagging BTC (about to move)
  4. Detects when alts are showing strength vs BTC (alt season)
  5. Warns when BTC is showing weakness (sell alts)

Key insight: low correlation + BTC rising = alt about to pump
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional

log = logging.getLogger(__name__)


def calc_correlation(btc_df: pd.DataFrame, alt_df: pd.DataFrame,
                     periods: int = 24) -> float:
    """Calculate rolling correlation between BTC and alt returns."""
    try:
        btc_ret = btc_df["close"].astype(float).pct_change().iloc[-periods:]
        alt_ret = alt_df["close"].astype(float).pct_change().iloc[-periods:]
        if len(btc_ret) < 10 or len(alt_ret) < 10:
            return 0.5
        corr = float(btc_ret.corr(alt_ret))
        return round(corr, 3) if not np.isnan(corr) else 0.5
    except Exception:
        return 0.5


def get_btc_regime(btc_df: pd.DataFrame) -> dict:
    """
    Full BTC analysis — the market leader.
    Returns direction, strength, and whether alts should follow.
    """
    try:
        close = btc_df["close"].astype(float)
        vol   = btc_df["volume"].astype(float)

        price     = float(close.iloc[-1])
        move_1h   = (price - float(close.iloc[-2])) / float(close.iloc[-2])
        move_4h   = (price - float(close.iloc[-5])) / float(close.iloc[-5])  if len(close) >= 5  else 0
        move_24h  = (price - float(close.iloc[-25])) / float(close.iloc[-25]) if len(close) >= 25 else 0

        # BTC trend
        sma20 = float(close.rolling(20).mean().iloc[-1])
        sma50 = float(close.rolling(50).mean().iloc[-1])
        above_20ma = price > sma20
        above_50ma = price > sma50

        # Volume confirmation
        avg_vol  = float(vol.rolling(20).mean().iloc[-1])
        vol_now  = float(vol.iloc[-1])
        vol_ratio = vol_now / avg_vol if avg_vol > 0 else 1.0

        # BTC momentum
        if move_4h > 0.03 and above_20ma:
            direction = "strong_bull"
            alt_signal = "follow_btc_up"
        elif move_4h > 0.01:
            direction = "bull"
            alt_signal = "cautious_buy"
        elif move_4h < -0.03:
            direction = "strong_bear"
            alt_signal = "avoid_alts"
        elif move_4h < -0.01:
            direction = "bear"
            alt_signal = "reduce_exposure"
        else:
            direction = "neutral"
            alt_signal = "trade_selectively"

        return {
            "price":       price,
            "move_1h":     round(move_1h,  4),
            "move_4h":     round(move_4h,  4),
            "move_24h":    round(move_24h, 4),
            "above_20ma":  above_20ma,
            "above_50ma":  above_50ma,
            "vol_ratio":   round(vol_ratio, 2),
            "direction":   direction,
            "alt_signal":  alt_signal,
        }
    except Exception as e:
        log.debug("BTC regime error: " + str(e))
        return {"direction": "unknown", "alt_signal": "neutral", "move_4h": 0}


def find_lagging_alts(btc_df: pd.DataFrame,
                      all_bars: Dict[str, pd.DataFrame]) -> list:
    """
    Find alts that are lagging BTC's recent move.
    If BTC moved up 2% but an alt only moved 0.5%, the alt may catch up.
    Returns list of symbols ranked by lag (most lagging first).
    """
    try:
        btc_close = btc_df["close"].astype(float)
        btc_move  = (float(btc_close.iloc[-1]) - float(btc_close.iloc[-4])) / float(btc_close.iloc[-4])

        if abs(btc_move) < 0.008:  # BTC needs to move at least 0.8% to look for laggards
            return []

        lagging = []
        for symbol, df in all_bars.items():
            if symbol == "BTC/USD":
                continue
            try:
                alt_close = df["close"].astype(float)
                alt_move  = (float(alt_close.iloc[-1]) - float(alt_close.iloc[-4])) / float(alt_close.iloc[-4])
                lag       = btc_move - alt_move  # positive = alt is lagging a BTC move up

                if btc_move > 0 and lag > 0.005:  # BTC up, alt hasn't caught up
                    corr = calc_correlation(btc_df, df, 12)
                    if corr > 0.6:  # only if historically correlated
                        lagging.append({
                            "symbol":   symbol,
                            "btc_move": round(btc_move, 4),
                            "alt_move": round(alt_move, 4),
                            "lag":      round(lag, 4),
                            "corr":     corr,
                            "score":    round(lag * corr * 10, 2),
                        })
            except Exception:
                continue

        lagging.sort(key=lambda x: x["score"], reverse=True)
        if lagging:
            log.info("🔗 Lagging alts: " +
                     ", ".join(x["symbol"] + "(" + str(x["lag"]) + ")" for x in lagging[:3]))
        return lagging[:5]

    except Exception as e:
        log.debug("Lagging alts error: " + str(e))
        return []


def get_alt_season_score(btc_df: pd.DataFrame,
                         all_bars: Dict[str, pd.DataFrame]) -> float:
    """
    Alt season score: how many alts are outperforming BTC?
    > 0.5 = alt season (favour alts)
    < 0.3 = BTC season (stick to BTC)
    """
    try:
        btc_close = btc_df["close"].astype(float)
        btc_move  = (float(btc_close.iloc[-1]) - float(btc_close.iloc[-25])) / float(btc_close.iloc[-25]) if len(btc_close) >= 25 else 0

        outperforming = 0
        total         = 0

        for symbol, df in all_bars.items():
            if symbol == "BTC/USD":
                continue
            try:
                alt_close = df["close"].astype(float)
                alt_move  = (float(alt_close.iloc[-1]) - float(alt_close.iloc[-25])) / float(alt_close.iloc[-25]) if len(alt_close) >= 25 else 0
                total    += 1
                if alt_move > btc_move:
                    outperforming += 1
            except Exception:
                continue

        score = outperforming / total if total > 0 else 0.5
        if score > 0.6:
            log.info("🌟 Alt season: " + str(round(score * 100)) + "% of alts outperforming BTC")
        return round(score, 3)

    except Exception as e:
        log.debug("Alt season error: " + str(e))
        return 0.5


def get_correlation_signals(btc_df: pd.DataFrame,
                             all_bars: Dict[str, pd.DataFrame]) -> dict:
    """
    Main correlation analysis.
    Returns BTC regime, lagging alts, alt season score.
    """
    btc_regime    = get_btc_regime(btc_df)
    lagging_alts  = find_lagging_alts(btc_df, all_bars)
    alt_season    = get_alt_season_score(btc_df, all_bars)

    log.info("🔗 BTC: " + btc_regime["direction"] +
             " 4h=" + str(btc_regime["move_4h"]) +
             " | AltSeason=" + str(round(alt_season * 100)) + "%" +
             " | Lagging=" + str(len(lagging_alts)))

    return {
        "btc_regime":   btc_regime,
        "lagging_alts": lagging_alts,
        "alt_season":   alt_season,
        "timestamp":    __import__("datetime").datetime.now().isoformat(),
    }
