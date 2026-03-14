"""
momentum.py - Breakout & Momentum Detection
============================================
Detects when a coin is starting a real momentum move.

Signals:
  - Volume breakout: volume suddenly 2x+ above average
  - Price breakout: price breaks above recent resistance
  - Momentum surge: price moved 1.5%+ in last 2 hours with volume
  - BTC leading: BTC moves first, alts follow 15-30 mins later
  - Squeeze breakout: low volatility followed by sudden expansion

These are the signals professional traders use to catch moves early.
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional

log = logging.getLogger(__name__)


def detect_volume_breakout(df: pd.DataFrame) -> dict:
    """
    Volume breakout: current volume is significantly above recent average.
    Smart money moves before price — volume spike often precedes price move.
    """
    try:
        vol     = df["volume"].astype(float)
        close   = df["close"].astype(float)
        avg_vol = float(vol.rolling(20).mean().iloc[-1])
        cur_vol = float(vol.iloc[-1])
        ratio   = cur_vol / avg_vol if avg_vol > 0 else 1.0

        # Check if volume is increasing over last 3 bars
        vol_trending_up = (float(vol.iloc[-1]) > float(vol.iloc[-2]) > float(vol.iloc[-3]))

        # Price direction during volume spike
        price_change = (float(close.iloc[-1]) - float(close.iloc[-3])) / float(close.iloc[-3])

        score = 0
        signal = "none"

        if ratio > 3.0 and price_change > 0:
            score  = 3
            signal = "volume_breakout_bull"
        elif ratio > 2.0 and price_change > 0 and vol_trending_up:
            score  = 2
            signal = "volume_surge_bull"
        elif ratio > 1.5 and price_change > 0:
            score  = 1
            signal = "volume_uptick"
        elif ratio > 2.0 and price_change < 0:
            score  = -2
            signal = "volume_breakout_bear"

        return {
            "vol_ratio":    round(ratio, 3),
            "vol_trending": vol_trending_up,
            "price_change": round(price_change, 4),
            "score":        score,
            "signal":       signal,
        }
    except Exception as e:
        log.debug("Volume breakout error: " + str(e))
        return {"vol_ratio": 1.0, "score": 0, "signal": "none"}


def detect_price_breakout(df: pd.DataFrame) -> dict:
    """
    Price breakout: price breaks above recent resistance level.
    Uses last 24 hours high as resistance, last 24 hours low as support.
    """
    try:
        close  = df["close"].astype(float)
        high   = df["high"].astype(float)
        low    = df["low"].astype(float)

        price     = float(close.iloc[-1])
        high_24h  = float(high.iloc[-24:].max()) if len(high) >= 24 else float(high.max())
        low_24h   = float(low.iloc[-24:].min())  if len(low)  >= 24 else float(low.min())
        range_24h = high_24h - low_24h

        # Where is price within the 24h range?
        range_pct = (price - low_24h) / range_24h if range_24h > 0 else 0.5

        # Breaking out above 90% of range = bullish breakout
        # Bouncing from below 10% of range = support bounce
        score  = 0
        signal = "none"

        if range_pct > 0.92:
            score  = 2
            signal = "resistance_breakout"
        elif range_pct > 0.80:
            score  = 1
            signal = "near_resistance"
        elif range_pct < 0.10:
            score  = 2
            signal = "support_bounce"
        elif range_pct < 0.20:
            score  = 1
            signal = "near_support_range"

        # Check if this is a new high vs yesterday
        high_48h = float(high.iloc[-48:].max()) if len(high) >= 48 else high_24h
        if price > high_48h * 0.999:
            score  += 1
            signal  = "48h_high_break"

        return {
            "range_pct":  round(range_pct, 3),
            "high_24h":   round(high_24h, 6),
            "low_24h":    round(low_24h, 6),
            "score":      score,
            "signal":     signal,
        }
    except Exception as e:
        log.debug("Price breakout error: " + str(e))
        return {"range_pct": 0.5, "score": 0, "signal": "none"}


def detect_squeeze(df: pd.DataFrame) -> dict:
    """
    Volatility squeeze: Bollinger Bands inside Keltner Channels.
    When volatility compresses then expands, a big move is coming.
    Low volatility -> high volatility transition = explosive move.
    """
    try:
        close  = df["close"].astype(float)
        high   = df["high"].astype(float)
        low    = df["low"].astype(float)

        # Bollinger Bands
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_up  = bb_mid + 2 * bb_std
        bb_low = bb_mid - 2 * bb_std
        bb_width = (bb_up - bb_low) / bb_mid

        # Keltner Channels (using ATR)
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs()
        ], axis=1).max(axis=1)
        atr    = tr.rolling(20).mean()
        kc_up  = bb_mid + 1.5 * atr
        kc_low = bb_mid - 1.5 * atr

        # Squeeze = BB inside KC
        squeeze_on  = (float(bb_up.iloc[-1])  < float(kc_up.iloc[-1])  and
                       float(bb_low.iloc[-1]) > float(kc_low.iloc[-1]))

        # Was squeeze on before but not now = breakout
        squeeze_was = (float(bb_up.iloc[-2])  < float(kc_up.iloc[-2])  and
                       float(bb_low.iloc[-2]) > float(kc_low.iloc[-2]))

        squeeze_fired = squeeze_was and not squeeze_on

        # Momentum at squeeze release (positive = bull breakout)
        momentum  = float(close.iloc[-1]) - float(close.iloc[-5])
        bb_w_now  = float(bb_width.iloc[-1])
        bb_w_prev = float(bb_width.iloc[-5])
        expanding = bb_w_now > bb_w_prev * 1.2  # BB expanding 20%+

        score  = 0
        signal = "none"

        if squeeze_fired and momentum > 0 and expanding:
            score  = 3
            signal = "squeeze_bull_breakout"
        elif squeeze_fired and momentum < 0 and expanding:
            score  = -2
            signal = "squeeze_bear_breakout"
        elif squeeze_on:
            score  = 0
            signal = "squeeze_building"
        elif expanding and momentum > 0:
            score  = 1
            signal = "volatility_expanding_bull"

        return {
            "squeeze_on":    squeeze_on,
            "squeeze_fired": squeeze_fired,
            "expanding":     expanding,
            "momentum":      round(momentum, 6),
            "score":         score,
            "signal":        signal,
        }
    except Exception as e:
        log.debug("Squeeze error: " + str(e))
        return {"squeeze_on": False, "score": 0, "signal": "none"}


def detect_btc_leading(btc_df: pd.DataFrame, alt_df: pd.DataFrame) -> dict:
    """
    BTC leads alts by 15-30 minutes.
    If BTC just made a strong move up, alts likely to follow.
    """
    try:
        btc_close = btc_df["close"].astype(float)
        alt_close = alt_df["close"].astype(float)

        # BTC move in last 2 bars
        btc_move_2h = (float(btc_close.iloc[-1]) - float(btc_close.iloc[-3])) / float(btc_close.iloc[-3])
        btc_move_1h = (float(btc_close.iloc[-1]) - float(btc_close.iloc[-2])) / float(btc_close.iloc[-2])

        # Alt move in last 2 bars
        alt_move_2h = (float(alt_close.iloc[-1]) - float(alt_close.iloc[-3])) / float(alt_close.iloc[-3])

        score  = 0
        signal = "none"

        # BTC moved strongly but alt hasn't caught up yet
        if btc_move_2h > 0.015 and alt_move_2h < btc_move_2h * 0.5:
            score  = 2
            signal = "btc_leading_bull_alt_lagging"
        elif btc_move_2h > 0.008 and alt_move_2h < 0:
            score  = 1
            signal = "btc_up_alt_diverging"
        elif btc_move_2h < -0.015:
            score  = -2
            signal = "btc_leading_bear"

        return {
            "btc_move_2h": round(btc_move_2h, 4),
            "btc_move_1h": round(btc_move_1h, 4),
            "alt_move_2h": round(alt_move_2h, 4),
            "score":       score,
            "signal":      signal,
        }
    except Exception as e:
        log.debug("BTC leading error: " + str(e))
        return {"btc_move_2h": 0, "score": 0, "signal": "none"}


def get_momentum_score(df: pd.DataFrame, btc_df: Optional[pd.DataFrame] = None,
                       symbol: str = "") -> dict:
    """
    Full momentum analysis combining all signals.
    Returns total momentum score and all signal details.
    """
    vol_break  = detect_volume_breakout(df)
    price_break = detect_price_breakout(df)
    squeeze    = detect_squeeze(df)

    btc_lead = {"score": 0, "signal": "none", "btc_move_2h": 0}
    if btc_df is not None and symbol != "BTC/USD":
        btc_lead = detect_btc_leading(btc_df, df)

    total_score = (
        vol_break["score"]   * 1.0 +
        price_break["score"] * 0.8 +
        squeeze["score"]     * 1.2 +
        btc_lead["score"]    * 0.7
    )

    signals_fired = []
    for s in [vol_break, price_break, squeeze, btc_lead]:
        if s.get("signal", "none") not in ("none", "neutral"):
            signals_fired.append(s["signal"])

    log.debug("Momentum " + symbol + " score=" + str(round(total_score, 2)) +
              " signals=" + str(signals_fired))

    return {
        "momentum_score":  round(total_score, 2),
        "vol_breakout":    vol_break,
        "price_breakout":  price_break,
        "squeeze":         squeeze,
        "btc_leading":     btc_lead,
        "signals_fired":   signals_fired,
    }
