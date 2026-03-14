"""
patterns.py - Candlestick Pattern Detection
============================================
Detects high-probability reversal and continuation patterns.

Best patterns for crypto:
  - Hammer / Inverted Hammer (bullish reversal after downtrend)
  - Bullish Engulfing (strong reversal signal)
  - Morning Star (3-bar reversal pattern)
  - Doji (indecision — often precedes reversal)
  - Three White Soldiers (strong uptrend continuation)
  - Shooting Star (bearish reversal)
  - Bearish Engulfing (strong bearish reversal)

Each pattern returns a score and direction (bull/bear/neutral)
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional

log = logging.getLogger(__name__)


def is_hammer(o: float, h: float, l: float, c: float) -> bool:
    """Hammer: small body at top, long lower wick. Bullish reversal."""
    body   = abs(c - o)
    candle = h - l
    if candle == 0:
        return False
    lower_wick = min(o, c) - l
    upper_wick = h - max(o, c)
    return (lower_wick >= body * 2.0 and
            upper_wick <= body * 0.5 and
            body <= candle * 0.35)


def is_inverted_hammer(o: float, h: float, l: float, c: float) -> bool:
    """Inverted Hammer: small body at bottom, long upper wick. Bullish reversal."""
    body       = abs(c - o)
    candle     = h - l
    if candle == 0:
        return False
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    return (upper_wick >= body * 2.0 and
            lower_wick <= body * 0.5 and
            body <= candle * 0.35)


def is_doji(o: float, h: float, l: float, c: float) -> bool:
    """Doji: open ≈ close. Indecision — often precedes reversal."""
    body   = abs(c - o)
    candle = h - l
    return candle > 0 and body <= candle * 0.1


def is_bullish_engulfing(o1: float, c1: float, o2: float, c2: float) -> bool:
    """
    Bullish Engulfing: bearish candle followed by larger bullish candle.
    Very reliable reversal pattern.
    """
    prev_bearish  = c1 < o1
    curr_bullish  = c2 > o2
    curr_engulfs  = o2 <= c1 and c2 >= o1
    return prev_bearish and curr_bullish and curr_engulfs


def is_bearish_engulfing(o1: float, c1: float, o2: float, c2: float) -> bool:
    """
    Bearish Engulfing: bullish candle followed by larger bearish candle.
    """
    prev_bullish = c1 > o1
    curr_bearish = c2 < o2
    curr_engulfs = o2 >= c1 and c2 <= o1
    return prev_bullish and curr_bearish and curr_engulfs


def is_morning_star(o1: float, c1: float,
                    o2: float, c2: float,
                    o3: float, c3: float) -> bool:
    """
    Morning Star: 3-bar bullish reversal.
    Bar 1: big bearish, Bar 2: small body (gap down), Bar 3: big bullish
    """
    bar1_bearish = c1 < o1 and (o1 - c1) > abs(o2 - c2) * 2
    bar2_small   = abs(o2 - c2) < abs(o1 - c1) * 0.4
    bar3_bullish = c3 > o3 and (c3 - o3) > abs(o2 - c2) * 2
    bar3_recovers = c3 > (o1 + c1) / 2
    return bar1_bearish and bar2_small and bar3_bullish and bar3_recovers


def is_evening_star(o1: float, c1: float,
                    o2: float, c2: float,
                    o3: float, c3: float) -> bool:
    """Evening Star: 3-bar bearish reversal (opposite of morning star)."""
    bar1_bullish = c1 > o1 and (c1 - o1) > abs(o2 - c2) * 2
    bar2_small   = abs(o2 - c2) < abs(o1 - c1) * 0.4
    bar3_bearish = c3 < o3 and (o3 - c3) > abs(o2 - c2) * 2
    bar3_drops   = c3 < (o1 + c1) / 2
    return bar1_bullish and bar2_small and bar3_bearish and bar3_drops


def is_three_white_soldiers(bars: list) -> bool:
    """
    Three White Soldiers: 3 consecutive bullish candles each closing higher.
    Strong continuation signal.
    """
    if len(bars) < 3:
        return False
    for o, h, l, c in bars[-3:]:
        if c <= o:
            return False
    # Each close higher than previous
    closes = [c for o, h, l, c in bars[-3:]]
    return closes[1] > closes[0] and closes[2] > closes[1]


def is_shooting_star(o: float, h: float, l: float, c: float) -> bool:
    """Shooting Star: small body at bottom, long upper wick. Bearish reversal."""
    body       = abs(c - o)
    candle     = h - l
    if candle == 0:
        return False
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    return (upper_wick >= body * 2.5 and
            lower_wick <= body * 0.3 and
            body <= candle * 0.3)


def detect_patterns(df: pd.DataFrame) -> dict:
    """
    Scan last 5 candles for all patterns.
    Returns score (positive = bullish, negative = bearish) and patterns found.
    """
    try:
        if len(df) < 5:
            return {"score": 0, "patterns": [], "direction": "neutral"}

        opens  = df["open"].astype(float).values
        highs  = df["high"].astype(float).values
        lows   = df["low"].astype(float).values
        closes = df["close"].astype(float).values

        # Check trend context (patterns more reliable after trend)
        trend_down = closes[-5] > closes[-3] > closes[-1]  # downtrend = bull patterns more reliable
        trend_up   = closes[-5] < closes[-3] < closes[-1]  # uptrend = bear patterns more reliable

        patterns = []
        score    = 0

        # Latest candle patterns
        o, h, l, c = opens[-1], highs[-1], lows[-1], closes[-1]

        if is_hammer(o, h, l, c) and trend_down:
            patterns.append("hammer")
            score += 3

        if is_inverted_hammer(o, h, l, c) and trend_down:
            patterns.append("inverted_hammer")
            score += 2

        if is_doji(o, h, l, c):
            patterns.append("doji")
            # Doji after downtrend = bullish, after uptrend = bearish
            if trend_down:
                score += 1
            elif trend_up:
                score -= 1

        if is_shooting_star(o, h, l, c) and trend_up:
            patterns.append("shooting_star")
            score -= 3

        # Two-candle patterns
        o1, c1 = opens[-2], closes[-2]
        o2, c2 = opens[-1], closes[-1]

        if is_bullish_engulfing(o1, c1, o2, c2) and trend_down:
            patterns.append("bullish_engulfing")
            score += 4  # very reliable

        if is_bearish_engulfing(o1, c1, o2, c2) and trend_up:
            patterns.append("bearish_engulfing")
            score -= 4

        # Three-candle patterns
        o1, c1 = opens[-3], closes[-3]
        o2, c2 = opens[-2], closes[-2]
        o3, c3 = opens[-1], closes[-1]

        if is_morning_star(o1, c1, o2, c2, o3, c3):
            patterns.append("morning_star")
            score += 5  # very strong

        if is_evening_star(o1, c1, o2, c2, o3, c3):
            patterns.append("evening_star")
            score -= 5

        # Multi-candle
        bars = list(zip(opens[-3:], highs[-3:], lows[-3:], closes[-3:]))
        if is_three_white_soldiers(bars) and not trend_up:
            patterns.append("three_white_soldiers")
            score += 3

        direction = "bull" if score > 0 else "bear" if score < 0 else "neutral"

        if patterns:
            log.debug("Patterns: " + str(patterns) + " score=" + str(score))

        return {
            "score":      score,
            "patterns":   patterns,
            "direction":  direction,
            "trend_down": trend_down,
            "trend_up":   trend_up,
        }

    except Exception as e:
        log.debug("Pattern detection error: " + str(e))
        return {"score": 0, "patterns": [], "direction": "neutral"}
