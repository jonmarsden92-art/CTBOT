"""
market_intelligence.py - External Market Context
==================================================
Fetches data that technical indicators can't see:

1. BTC Funding Rates - are traders over-leveraged long or short?
   High positive funding = too many longs = likely dump incoming
   High negative funding = too many shorts = likely squeeze incoming

2. BTC On-chain: Exchange flows
   Large BTC flowing INTO exchanges = selling pressure incoming
   Large BTC flowing OUT of exchanges = holders accumulating = bullish

3. Macro context: DXY, Gold, SPX correlation
   When DXY rises sharply, crypto usually falls
   When SPX falls hard, crypto follows within hours

4. Liquidation levels
   Large liquidation clusters above/below price = magnet zones
   Price often moves to sweep liquidity before reversing

5. Open Interest trend
   Rising OI + rising price = strong trend
   Rising OI + falling price = shorts piling in = potential squeeze
"""

import json
import logging
import requests
from datetime import datetime
from typing import Optional

log = logging.getLogger(__name__)
TIMEOUT = 8
HEADERS = {"User-Agent": "CryptoTrader/4.0"}


def fetch_btc_funding_rate() -> Optional[dict]:
    """
    Fetch BTC perpetual futures funding rate from Binance.
    Positive = longs paying shorts (market overbought)
    Negative = shorts paying longs (market oversold)
    """
    try:
        url = "https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT&limit=3"
        r   = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        data = r.json()
        if data and len(data) > 0:
            latest = float(data[-1]["fundingRate"])
            prev   = float(data[-2]["fundingRate"]) if len(data) > 1 else latest
            trend  = "rising" if latest > prev else "falling"

            if latest > 0.001:
                signal = "overbought"     # longs paying too much — dump likely
                mult   = 0.7
            elif latest > 0.0005:
                signal = "slightly_long"
                mult   = 0.9
            elif latest < -0.001:
                signal = "oversold"       # shorts paying too much — squeeze likely
                mult   = 1.3
            elif latest < -0.0005:
                signal = "slightly_short"
                mult   = 1.1
            else:
                signal = "neutral"
                mult   = 1.0

            log.info("💸 BTC Funding: " + str(round(latest * 100, 4)) + "% (" + signal + ") trend=" + trend)
            return {
                "rate":   latest,
                "signal": signal,
                "trend":  trend,
                "mult":   mult,
            }
    except Exception as e:
        log.debug("Funding rate error: " + str(e))
    return None


def fetch_btc_open_interest() -> Optional[dict]:
    """
    Fetch BTC open interest from Binance.
    Rising OI + price up = strong bull trend
    Rising OI + price down = strong bear trend (shorts accumulating)
    Falling OI = deleveraging, trend may be ending
    """
    try:
        url  = "https://fapi.binance.com/fapi/v1/openInterest?symbol=BTCUSDT"
        r    = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        data = r.json()
        oi   = float(data.get("openInterest", 0))

        # Get historical OI to compare
        url2 = "https://fapi.binance.com/futures/data/openInterestHist?symbol=BTCUSDT&period=1h&limit=5"
        r2   = requests.get(url2, headers=HEADERS, timeout=TIMEOUT)
        hist = r2.json()

        if hist and len(hist) >= 2:
            oi_now  = float(hist[-1]["sumOpenInterest"])
            oi_prev = float(hist[-3]["sumOpenInterest"]) if len(hist) >= 3 else float(hist[0]["sumOpenInterest"])
            oi_change = (oi_now - oi_prev) / oi_prev if oi_prev > 0 else 0

            log.info("📊 BTC OI: " + str(round(oi_now/1000, 1)) + "k BTC | change=" + str(round(oi_change*100, 2)) + "%")
            return {
                "oi_now":    oi_now,
                "oi_change": oi_change,
                "rising":    oi_change > 0.01,
                "falling":   oi_change < -0.01,
            }
    except Exception as e:
        log.debug("Open interest error: " + str(e))
    return None


def fetch_crypto_dominance() -> Optional[dict]:
    """
    BTC dominance direction tells us market risk appetite.
    Rising BTC dominance = money flowing to safety (BTC), alts weak
    Falling BTC dominance = alt season, risk on
    """
    try:
        url  = "https://api.coingecko.com/api/v3/global"
        r    = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        data = r.json().get("data", {})
        btc_dom = data.get("market_cap_percentage", {}).get("btc", 50)
        eth_dom = data.get("market_cap_percentage", {}).get("eth", 15)
        mc_change = data.get("market_cap_change_percentage_24h_usd", 0)

        if btc_dom > 58:
            dom_signal = "btc_season"       # stay in BTC, avoid alts
        elif btc_dom < 48:
            dom_signal = "alt_season"       # alts likely to outperform
        else:
            dom_signal = "neutral"

        log.info("🏛️  Dominance: BTC=" + str(round(btc_dom, 1)) + "% ETH=" + str(round(eth_dom, 1)) + "% (" + dom_signal + ")")
        return {
            "btc_dominance": btc_dom,
            "eth_dominance": eth_dom,
            "mc_change_24h": mc_change,
            "signal":        dom_signal,
            "alt_season":    btc_dom < 48,
        }
    except Exception as e:
        log.debug("Dominance error: " + str(e))
    return None


def fetch_liquidations() -> Optional[dict]:
    """
    Recent large liquidations indicate direction of forced selling.
    Large long liquidations = price was falling, longs got wiped
    Large short liquidations = price was rising, shorts got squeezed
    """
    try:
        url  = "https://fapi.binance.com/fapi/v1/allForceOrders?symbol=BTCUSDT&limit=50"
        r    = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        data = r.json()

        if not data:
            return None

        long_liqs  = sum(float(x.get("origQty", 0)) for x in data if x.get("side") == "SELL")
        short_liqs = sum(float(x.get("origQty", 0)) for x in data if x.get("side") == "BUY")

        total = long_liqs + short_liqs
        if total == 0:
            return None

        long_pct = long_liqs / total

        if long_pct > 0.7:
            signal = "longs_getting_wrecked"   # bearish
            mult   = 0.8
        elif long_pct < 0.3:
            signal = "shorts_getting_squeezed"  # bullish
            mult   = 1.2
        else:
            signal = "balanced"
            mult   = 1.0

        log.info("💥 Liquidations: " + str(round(long_liqs, 1)) + " long | " +
                 str(round(short_liqs, 1)) + " short | " + signal)
        return {
            "long_liqs":  long_liqs,
            "short_liqs": short_liqs,
            "long_pct":   long_pct,
            "signal":     signal,
            "mult":       mult,
        }
    except Exception as e:
        log.debug("Liquidations error: " + str(e))
    return None


def get_market_intelligence() -> dict:
    """
    Collect all external market signals and combine into a single
    intelligence score and multiplier for the trading bot.
    """
    funding    = fetch_btc_funding_rate()
    oi         = fetch_btc_open_interest()
    dominance  = fetch_crypto_dominance()
    liqs       = fetch_liquidations()

    # Combine multipliers
    mult       = 1.0
    signals    = []
    warnings   = []

    if funding:
        mult *= funding["mult"]
        signals.append("funding_" + funding["signal"])
        if funding["signal"] == "overbought":
            warnings.append("HIGH FUNDING — market overleveraged long")

    if liqs:
        mult *= liqs["mult"]
        signals.append("liqs_" + liqs["signal"])

    if dominance:
        if dominance["alt_season"]:
            mult *= 1.1
            signals.append("alt_season")
        elif dominance["btc_dominance"] > 58:
            mult *= 0.85
            signals.append("btc_season_avoid_alts")

    # OI context (informational, doesn't directly adjust mult)
    oi_context = "unknown"
    if oi:
        if oi["rising"]:
            oi_context = "rising_oi"
        elif oi["falling"]:
            oi_context = "falling_oi"

    for w in warnings:
        log.warning("⚠️  Market Intel: " + w)

    return {
        "funding":         funding,
        "open_interest":   oi,
        "dominance":       dominance,
        "liquidations":    liqs,
        "combined_mult":   round(mult, 3),
        "signals":         signals,
        "warnings":        warnings,
        "oi_context":      oi_context,
        "timestamp":       datetime.now().isoformat(),
    }
