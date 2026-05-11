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
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)
TIMEOUT = 8
HEADERS = {"User-Agent": "CryptoTrader/4.0"}


def fetch_btc_funding_rate() -> Optional[dict]:
    """
    Fetch BTC funding rate from CoinGlass (aggregated across exchanges).
    Positive = longs paying shorts (market overbought)
    Negative = shorts paying longs (market oversold)
    """
    try:
        # CoinGlass free API - aggregated funding rates
        url = "https://open-api.coinglass.com/public/v2/funding"
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        data = r.json()
        
        if data.get("success") and data.get("data"):
            # Get BTC funding rate (average across exchanges)
            btc_data = next((x for x in data["data"] if x.get("symbol") == "BTC"), None)
            if not btc_data:
                return None
                
            latest = float(btc_data.get("uMarginList", [{}])[0].get("rate", 0)) / 100  # convert from %
            trend = "neutral"  # CoinGlass doesn't provide historical in free tier

            if latest > 0.001:
                signal = "overbought"
                mult   = 0.7
            elif latest > 0.0005:
                signal = "slightly_long"
                mult   = 0.9
            elif latest < -0.001:
                signal = "oversold"
                mult   = 1.3
            elif latest < -0.0005:
                signal = "slightly_short"
                mult   = 1.1
            else:
                signal = "neutral"
                mult   = 1.0

            log.info("💸 BTC Funding: " + str(round(latest * 100, 4)) + "% (" + signal + ")")
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
    OI data disabled - requires Binance access or paid API.
    Keeping function to avoid breaking imports.
    """
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
    Liquidations data disabled - requires Binance access.
    Keeping function to avoid breaking imports.
    """
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
        if oi.get("rising"):
            oi_context = "rising_oi"
        elif oi.get("falling"):
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


if __name__ == "__main__":
    # When run directly, save market intelligence to logs/
    logging.basicConfig(level=logging.INFO)
    intel = get_market_intelligence()
    logs_dir = Path(__file__).parent.parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    out_file = logs_dir / "market_intel.json"
    with open(out_file, "w") as f:
        json.dump(intel, f, indent=2, default=str)
    print(f"Market intelligence saved to {out_file}")
    print(json.dumps(intel, indent=2, default=str))
