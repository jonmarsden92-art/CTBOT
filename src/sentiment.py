"""
sentiment.py - Crypto Sentiment Analysis
=========================================
Fetches and scores market sentiment from:
  - Fear & Greed Index (alternative.me)
  - CoinGecko market data (trending, volume)
  - Recent price momentum as sentiment proxy

Returns a SentimentScore between -1.0 (extreme fear/sell) and +1.0 (extreme greed/buy)
and a regime: fear | neutral | greed | extreme_greed

Best signal historically:
  - Extreme Fear (score 0-25) + market stabilising = STRONG BUY
  - Extreme Greed (score 75-100) = caution, potential reversal
"""

import json
import logging
import requests
from datetime import datetime
from typing import Optional

log = logging.getLogger(__name__)

FEAR_GREED_URL  = "https://api.alternative.me/fng/?limit=2&format=json"
COINGECKO_URL   = "https://api.coingecko.com/api/v3/global"
TRENDING_URL    = "https://api.coingecko.com/api/v3/search/trending"

HEADERS = {"User-Agent": "CryptoTrader/4.0"}
TIMEOUT = 8


def fetch_fear_greed() -> Optional[dict]:
    """Fetch Fear & Greed Index from alternative.me"""
    try:
        r    = requests.get(FEAR_GREED_URL, headers=HEADERS, timeout=TIMEOUT)
        data = r.json()
        if "data" in data and len(data["data"]) >= 1:
            today     = data["data"][0]
            yesterday = data["data"][1] if len(data["data"]) > 1 else today
            score     = int(today["value"])
            prev      = int(yesterday["value"])
            return {
                "score":      score,
                "prev_score": prev,
                "label":      today["value_classification"],
                "change":     score - prev,
                "timestamp":  today["timestamp"],
            }
    except Exception as e:
        log.debug("Fear & Greed fetch failed: " + str(e))
    return None


def fetch_global_market() -> Optional[dict]:
    """Fetch global crypto market data from CoinGecko"""
    try:
        r    = requests.get(COINGECKO_URL, headers=HEADERS, timeout=TIMEOUT)
        data = r.json().get("data", {})
        return {
            "market_cap_change_24h": data.get("market_cap_change_percentage_24h_usd", 0),
            "btc_dominance":         data.get("market_cap_percentage", {}).get("btc", 50),
            "total_volume_24h":      data.get("total_volume", {}).get("usd", 0),
            "active_cryptos":        data.get("active_cryptocurrencies", 0),
        }
    except Exception as e:
        log.debug("Global market fetch failed: " + str(e))
    return None


def fetch_trending() -> list:
    """Fetch trending coins from CoinGecko"""
    try:
        r    = requests.get(TRENDING_URL, headers=HEADERS, timeout=TIMEOUT)
        data = r.json()
        coins = data.get("coins", [])
        return [c["item"]["symbol"].upper() for c in coins[:7]]
    except Exception as e:
        log.debug("Trending fetch failed: " + str(e))
    return []


def get_sentiment() -> dict:
    """
    Get full sentiment picture.
    Returns dict with score, regime, multiplier and signals.
    """
    fg        = fetch_fear_greed()
    market    = fetch_global_market()
    trending  = fetch_trending()

    score      = 50  # neutral default
    regime     = "neutral"
    signals    = []
    multiplier = 1.0

    if fg:
        score  = fg["score"]
        change = fg["change"]

        if score <= 25:
            regime = "extreme_fear"
            # Extreme fear = best buying opportunity historically
            # But only if sentiment is improving (change > 0)
            if change > 0:
                multiplier = 1.4
                signals.append("extreme_fear_improving")
            else:
                multiplier = 0.8  # still falling, wait
                signals.append("extreme_fear_worsening")
        elif score <= 45:
            regime     = "fear"
            multiplier = 1.15 if change >= 0 else 0.9
            signals.append("fear_zone")
        elif score <= 55:
            regime     = "neutral"
            multiplier = 1.0
            signals.append("neutral")
        elif score <= 75:
            regime     = "greed"
            multiplier = 0.85  # getting expensive, be cautious
            signals.append("greed_caution")
        else:
            regime     = "extreme_greed"
            multiplier = 0.6   # very likely reversal coming
            signals.append("extreme_greed_warning")

        log.info("😨 Fear & Greed: " + str(score) + " (" + fg["label"] + ")" +
                 " change=" + str(change) + " mult=" + str(multiplier))

    if market:
        mc_change = market["market_cap_change_24h"]
        btc_dom   = market["btc_dominance"]

        # Rising market cap = positive
        if mc_change > 2:
            multiplier *= 1.1
            signals.append("market_cap_rising")
        elif mc_change < -3:
            multiplier *= 0.85
            signals.append("market_cap_falling")

        # High BTC dominance = alts may be weak
        if btc_dom > 60:
            signals.append("btc_dominance_high")
        elif btc_dom < 45:
            multiplier *= 1.1
            signals.append("alt_season_possible")

        log.info("🌍 Market: cap_change=" + str(round(mc_change, 2)) + "%" +
                 " btc_dom=" + str(round(btc_dom, 1)) + "%")

    if trending:
        log.info("🔥 Trending: " + ", ".join(trending[:5]))

    return {
        "score":      score,
        "regime":     regime,
        "multiplier": round(multiplier, 3),
        "signals":    signals,
        "trending":   trending,
        "fear_greed": fg,
        "market":     market,
        "timestamp":  datetime.now().isoformat(),
    }
