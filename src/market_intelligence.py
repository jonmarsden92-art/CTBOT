import json
import os
import random
import logging
from datetime import datetime

# -----------------------------
# Logger setup (FIX)
# -----------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("market_intelligence")


os.makedirs("logs", exist_ok=True)


def simulate_market_data():
    coins = ["BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "AVAX"]

    all_analyses = {}

    for c in coins:
        all_analyses[c] = {
            "trend_24h": random.uniform(-0.12, 0.12),
            "rsi": random.uniform(20, 80),
            "volume_score": random.uniform(0.1, 1.0)
        }

    return all_analyses


def detect_regime(all_analyses):
    trends = [v["trend_24h"] for v in all_analyses.values()]
    avg_trend = sum(trends) / len(trends)

    bearish = sum(1 for t in trends if t < -0.08)
    ratio = bearish / len(trends)

    if ratio > 0.8 or avg_trend < -0.06:
        return "crash"
    elif ratio > 0.5:
        return "downtrend"
    elif ratio > 0.3:
        return "dip_buy"
    else:
        return "healthy"


def main():
    all_analyses = simulate_market_data()
    regime = detect_regime(all_analyses)

    report = {
        "all_analyses": all_analyses,
        "regime": regime,
        "updated_at": datetime.utcnow().isoformat()
    }

    with open("logs/crypto_report.json", "w") as f:
        json.dump(report, f, indent=2)

    log.info(f"Market report generated | regime={regime}")


if __name__ == "__main__":
    main()
