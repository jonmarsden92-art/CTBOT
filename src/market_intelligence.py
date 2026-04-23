import json
import os
from datetime import datetime

REPORT_PATH = "crypto_report.json"
STATE_PATH = "crypto_state.json"

def load_json(path):
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    except Exception as e:
        log.warning(f"⚠️ Failed to load {path}: {e}")
    return {}

def save_json(path, data):
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        log.warning(f"⚠️ Failed to save {path}: {e}")


# -------------------------------
# Load GitHub-generated analysis
# -------------------------------
report = load_json(REPORT_PATH)

all_analyses = report.get("all_analyses", {})
regime = report.get("regime", "unknown")


# -------------------------------
# Market health calculation
# -------------------------------
if not isinstance(all_analyses, dict) or not all_analyses:
    log.warning("⚠️ all_analyses not available — skipping market health check")
    bearish_count = 0
    total_coins = 0
    bearish_ratio = 0
    avg_trend = 0
else:
    valid_analyses = [a for a in all_analyses.values() if isinstance(a, dict)]
    total_coins = len(valid_analyses)

    bearish_coins = [a for a in valid_analyses if a.get("trend_24h", 0) < -0.08]
    bearish_count = len(bearish_coins)

    bearish_ratio = bearish_count / total_coins if total_coins else 0
    avg_trend = sum(a.get("trend_24h", 0) for a in valid_analyses) / total_coins


# -------------------------------
# Decision logic
# -------------------------------
market_bearish = False
market_mode = "neutral"

if regime == "crash":
    market_bearish = True
    market_mode = "crash"
    log.warning("🚨 CRASH REGIME — pausing all buys")

elif bearish_ratio >= 0.8 or avg_trend < -0.06:
    market_bearish = True
    market_mode = "collapse"
    log.warning(f"📉 MARKET COLLAPSE — {bearish_count}/{total_coins} coins down | avg {avg_trend:.2%}")

elif bearish_ratio >= 0.5 and avg_trend < -0.03:
    market_bearish = True
    market_mode = "downtrend"
    log.warning(f"⚠️ HEAVY DOWNTREND — {bearish_count}/{total_coins} coins down | avg {avg_trend:.2%}")

elif bearish_ratio >= 0.3:
    market_bearish = False
    market_mode = "dip_buy"
    log.info(f"📉 DIP BUYING MODE — {bearish_count}/{total_coins} coins down | avg {avg_trend:.2%}")

else:
    market_mode = "healthy"
    log.info(f"✅ Market healthy — {bearish_count}/{total_coins} coins down | avg {avg_trend:.2%}")


# -------------------------------
# Write back to GitHub state
# -------------------------------
state_update = load_json(STATE_PATH)

state_update["market_health"] = {
    "market_bearish": market_bearish,
    "market_mode": market_mode,
    "bearish_ratio": bearish_ratio,
    "avg_trend": avg_trend,
    "total_coins": total_coins,
    "bearish_count": bearish_count,
    "updated_at": datetime.utcnow().isoformat()
}

save_json(STATE_PATH, state_update)


# Optional: also inject back into report
report["market_health"] = state_update["market_health"]
save_json(REPORT_PATH, report)
