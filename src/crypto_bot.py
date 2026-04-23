import json
import os
from datetime import datetime

STATE_PATH = "crypto_state.json"
REPORT_PATH = "crypto_report.json"


# -----------------------------
# Helpers
# -----------------------------
def load_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            try:
                return json.load(f)
            except:
                return {}
    return {}


def log_event(msg):
    print(f"[BOT] {msg}")


# -----------------------------
# Load system state
# -----------------------------
state = load_json(STATE_PATH)
report = load_json(REPORT_PATH)

market_health = state.get("market_health", {})
market_mode = market_health.get("market_mode", "unknown")
market_bearish = market_health.get("market_bearish", False)


all_analyses = report.get("all_analyses", {})


# -----------------------------
# Risk engine (simple adaptive rules)
# -----------------------------
def position_sizing(market_mode):
    if market_mode in ["crash", "collapse"]:
        return 0.0
    elif market_mode == "downtrend":
        return 0.25
    elif market_mode == "dip_buy":
        return 0.6
    else:
        return 0.4


def should_trade_coin(analysis, market_mode):
    momentum = analysis.get("trend_24h", 0)
    volume = analysis.get("volume_score", 1)

    # hard safety filter
    if market_mode in ["crash", "collapse"]:
        return False

    # avoid falling knives in weak market
    if market_mode == "downtrend" and momentum < -0.05:
        return False

    # require minimum strength
    if momentum < -0.12:
        return False

    if volume < 0.2:
        return False

    return True


# -----------------------------
# Strategy selection
# -----------------------------
def generate_signal(symbol, analysis, market_mode):
    momentum = analysis.get("trend_24h", 0)
    rsi = analysis.get("rsi", 50)

    if not should_trade_coin(analysis, market_mode):
        return None

    # Dip buy logic
    if market_mode == "dip_buy" and momentum > -0.03 and rsi < 40:
        return "BUY"

    # Normal trend follow
    if momentum > 0.03 and rsi < 70:
        return "BUY"

    # Exit / avoid
    if momentum < -0.05 or rsi > 75:
        return "SELL"

    return None


# -----------------------------
# Main execution
# -----------------------------
def run_bot():
    log_event(f"Market mode: {market_mode} | bearish={market_bearish}")

    trades = []
    position_scale = position_sizing(market_mode)

    for symbol, analysis in all_analyses.items():
        if not isinstance(analysis, dict):
            continue

        signal = generate_signal(symbol, analysis, market_mode)

        if signal:
            trade = {
                "symbol": symbol,
                "signal": signal,
                "momentum": analysis.get("trend_24h", 0),
                "rsi": analysis.get("rsi", None),
                "position_size": position_scale,
                "timestamp": datetime.utcnow().isoformat()
            }

            trades.append(trade)
            log_event(f"{signal} {symbol} | mom={analysis.get('trend_24h', 0):.3f}")

    # Save trades output for workflow / execution layer
    output = {
        "market_mode": market_mode,
        "trade_count": len(trades),
        "trades": trades,
        "updated_at": datetime.utcnow().isoformat()
    }

    with open("crypto_trades.json", "w") as f:
        json.dump(output, f, indent=2)

    log_event(f"Generated {len(trades)} trades")


if __name__ == "__main__":
    run_bot()
