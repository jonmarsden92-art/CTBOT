# Advanced market health check — adaptive, weighted, and harder to misfire

if 'all_analyses' not in globals() or not isinstance(all_analyses, dict) or not all_analyses:
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
    avg_trend = sum(a.get("trend_24h", 0) for a in valid_analyses) / total_coins if total_coins else 0

market_bearish = False

# --- Decision logic ---
if regime == "crash":
    market_bearish = True
    log.warning("🚨 CRASH REGIME — pausing all buys")

elif bearish_ratio >= 0.8 or avg_trend < -0.06:
    # 80% of market down hard OR overall trend deeply negative
    market_bearish = True
    log.warning(f"📉 MARKET COLLAPSE — {bearish_count}/{total_coins} coins down | avg trend {avg_trend:.2%}")

elif bearish_ratio >= 0.5 and avg_trend < -0.03:
    # Broad but not catastrophic decline → cautious
    market_bearish = True
    log.warning(f"⚠️ HEAVY DOWNTREND — {bearish_count}/{total_coins} coins down | avg trend {avg_trend:.2%}")

elif bearish_ratio >= 0.3:
    # Healthy dip → best opportunity zone
    market_bearish = False
    log.info(f"📉 DIP BUYING MODE — {bearish_count}/{total_coins} coins down | avg trend {avg_trend:.2%}")

else:
    log.info(f"✅ Market healthy — {bearish_count}/{total_coins} coins down | avg trend {avg_trend:.2%} | regime={regime}")
