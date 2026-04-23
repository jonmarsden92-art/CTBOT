# Simple market health check - only block in genuine crash or near-total collapse

# Ensure all_analyses exists and is usable
if 'all_analyses' not in globals() or not isinstance(all_analyses, dict):
    log.warning("⚠️ all_analyses not available — skipping market health check")
    bearish_count = 0
else:
    bearish_count = sum(
        1 for a in all_analyses.values()
        if isinstance(a, dict) and a.get("trend_24h", 0) < -0.08
    )

market_bearish = False

# Core logic
if regime == "crash":
    market_bearish = True
    log.warning(f"🚨 CRASH REGIME — pausing all buys")

elif bearish_count >= 16:
    # All or nearly all coins down 8%+ today = genuine market collapse
    market_bearish = True
    log.warning(f"📉 MARKET COLLAPSE ({bearish_count} coins -8%+) — pausing buys")

elif bearish_count >= 10:
    # Aggressive dip buying mode (only if NOT crash)
    log.info(f"📉 DIP BUYING MODE: {bearish_count} coins -8%+ — AGGRESSIVE BUYS ENABLED")
    market_bearish = False  # Explicit override

else:
    log.info(f"✅ Market check OK — {bearish_count} coins down 8%+ | regime={regime}")
