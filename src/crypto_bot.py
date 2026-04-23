     # Simple market health check - only block in genuine crash or near-total collapse
    bearish_count  = sum(1 for a in all_analyses.values() if a.get("trend_24h", 0) < -0.08)
    market_bearish = False

    if regime == "crash":
        market_bearish = True
        log.warning("🚨 CRASH REGIME — pausing all buys")
    elif bearish_count >= 16:
        # All or nearly all coins down 8%+ today = genuine market collapse
        market_bearish = True
        log.warning("📉 MARKET COLLAPSE (" + str(bearish_count) + " coins -8%+) — pausing buys")
    else:
        log.info("✅ Market check OK — " + str(bearish_count) + " coins down 8%+ | regime=" + regime)
    
    # AGGRESSIVE: Buy the dip — if coins are down but not in crash regime, BUY MORE
    if bearish_count >= 10 and regime != "crash":
        log.info("📉 DIP BUYING MODE: " + str(bearish_count) + " coins -8%+ — AGGRESSIVE BUYS ENABLED")
        market_bearish = False  # Override — allow aggressive buying on dips
