#!/usr/bin/env python3
"""
crypto_bot.py - Main Trading Bot
================================
Integrates correlation, momentum, patterns, sentiment, predictor, risk, and grid.
Executes trades via Alpaca and updates agent memory.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent))

import alpaca_trade_api as tradeapi
import pandas as pd

from src.correlation import get_correlation_signals
from src.momentum import get_momentum_score
from src.patterns import detect_patterns
from src.sentiment import get_sentiment
from src.risk import (
    load_risk_state, update_risk_state, check_risk,
    get_position_size, get_position_size_multiplier, record_trade_result
)
from src.predictor import (
    load_model, predict_probability, record_training_example,
    run_predictor, load_training_data, save_training_data
)
from src.grid import run_grid, load_grid_state, save_grid_state

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger("crypto_bot")

# ========== CONFIGURATION ==========
SYMBOLS = [
    "BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "DOGE/USD",
    "ADA/USD", "AVAX/USD", "DOT/USD", "LINK/USD", "MATIC/USD",
    "UNI/USD", "ATOM/USD", "LTC/USD", "BCH/USD", "NEAR/USD",
    "ALGO/USD", "VET/USD", "FIL/USD", "ICP/USD", "APT/USD"
]
AGENT_FILE = Path("logs/crypto_agent.json")
GRID_FILE = Path("logs/grid_state.json")
RISK_FILE = Path("logs/risk_state.json")
TRAINING_FILE = Path("logs/training_data.json")
MAX_POSITIONS = 5                     # maximum concurrent positions

def load_agent() -> dict:
    if AGENT_FILE.exists():
        try:
            return json.loads(AGENT_FILE.read_text())
        except:
            pass
    # Default agent with safe thresholds
    return {
        "version": 2,
        "total_learned": 0,
        "wins": 0,
        "losses": 0,
        "win_rate_7d": 0.5,
        "win_rate_30d": 0.5,
        "avg_pnl": 0.0,
        "sharpe": 0.0,
        "thresholds": {
            "take_profit": 0.04,
            "stop_loss": 0.025,
            "trailing_stop": 0.02,
            "signal_min": 5,
            "rsi_oversold": 35,
            "rsi_overbought": 65
        },
        "weights": {},
        "symbol_memory": {},
        "hour_performance": {},
        "regime_performance": {},
        "signal_type_performance": {},
        "open_trades": {},
        "closed_trades": [],
        "calibration_log": [],
        "last_calibrated": datetime.now().isoformat()
    }

def save_agent(agent: dict):
    AGENT_FILE.parent.mkdir(exist_ok=True)
    AGENT_FILE.write_text(json.dumps(agent, indent=2, default=str))

def fetch_bars(api, symbol: str, limit: int = 100) -> pd.DataFrame:
    """Fetch recent bars from Alpaca."""
    try:
        bars = api.get_crypto_bars(symbol, timeframe="1Hour", limit=limit).df
        if bars.empty:
            return pd.DataFrame()
        # Alpaca returns multi-index; filter for the exact symbol
        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.xs(symbol, level="symbol")
        bars = bars.reset_index()
        bars.rename(columns={"timestamp": "time"}, inplace=True)
        bars.set_index("time", inplace=True)
        return bars
    except Exception as e:
        log.error(f"Failed to fetch bars for {symbol}: {e}")
        return pd.DataFrame()

def update_agent_with_trade(agent: dict, trade: dict, won: bool):
    """Update agent statistics after a closed trade."""
    agent["total_learned"] += 1
    if won:
        agent["wins"] += 1
    else:
        agent["losses"] += 1

    # Update win rates
    closed = agent.get("closed_trades", [])
    last_7d = [t for t in closed if t.get("time", "") > (datetime.now() - pd.Timedelta(days=7)).isoformat()]
    last_30d = [t for t in closed if t.get("time", "") > (datetime.now() - pd.Timedelta(days=30)).isoformat()]
    agent["win_rate_7d"] = sum(1 for t in last_7d if t.get("won")) / len(last_7d) if last_7d else 0.5
    agent["win_rate_30d"] = sum(1 for t in last_30d if t.get("won")) / len(last_30d) if last_30d else 0.5

    # Average PnL
    if closed:
        agent["avg_pnl"] = sum(t.get("pnl_pct", 0) for t in closed) / len(closed)

    # Update performance by regime and signal type
    regime = trade.get("regime", "unknown")
    if regime not in agent["regime_performance"]:
        agent["regime_performance"][regime] = {"wins": 0, "losses": 0}
    if won:
        agent["regime_performance"][regime]["wins"] += 1
    else:
        agent["regime_performance"][regime]["losses"] += 1

    for sig in trade.get("signals_fired", []):
        if sig not in agent["signal_type_performance"]:
            agent["signal_type_performance"][sig] = {"wins": 0, "losses": 0, "trades": 0}
        agent["signal_type_performance"][sig]["trades"] += 1
        if won:
            agent["signal_type_performance"][sig]["wins"] += 1
        else:
            agent["signal_type_performance"][sig]["losses"] += 1

def main():
    log.info("Starting Crypto Bot")

    # 1. Load state
    agent = load_agent()
    training_data = load_training_data()
    grid_state = load_grid_state()
    risk_state = load_risk_state()

    # 2. Connect to Alpaca
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    base_url = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    if not api_key or not secret_key:
        log.error("Missing Alpaca API keys. Set ALPACA_API_KEY and ALPACA_SECRET_KEY")
        return

    api = tradeapi.REST(api_key, secret_key, base_url=base_url)

    # 3. Fetch account info
    try:
        account = api.get_account()
        cash = float(account.cash)
        portfolio_value = float(account.portfolio_value)
        log.info(f"Account: cash=${cash:.2f}, portfolio=${portfolio_value:.2f}")
    except Exception as e:
        log.error(f"Failed to get account: {e}")
        return

    # 4. Fetch market data for all symbols
    all_bars = {}
    for symbol in SYMBOLS:
        bars = fetch_bars(api, symbol)
        if not bars.empty:
            all_bars[symbol] = bars
    if "BTC/USD" not in all_bars:
        log.error("BTC/USD data not available – cannot continue")
        return
    btc_bars = all_bars["BTC/USD"]

    # 5. Run market intelligence (sentiment, correlation)
    sentiment = get_sentiment()
    correlation = get_correlation_signals(btc_bars, all_bars)
    btc_regime = correlation.get("btc_regime", {})
    alt_season = correlation.get("alt_season", 0.5)

    # 6. Load ML model if available
    model_tuple = load_model()

    # 7. Update outcomes and retrain predictor
    training_data, model_tuple = run_predictor(training_data, all_bars, model_tuple)

    # 8. Risk check
    risk_check = check_risk(risk_state, portfolio_value, agent.get("win_rate_30d", 0.5), len(agent.get("open_trades", {})))
    if not risk_check["can_trade"]:
        log.warning(f"Risk block: {risk_check['reason']}")
        # Still run grid because grid is low-risk
        grid_state = run_grid(api, all_bars, cash, btc_regime.get("direction", "neutral"), grid_state, agent)
        save_grid_state(grid_state)
        save_agent(agent)
        save_training_data(training_data)
        return

    # 9. Evaluate each symbol for new trades
    open_positions = len(agent.get("open_trades", {}))
    if open_positions < MAX_POSITIONS:
        for symbol, df in all_bars.items():
            if symbol == "BTC/USD":
                continue
            # Skip if already have open trade for this symbol
            if symbol in agent.get("open_trades", {}):
                continue

            # Run analysis modules
            momentum = get_momentum_score(df, btc_bars, symbol)
            patterns = detect_patterns(df)
            # Combine into a single analysis dict
            price = float(df["close"].iloc[-1])
            rsi = 50  # placeholder – would come from a TA module
            vol_ratio = momentum.get("vol_breakout", {}).get("vol_ratio", 1.0)
            signals_fired = momentum.get("signals_fired", []) + patterns.get("patterns", [])

            # Compute rule-based score
            buy_score = momentum.get("momentum_score", 0) * 0.6 + patterns.get("score", 0) * 0.4
            buy_score = max(0, buy_score)

            # Get ML probability
            analysis_for_ml = {
                "rsi": rsi,
                "bb_pct": 0.5,
                "macd_hist": 0.0,
                "trend_1h": momentum.get("btc_leading", {}).get("btc_move_1h", 0),
                "vol_ratio": vol_ratio,
                "vol_strong": vol_ratio > 1.5,
                "stabilising": "stabilising" in signals_fired,
                "near_support": "support_bounce" in signals_fired,
                "rsi_divergence": "bullish" if "rsi_divergence_bull" in signals_fired else "bearish" if "rsi_divergence_bear" in signals_fired else None,
                "atr_pct": 0.02,
            }
            ml_prob = predict_probability(model_tuple, analysis_for_ml, sentiment.get("score", 50))

            # Final score: rule score (0-10) * (0.5 + 0.5*ml_prob)
            final_score = buy_score * (0.5 + ml_prob * 0.5)
            final_score = min(10, final_score)

            if final_score >= agent["thresholds"]["signal_min"]:
                # Apply regime and sentiment multipliers
                if btc_regime.get("direction") == "bull":
                    final_score *= 1.2
                elif btc_regime.get("direction") == "bear":
                    final_score *= 0.6
                final_score *= sentiment.get("multiplier", 1.0)

                if final_score >= agent["thresholds"]["signal_min"]:
                    # Position sizing
                    base_size = get_position_size(api, agent, final_score, cash)
                    mult = get_position_size_multiplier(risk_state, portfolio_value, analysis_for_ml["atr_pct"])
                    size_usd = base_size * mult
                    qty = size_usd / price
                    qty = round(qty, 8) if price > 10000 else round(qty, 6) if price > 1 else round(qty, 2)

                    if qty * price > 1.0:  # minimum trade $1
                        try:
                            order = api.submit_order(
                                symbol=symbol,
                                qty=qty,
                                side="buy",
                                type="market",
                                time_in_force="day"
                            )
                            log.info(f"BUY {symbol} {qty} @ ${price:.4f} (score={final_score:.1f}, ML={ml_prob:.2f})")
                            # Record open trade
                            agent["open_trades"][symbol] = {
                                "entry": price,
                                "qty": qty,
                                "time": datetime.now().isoformat(),
                                "score": final_score,
                                "signals_fired": signals_fired
                            }
                            # Record training example
                            training_data = record_training_example(training_data, symbol, analysis_for_ml, sentiment.get("score", 50))
                        except Exception as e:
                            log.error(f"Buy order failed for {symbol}: {e}")

    # 10. Check existing positions for take profit / stop loss / trailing stop
    to_close = []
    for symbol, trade in agent.get("open_trades", {}).items():
        if symbol not in all_bars:
            continue
        current_price = float(all_bars[symbol]["close"].iloc[-1])
        entry = trade["entry"]
        pnl_pct = (current_price - entry) / entry

        tp = agent["thresholds"]["take_profit"]
        sl = agent["thresholds"]["stop_loss"]

        if pnl_pct >= tp:
            to_close.append((symbol, "take_profit", pnl_pct))
        elif pnl_pct <= -sl:
            to_close.append((symbol, "stop_loss", pnl_pct))
        else:
            # Trailing stop
            if "highest_price" not in trade:
                trade["highest_price"] = entry
            if current_price > trade["highest_price"]:
                trade["highest_price"] = current_price
            trail = agent["thresholds"]["trailing_stop"]
            if (trade["highest_price"] - current_price) / trade["highest_price"] >= trail:
                to_close.append((symbol, "trailing_stop", pnl_pct))

    for symbol, reason, pnl_pct in to_close:
        trade = agent["open_trades"].pop(symbol, None)
        if not trade:
            continue
        qty = trade["qty"]
        try:
            api.submit_order(symbol=symbol, qty=qty, side="sell", type="market", time_in_force="day")
            won = pnl_pct > 0
            log.info(f"SELL {symbol} @ ${current_price:.4f} | PnL={pnl_pct*100:.2f}% | {reason}")
            # Record closed trade
            closed_trade = {
                "symbol": symbol,
                "entry": trade["entry"],
                "exit": current_price,
                "pnl_pct": pnl_pct,
                "won": won,
                "reason": reason,
                "regime": btc_regime.get("direction", "unknown"),
                "score": trade.get("score", 0),
                "signals_fired": trade.get("signals_fired", []),
                "time": datetime.now().isoformat()
            }
            agent["closed_trades"].append(closed_trade)
            update_agent_with_trade(agent, closed_trade, won)
            risk_state = record_trade_result(risk_state, won, pnl_pct)
        except Exception as e:
            log.error(f"Sell order failed for {symbol}: {e}")

    # 11. Run grid trading (low-risk scalping)
    grid_state = run_grid(api, all_bars, cash, btc_regime.get("direction", "neutral"), grid_state, agent)

    # 12. Update risk state with current portfolio value
    try:
        portfolio_value = float(api.get_account().portfolio_value)
    except:
        pass
    risk_state = update_risk_state(risk_state, portfolio_value, None)

    # 13. Save all state
    save_agent(agent)
    save_grid_state(grid_state)
    save_training_data(training_data)
    # risk state saved inside risk functions (load_risk_state/save_risk_state called inside)
    from src.risk import save_risk_state
    save_risk_state(risk_state)

    log.info("Crypto Bot finished successfully")

if __name__ == "__main__":
    main()
