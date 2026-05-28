#!/usr/bin/env python3
"""
crypto_bot.py - AGGRESSIVE TRADING (Freqtrade-Inspired)
========================================================
Incorporates Freqtrade's best practices:
- Hyperopt-like parameter optimization
- Better position sizing (Kelly with max cap)
- Multiple confirmation signals
- Tighter risk management
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np

from src.correlation import get_correlation_signals
from src.momentum import get_momentum_score
from src.patterns import detect_patterns
from src.sentiment import get_sentiment
from src.risk import (
    load_risk_state, update_risk_state, check_risk,
    get_position_size, get_position_size_multiplier, record_trade_result,
    save_risk_state
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

# ========== FREQTRADE-INSPIRED SETTINGS ==========
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

# Trading parameters (optimized for aggressive trading)
MAX_POSITIONS = 10           # Freqtrade default
SIGNAL_MIN = 0.1             # Very low threshold - trade on any positive signal
MIN_TRADE_USD = 2.0          # Minimum $2 per trade
MAX_TRADE_PCT = 0.15         # Max 15% of cash per trade

# Risk management (tighter than before)
STOPLOSS = -0.02             # 2% stoploss (Freqtrade conservative)
TRAILING_STOP = True
TRAILING_STOP_PCT = 0.015    # 1.5% trailing
TAKE_PROFIT = 0.03           # 3% take profit

# Technical indicator parameters (Hyperopt-optimized)
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
BB_PERIOD = 20
BB_STD = 2
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9


def load_agent() -> dict:
    """Load or create agent with Freqtrade-inspired thresholds."""
    if AGENT_FILE.exists():
        try:
            agent = json.loads(AGENT_FILE.read_text())
            # Override with aggressive settings
            agent["thresholds"]["signal_min"] = SIGNAL_MIN
            agent["thresholds"]["take_profit"] = TAKE_PROFIT
            agent["thresholds"]["stop_loss"] = abs(STOPLOSS)
            agent["thresholds"]["trailing_stop"] = TRAILING_STOP_PCT
            return agent
        except Exception:
            pass
    
    # Default agent with aggressive settings
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
            "take_profit": TAKE_PROFIT,
            "stop_loss": abs(STOPLOSS),
            "trailing_stop": TRAILING_STOP_PCT,
            "signal_min": SIGNAL_MIN,
            "rsi_oversold": RSI_OVERSOLD,
            "rsi_overbought": RSI_OVERBOUGHT
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
    """Fetch OHLCV bars from Alpaca."""
    try:
        bars = api.get_crypto_bars(symbol, timeframe="1Hour", limit=limit).df
        if bars.empty:
            return pd.DataFrame()
        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.xs(symbol, level="symbol")
        bars = bars.reset_index()
        bars.rename(columns={"timestamp": "time"}, inplace=True)
        bars.set_index("time", inplace=True)
        return bars
    except Exception as e:
        log.error(f"Failed to fetch bars for {symbol}: {e}")
        return pd.DataFrame()


def calculate_rsi(series: pd.Series, period: int = 14) -> float:
    """Calculate RSI with fallback."""
    if len(series) < period + 1:
        return 50.0
    try:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        val = float(rsi.iloc[-1])
        return val if not pd.isna(val) else 50.0
    except Exception:
        return 50.0


def calculate_macd(close: pd.Series) -> Dict[str, float]:
    """Calculate MACD with fallback."""
    if len(close) < MACD_SLOW:
        return {"macd": 0.0, "signal": 0.0, "hist": 0.0}
    try:
        exp1 = close.ewm(span=MACD_FAST, adjust=False).mean()
        exp2 = close.ewm(span=MACD_SLOW, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=MACD_SIGNAL, adjust=False).mean()
        hist = macd - signal
        return {
            "macd": float(macd.iloc[-1]),
            "signal": float(signal.iloc[-1]),
            "hist": float(hist.iloc[-1])
        }
    except Exception:
        return {"macd": 0.0, "signal": 0.0, "hist": 0.0}


def calculate_bollinger_bands(close: pd.Series) -> Dict[str, float]:
    """Calculate Bollinger Bands with fallback."""
    if len(close) < BB_PERIOD:
        return {"upper": 0, "lower": 0, "mid": 0, "pct": 0.5, "width": 0}
    try:
        sma = close.rolling(window=BB_PERIOD).mean()
        std = close.rolling(window=BB_PERIOD).std()
        upper = sma + (std * BB_STD)
        lower = sma - (std * BB_STD)
        pct = (close.iloc[-1] - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1]) if (upper.iloc[-1] - lower.iloc[-1]) != 0 else 0.5
        width = (upper.iloc[-1] - lower.iloc[-1]) / sma.iloc[-1] if sma.iloc[-1] != 0 else 0
        return {
            "upper": float(upper.iloc[-1]),
            "lower": float(lower.iloc[-1]),
            "mid": float(sma.iloc[-1]),
            "pct": float(pct) if not pd.isna(pct) else 0.5,
            "width": float(width) if not pd.isna(width) else 0
        }
    except Exception:
        return {"upper": 0, "lower": 0, "mid": 0, "pct": 0.5, "width": 0}


def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Calculate ATR percentage."""
    try:
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        atr_pct = atr.iloc[-1] / close.iloc[-1] if close.iloc[-1] != 0 else 0.02
        return float(atr_pct) if not pd.isna(atr_pct) else 0.02
    except Exception:
        return 0.02


def get_final_score(symbol: str, df: pd.DataFrame, btc_bars: pd.DataFrame,
                    sentiment: dict, btc_regime: dict, alt_season: float,
                    model_tuple: tuple) -> tuple:
    """
    Calculate final trading score using Freqtrade-inspired multi-factor model.
    Returns (final_score, signals_fired, analysis_for_ml)
    """
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)
    
    # Technical indicators
    rsi = calculate_rsi(close)
    macd = calculate_macd(close)
    bb = calculate_bollinger_bands(close)
    atr_pct = calculate_atr(df)
    
    # Volume analysis
    avg_volume = volume.rolling(20).mean().iloc[-1]
    vol_ratio = volume.iloc[-1] / avg_volume if avg_volume > 0 else 1.0
    
    # Momentum and patterns
    momentum = get_momentum_score(df, btc_bars, symbol)
    patterns = detect_patterns(df)
    signals_fired = momentum.get("signals_fired", []) + patterns.get("patterns", [])
    
    # ---------- Freqtrade-inspired scoring ----------
    score = 0.0
    
    # 1. RSI score (oversold = bullish)
    if rsi < RSI_OVERSOLD:
        score += 3.0
    elif rsi < 40:
        score += 1.0
    elif rsi > RSI_OVERBOUGHT:
        score -= 2.0
    elif rsi > 60:
        score -= 1.0
    
    # 2. Bollinger Bands score (lower band bounce)
    if bb["pct"] < 0.1:
        score += 2.0
    elif bb["pct"] < 0.2:
        score += 1.0
    elif bb["pct"] > 0.9:
        score -= 1.0
    
    # 3. MACD score (histogram positive and increasing)
    if macd["hist"] > 0:
        score += 1.0
        if macd["hist"] > macd["hist"] * 1.1:  # Increasing
            score += 0.5
    elif macd["hist"] < 0 and macd["hist"] < macd["hist"] * 0.9:
        score -= 1.0
    
    # 4. Volume score
    if vol_ratio > 2.0:
        score += 1.5
    elif vol_ratio > 1.5:
        score += 0.5
    elif vol_ratio < 0.5:
        score -= 0.5
    
    # 5. Pattern score (boosted for strong patterns)
    pattern_score = patterns.get("score", 0)
    if "morning_star" in patterns.get("patterns", []):
        pattern_score += 3
    if "bullish_engulfing" in patterns.get("patterns", []):
        pattern_score += 2
    if "hammer" in patterns.get("patterns", []):
        pattern_score += 1
    score += pattern_score * 0.5
    
    # 6. Momentum score
    score += momentum.get("momentum_score", 0) * 0.3
    
    # 7. Market regime multipliers (Freqtrade-style)
    if btc_regime.get("direction") in ["bull", "strong_bull"]:
        score *= 1.3
    elif btc_regime.get("direction") in ["bear", "strong_bear"]:
        score *= 0.5
    
    if alt_season > 0.6:
        score *= 1.2
    
    # Sentiment multiplier (extreme fear = buying opportunity)
    score *= sentiment.get("multiplier", 1.0)
    
    # ML probability boost (if model exists)
    analysis_for_ml = {
        "price": float(close.iloc[-1]),
        "rsi": rsi,
        "bb_pct": bb["pct"],
        "macd_hist": macd["hist"],
        "trend_1h": momentum.get("btc_leading", {}).get("btc_move_1h", 0),
        "vol_ratio": vol_ratio,
        "vol_strong": vol_ratio > 1.5,
        "stabilising": "stabilising" in signals_fired,
        "near_support": "support_bounce" in signals_fired,
        "rsi_divergence": "bullish" if "rsi_divergence_bull" in signals_fired else "bearish" if "rsi_divergence_bear" in signals_fired else None,
        "atr_pct": atr_pct,
    }
    
    ml_prob = predict_probability(model_tuple, analysis_for_ml, sentiment.get("score", 50))
    if model_tuple is not None:
        score = score * (0.5 + ml_prob * 0.5)
    
    final_score = max(0, min(10, score))
    
    # Detailed logging
    log.info(f"📊 {symbol} | rsi={rsi:.1f} | bb={bb['pct']:.2f} | macd={macd['hist']:.3f} | "
             f"vol={vol_ratio:.1f} | patterns={pattern_score:.1f} | final={final_score:.2f} | "
             f"signals={signals_fired[:3]}")
    
    return final_score, signals_fired, analysis_for_ml


def main():
    log.info("Starting Crypto Bot (Freqtrade-Inspired Aggressive Mode)")
    
    # Load state
    agent = load_agent()
    training_data = load_training_data()
    grid_state = load_grid_state()
    risk_state = load_risk_state()
    
    # Connect to Alpaca
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    base_url = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    
    if not api_key or not secret_key:
        log.error("Missing Alpaca API keys")
        return
    
    api = tradeapi.REST(api_key, secret_key, base_url=base_url)
    
    try:
        account = api.get_account()
        cash = float(account.cash)
        portfolio_value = float(account.portfolio_value)
        log.info(f"💰 Account: cash=${cash:.2f}, portfolio=${portfolio_value:.2f}")
    except Exception as e:
        log.error(f"Failed to get account: {e}")
        return
    
    # Fetch market data
    all_bars = {}
    for symbol in SYMBOLS:
        bars = fetch_bars(api, symbol)
        if not bars.empty:
            all_bars[symbol] = bars
    
    if "BTC/USD" not in all_bars:
        log.error("BTC/USD data missing")
        return
    btc_bars = all_bars["BTC/USD"]
    
    # Market context
    sentiment = get_sentiment()
    correlation = get_correlation_signals(btc_bars, all_bars)
    btc_regime = correlation.get("btc_regime", {})
    alt_season = correlation.get("alt_season", 0.5)
    
    # ML predictor
    model_tuple = load_model()
    training_data, model_tuple = run_predictor(training_data, all_bars, model_tuple)
    
    # Risk check
    risk_check = check_risk(risk_state, portfolio_value, agent.get("win_rate_30d", 0.5),
                            len(agent.get("open_trades", {})))
    if not risk_check["can_trade"]:
        log.warning(f"🛑 Risk block: {risk_check['reason']}")
        grid_state = run_grid(api, all_bars, cash, btc_regime.get("direction", "neutral"), grid_state, agent)
        save_grid_state(grid_state)
        save_agent(agent)
        save_training_data(training_data)
        return
    
    # Check for new trades
    open_positions = len(agent.get("open_trades", {}))
    log.info(f"📊 Open positions: {open_positions}/{MAX_POSITIONS}")
    
    if open_positions < MAX_POSITIONS:
        for symbol, df in all_bars.items():
            if symbol == "BTC/USD":
                continue
            
            symbol_key = symbol.replace("/", "")
            if symbol_key in agent.get("open_trades", {}):
                continue
            
            # Calculate score
            final_score, signals_fired, analysis_for_ml = get_final_score(
                symbol, df, btc_bars, sentiment, btc_regime, alt_season, model_tuple
            )
            
            # Entry signal
            if final_score >= agent["thresholds"]["signal_min"]:
                price = float(df["close"].astype(float).iloc[-1])
                
                # Position sizing (Kelly-inspired with max cap)
                size_usd = max(MIN_TRADE_USD, cash * 0.10)
                size_usd = min(size_usd, cash * MAX_TRADE_PCT, cash * (final_score / 10))
                
                qty = size_usd / price
                qty = round(qty, 8) if price > 10000 else round(qty, 6) if price > 1 else round(qty, 4)
                
                if qty * price >= MIN_TRADE_USD:
                    log.info(f"🔍 BUY SIGNAL: {symbol} | ${size_usd:.2f} @ ${price:.4f} = {qty} | score={final_score:.2f}")
                    try:
                        api.submit_order(
                            symbol=symbol, qty=qty, side="buy",
                            type="market", time_in_force="gtc"
                        )
                        log.info(f"✅ BUY EXECUTED: {symbol} {qty} @ ${price:.4f}")
                        
                        agent["open_trades"][symbol_key] = {
                            "entry": price,
                            "qty": qty,
                            "time": datetime.now().isoformat(),
                            "score": final_score,
                            "signals_fired": signals_fired,
                            "highest_price": price
                        }
                        cash -= qty * price
                        training_data = record_training_example(training_data, symbol_key, analysis_for_ml, sentiment.get("score", 50))
                    except Exception as e:
                        log.error(f"Buy order failed for {symbol}: {e}")
                else:
                    log.info(f"❌ BUY SKIPPED {symbol}: ${qty * price:.2f} < ${MIN_TRADE_USD}")
    else:
        log.info(f"Max positions reached ({MAX_POSITIONS})")
    
    # Check existing positions for exit signals
    to_close = []
    for symbol_key, trade in agent.get("open_trades", {}).items():
        symbol = f"{symbol_key[:3]}/{symbol_key[3:]}" if len(symbol_key) > 3 else symbol_key
        if symbol not in all_bars:
            continue
        
        current_price = float(all_bars[symbol]["close"].astype(float).iloc[-1])
        entry = trade["entry"]
        pnl_pct = (current_price - entry) / entry
        
        # Exit conditions
        if pnl_pct >= TAKE_PROFIT:
            to_close.append((symbol_key, "take_profit", pnl_pct, current_price))
        elif pnl_pct <= STOPLOSS:
            to_close.append((symbol_key, "stop_loss", pnl_pct, current_price))
        else:
            # Trailing stop
            if current_price > trade.get("highest_price", entry):
                trade["highest_price"] = current_price
            drawdown = (trade["highest_price"] - current_price) / trade["highest_price"]
            if drawdown >= TRAILING_STOP_PCT:
                to_close.append((symbol_key, "trailing_stop", pnl_pct, current_price))
    
    # Execute exits
    for symbol_key, reason, pnl_pct, exit_price in to_close:
        trade = agent["open_trades"].pop(symbol_key, None)
        if not trade:
            continue
        
        qty = trade["qty"]
        symbol = f"{symbol_key[:3]}/{symbol_key[3:]}" if len(symbol_key) > 3 else symbol_key
        
        try:
            api.submit_order(symbol=symbol, qty=qty, side="sell", type="market", time_in_force="gtc")
            won = pnl_pct > 0
            log.info(f"💰 SELL {symbol} @ ${exit_price:.4f} | PnL={pnl_pct*100:.2f}% | {reason}")
            
            closed_trade = {
                "symbol": symbol_key,
                "entry": trade["entry"],
                "exit": exit_price,
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
            cash += qty * exit_price
        except Exception as e:
            log.error(f"Sell order failed for {symbol}: {e}")
    
    # Run grid strategy
    grid_state = run_grid(api, all_bars, cash, btc_regime.get("direction", "neutral"), grid_state, agent)
    
    # Update final state
    try:
        portfolio_value = float(api.get_account().portfolio_value)
    except Exception:
        pass
    
    risk_state = update_risk_state(risk_state, portfolio_value, None)
    save_agent(agent)
    save_grid_state(grid_state)
    save_training_data(training_data)
    save_risk_state(risk_state)
    
    log.info(f"✅ Bot finished | Open: {len(agent.get('open_trades', {}))} | Total trades: {len(agent.get('closed_trades', []))}")


def update_agent_with_trade(agent: dict, trade: dict, won: bool):
    """Update agent statistics after a trade."""
    agent["total_learned"] += 1
    if won:
        agent["wins"] += 1
    else:
        agent["losses"] += 1
    
    # Update win rates
    closed = agent.get("closed_trades", [])
    now = datetime.now()
    last_7d = [t for t in closed if t.get("time", "") > (now - pd.Timedelta(days=7)).isoformat()]
    last_30d = [t for t in closed if t.get("time", "") > (now - pd.Timedelta(days=30)).isoformat()]
    agent["win_rate_7d"] = sum(1 for t in last_7d if t.get("won")) / len(last_7d) if last_7d else 0.5
    agent["win_rate_30d"] = sum(1 for t in last_30d if t.get("won")) / len(last_30d) if last_30d else 0.5
    
    if closed:
        agent["avg_pnl"] = sum(t.get("pnl_pct", 0) for t in closed) / len(closed)
    
    # Update performance by regime
    regime = trade.get("regime", "unknown")
    if regime not in agent["regime_performance"]:
        agent["regime_performance"][regime] = {"wins": 0, "losses": 0}
    if won:
        agent["regime_performance"][regime]["wins"] += 1
    else:
        agent["regime_performance"][regime]["losses"] += 1
    
    # Update signal performance
    for sig in trade.get("signals_fired", []):
        if sig not in agent["signal_type_performance"]:
            agent["signal_type_performance"][sig] = {"wins": 0, "losses": 0, "trades": 0}
        agent["signal_type_performance"][sig]["trades"] += 1
        if won:
            agent["signal_type_performance"][sig]["wins"] += 1
        else:
            agent["signal_type_performance"][sig]["losses"] += 1


if __name__ == "__main__":
    main()
