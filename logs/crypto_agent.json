"""
AlgoTrader Learning Agent
Self-improving rule-based agent that learns from trade outcomes.

Tracks:
  - Per-symbol win/loss history
  - Signal type performance (RSI, MACD, MA crossover)
  - Time-of-day performance
  - Volatility regime performance
  - Dynamic threshold calibration

Learns by adjusting signal weights and thresholds based on
actual closed trade P&L. Gets smarter with every trade.
"""

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

AGENT_FILE = Path("logs/agent.json")

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_STATE = {
    "version": 2,
    "total_trades_learned": 0,
    "symbol_memory": {},       # {symbol: {wins, losses, total_pnl, avg_pnl, last_seen}}
    "signal_weights": {        # multipliers applied to buy/sell scores
        "ma_cross":   1.0,     # MA crossover signal weight
        "rsi":        1.0,     # RSI signal weight
        "macd":       1.0,     # MACD signal weight
        "bollinger":  1.0,     # Bollinger Band signal weight
        "volume":     1.0,     # Volume boost weight
    },
    "thresholds": {
        "rsi_oversold":    38,  # dynamic RSI oversold level
        "rsi_overbought":  65,  # dynamic RSI overbought level
        "signal_min":       3,  # minimum score to act on
        "min_win_rate":   0.0,  # minimum symbol win rate to trade
    },
    "regime_performance": {    # {regime: {wins, losses}}
        "bull": {"wins": 0, "losses": 0},
        "bear": {"wins": 0, "losses": 0},
        "flat": {"wins": 0, "losses": 0},
    },
    "hour_performance": {},    # {hour: {wins, losses, avg_pnl}}
    "open_trades": {},         # {symbol: {entry_price, entry_time, signals, scores}}
    "closed_trades": [],       # last 200 closed trades with outcomes
    "calibration_log": [],     # log of threshold changes
    "last_calibrated": None,
    "win_rate_7d":    0.5,
    "win_rate_30d":   0.5,
    "avg_pnl_7d":     0.0,
    "sharpe_approx":  0.0,
}


# ── Load / Save ───────────────────────────────────────────────────────────────
def load_agent() -> dict:
    if AGENT_FILE.exists():
        try:
            with open(AGENT_FILE) as f:
                data = json.load(f)
            # Merge with defaults to handle new keys
            for k, v in DEFAULT_STATE.items():
                if k not in data:
                    data[k] = v
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        if kk not in data[k]:
                            data[k][kk] = vv
            return data
        except Exception as e:
            log.warning(f"Agent state load failed: {e} — starting fresh")
    return dict(DEFAULT_STATE)


def save_agent(agent: dict):
    AGENT_FILE.parent.mkdir(exist_ok=True)
    # Keep only last 200 closed trades
    agent["closed_trades"] = agent["closed_trades"][-200:]
    agent["calibration_log"] = agent["calibration_log"][-50:]
    with open(AGENT_FILE, "w") as f:
        json.dump(agent, f, indent=2, default=str)


# ── Record trade open ─────────────────────────────────────────────────────────
def record_open(agent: dict, symbol: str, entry_price: float, indicators: dict):
    """Called when bot buys a position."""
    agent["open_trades"][symbol] = {
        "entry_price":  entry_price,
        "entry_time":   datetime.now().isoformat(),
        "entry_hour":   datetime.now().hour,
        "buy_score":    indicators.get("buy_score", 0),
        "rsi":          indicators.get("rsi", 50),
        "macd_hist":    indicators.get("macd_hist", 0),
        "above_trend":  indicators.get("above_trend", True),
        "vol_ratio":    indicators.get("vol_ratio", 1.0),
        "kelly":        indicators.get("kelly", 0.5),
        "regime":       "bull" if indicators.get("above_trend") else "bear",
    }
    log.info(f"🧠 Agent: recorded open {symbol} @ ${entry_price:.2f}")


# ── Record trade close ────────────────────────────────────────────────────────
def record_close(agent: dict, symbol: str, exit_price: float, reason: str):
    """Called when bot sells a position. Learns from the outcome."""
    if symbol not in agent["open_trades"]:
        return

    trade = agent["open_trades"].pop(symbol)
    entry = trade["entry_price"]
    pnl_pct = (exit_price - entry) / entry * 100
    won = pnl_pct > 0

    duration_str = ""
    try:
        entry_dt = datetime.fromisoformat(trade["entry_time"])
        duration = (datetime.now() - entry_dt).total_seconds() / 3600
        duration_str = f"{duration:.1f}h"
    except Exception:
        pass

    log.info(f"🧠 Agent: closed {symbol} | P&L {pnl_pct:+.2f}% | {reason} | {duration_str}")

    # ── Update symbol memory ──
    sym_mem = agent["symbol_memory"].setdefault(symbol, {
        "wins": 0, "losses": 0, "total_pnl": 0.0,
        "avg_pnl": 0.0, "win_rate": 0.5, "trades": 0, "last_seen": None,
    })
    sym_mem["trades"] += 1
    sym_mem["total_pnl"] += pnl_pct
    sym_mem["avg_pnl"] = sym_mem["total_pnl"] / sym_mem["trades"]
    sym_mem["last_seen"] = datetime.now().isoformat()
    if won:
        sym_mem["wins"] += 1
    else:
        sym_mem["losses"] += 1
    sym_mem["win_rate"] = sym_mem["wins"] / sym_mem["trades"]

    # ── Update hour performance ──
    hour = str(trade.get("entry_hour", datetime.now().hour))
    hour_perf = agent["hour_performance"].setdefault(hour, {"wins": 0, "losses": 0, "total_pnl": 0.0, "trades": 0})
    hour_perf["trades"] += 1
    hour_perf["total_pnl"] += pnl_pct
    if won:
        hour_perf["wins"] += 1
    else:
        hour_perf["losses"] += 1

    # ── Update regime performance ──
    regime = trade.get("regime", "bull")
    reg_perf = agent["regime_performance"].setdefault(regime, {"wins": 0, "losses": 0})
    if won:
        reg_perf["wins"] += 1
    else:
        reg_perf["losses"] += 1

    # ── Store closed trade ──
    agent["closed_trades"].append({
        "symbol":      symbol,
        "entry_price": entry,
        "exit_price":  exit_price,
        "pnl_pct":     pnl_pct,
        "won":         won,
        "reason":      reason,
        "buy_score":   trade.get("buy_score", 0),
        "rsi":         trade.get("rsi", 50),
        "vol_ratio":   trade.get("vol_ratio", 1.0),
        "regime":      regime,
        "time":        datetime.now().isoformat(),
    })

    agent["total_trades_learned"] += 1

    # Calibrate after every 5 closed trades
    if agent["total_trades_learned"] % 5 == 0:
        calibrate(agent)


# ── Calibration ───────────────────────────────────────────────────────────────
def calibrate(agent: dict):
    """
    Analyse recent trade outcomes and adjust signal weights + thresholds.
    Called every 5 trades.
    """
    trades = agent["closed_trades"]
    if len(trades) < 5:
        return

    recent_20  = trades[-20:]
    recent_7   = trades[-7:]

    wins_20    = sum(1 for t in recent_20 if t["won"])
    win_rate_20 = wins_20 / len(recent_20)
    avg_pnl_20  = sum(t["pnl_pct"] for t in recent_20) / len(recent_20)

    wins_7     = sum(1 for t in recent_7 if t["won"])
    win_rate_7  = wins_7 / len(recent_7)

    agent["win_rate_7d"]  = win_rate_7
    agent["win_rate_30d"] = win_rate_20
    agent["avg_pnl_7d"]   = avg_pnl_20

    changes = []

    # ── Adjust RSI thresholds ──
    # If high RSI trades (>60 at entry) are losing → raise overbought threshold
    high_rsi_trades = [t for t in recent_20 if t.get("rsi", 50) > 60]
    if len(high_rsi_trades) >= 3:
        hr_wins = sum(1 for t in high_rsi_trades if t["won"])
        hr_wr   = hr_wins / len(high_rsi_trades)
        if hr_wr < 0.35:
            old = agent["thresholds"]["rsi_overbought"]
            agent["thresholds"]["rsi_overbought"] = max(55, old - 2)
            changes.append(f"RSI overbought {old}→{agent['thresholds']['rsi_overbought']} (high RSI losers)")

    # If low RSI trades (<40 at entry) are winning → lower oversold threshold
    low_rsi_trades = [t for t in recent_20 if t.get("rsi", 50) < 40]
    if len(low_rsi_trades) >= 3:
        lr_wins = sum(1 for t in low_rsi_trades if t["won"])
        lr_wr   = lr_wins / len(low_rsi_trades)
        if lr_wr > 0.65:
            old = agent["thresholds"]["rsi_oversold"]
            agent["thresholds"]["rsi_oversold"] = min(45, old + 2)
            changes.append(f"RSI oversold {old}→{agent['thresholds']['rsi_oversold']} (low RSI winners)")
        elif lr_wr < 0.35:
            old = agent["thresholds"]["rsi_oversold"]
            agent["thresholds"]["rsi_oversold"] = max(28, old - 2)
            changes.append(f"RSI oversold {old}→{agent['thresholds']['rsi_oversold']} (low RSI losers)")

    # ── Adjust signal weights ──
    # MA crossover: check trades where above_trend vs not
    above_trades = [t for t in recent_20 if t.get("regime") == "bull"]
    below_trades = [t for t in recent_20 if t.get("regime") == "bear"]

    if len(above_trades) >= 4:
        above_wr = sum(1 for t in above_trades if t["won"]) / len(above_trades)
        if above_wr > 0.65:
            w = agent["signal_weights"]["ma_cross"]
            agent["signal_weights"]["ma_cross"] = min(1.5, w + 0.05)
            changes.append(f"MA weight ↑ {w:.2f}→{agent['signal_weights']['ma_cross']:.2f}")
        elif above_wr < 0.35:
            w = agent["signal_weights"]["ma_cross"]
            agent["signal_weights"]["ma_cross"] = max(0.5, w - 0.05)
            changes.append(f"MA weight ↓ {w:.2f}→{agent['signal_weights']['ma_cross']:.2f}")

    # Volume: high volume trades winning more?
    high_vol = [t for t in recent_20 if t.get("vol_ratio", 1) > 1.5]
    if len(high_vol) >= 3:
        hv_wr = sum(1 for t in high_vol if t["won"]) / len(high_vol)
        if hv_wr > 0.65:
            w = agent["signal_weights"]["volume"]
            agent["signal_weights"]["volume"] = min(1.8, w + 0.1)
            changes.append(f"Volume weight ↑ {w:.2f}→{agent['signal_weights']['volume']:.2f}")
        elif hv_wr < 0.35:
            w = agent["signal_weights"]["volume"]
            agent["signal_weights"]["volume"] = max(0.6, w - 0.1)
            changes.append(f"Volume weight ↓ {w:.2f}→{agent['signal_weights']['volume']:.2f}")

    # ── Adjust signal minimum threshold based on overall win rate ──
    if win_rate_20 < 0.35 and avg_pnl_20 < -1.0:
        old = agent["thresholds"]["signal_min"]
        agent["thresholds"]["signal_min"] = min(6, old + 1)
        changes.append(f"Signal min ↑ {old}→{agent['thresholds']['signal_min']} (poor win rate)")
    elif win_rate_20 > 0.65 and avg_pnl_20 > 1.0:
        old = agent["thresholds"]["signal_min"]
        agent["thresholds"]["signal_min"] = max(2, old - 1)
        changes.append(f"Signal min ↓ {old}→{agent['thresholds']['signal_min']} (strong win rate)")

    # ── Approximate Sharpe ──
    if len(trades) >= 10:
        pnls  = [t["pnl_pct"] for t in trades[-30:]]
        mean  = sum(pnls) / len(pnls)
        var   = sum((p - mean) ** 2 for p in pnls) / len(pnls)
        std   = math.sqrt(var) if var > 0 else 1
        agent["sharpe_approx"] = mean / std

    agent["last_calibrated"] = datetime.now().isoformat()

    if changes:
        entry = {"time": datetime.now().isoformat(), "changes": changes, "win_rate_20": win_rate_20, "avg_pnl_20": avg_pnl_20}
        agent["calibration_log"].append(entry)
        for c in changes:
            log.info(f"🧠 Agent calibrated: {c}")
    else:
        log.info(f"🧠 Agent: no calibration needed (WR={win_rate_20:.0%}, avg={avg_pnl_20:+.2f}%)")


# ── Score a signal using learned weights ──────────────────────────────────────
def score_signal(agent: dict, symbol: str, indicators: dict) -> dict:
    """
    Apply learned weights to raw indicator scores.
    Returns adjusted buy_score, sell_score, and a confidence value.
    """
    weights   = agent["signal_weights"]
    thresholds = agent["thresholds"]
    sym_mem   = agent["symbol_memory"].get(symbol, {})

    buy_score  = indicators.get("buy_score", 0)
    sell_score = indicators.get("sell_score", 0)

    # Apply signal weights
    ma_w  = weights.get("ma_cross", 1.0)
    rsi_w = weights.get("rsi", 1.0)
    vol_w = weights.get("volume", 1.0)

    # Rough decomposition (re-weight components)
    buy_score  = buy_score  * (0.4 * ma_w + 0.35 * rsi_w + 0.25)
    sell_score = sell_score * (0.4 * ma_w + 0.35 * rsi_w + 0.25)

    # Volume boost with learned weight
    vol_ratio = indicators.get("vol_ratio", 1.0)
    if vol_ratio > 1.5:
        buy_score  *= (1 + (vol_w - 1) * 0.2)
        sell_score *= (1 + (vol_w - 1) * 0.2)

    # Symbol-specific boost/penalty
    if sym_mem.get("trades", 0) >= 3:
        sym_wr = sym_mem.get("win_rate", 0.5)
        if sym_wr > 0.65:
            buy_score  *= 1.15  # boost trusted symbols
        elif sym_wr < 0.35:
            buy_score  *= 0.75  # penalise losing symbols
            log.debug(f"🧠 {symbol}: penalised (WR={sym_wr:.0%})")

    # Dynamic RSI threshold override
    rsi = indicators.get("rsi", 50)
    rsi_oversold   = thresholds.get("rsi_oversold",  38)
    rsi_overbought = thresholds.get("rsi_overbought", 65)

    if rsi < rsi_oversold:
        buy_score  += rsi_w * 2
    elif rsi > rsi_overbought:
        sell_score += rsi_w * 2

    # Confidence: blend of score strength + symbol history + recent win rate
    raw_conf = min(1.0, buy_score / 10.0)
    sym_conf = sym_mem.get("win_rate", 0.5) if sym_mem.get("trades", 0) >= 3 else 0.5
    agent_wr = agent.get("win_rate_7d", 0.5)
    confidence = (raw_conf * 0.5) + (sym_conf * 0.3) + (agent_wr * 0.2)

    return {
        **indicators,
        "buy_score":   round(buy_score, 2),
        "sell_score":  round(sell_score, 2),
        "confidence":  round(confidence, 3),
        "signal": (
            "BUY"  if buy_score  >= thresholds.get("signal_min", 3) else
            "SELL" if sell_score >= thresholds.get("signal_min", 3) else
            "BULLISH" if buy_score > sell_score else
            "BEARISH" if sell_score > buy_score else "HOLD"
        )
    }


# ── Symbol filter ─────────────────────────────────────────────────────────────
def should_trade_symbol(agent: dict, symbol: str) -> bool:
    """Return False for symbols with consistently poor performance."""
    sym_mem = agent["symbol_memory"].get(symbol, {})
    trades  = sym_mem.get("trades", 0)
    if trades < 5:
        return True  # not enough data to block
    win_rate = sym_mem.get("win_rate", 0.5)
    avg_pnl  = sym_mem.get("avg_pnl", 0.0)
    min_wr   = agent["thresholds"].get("min_win_rate", 0.0)
    if win_rate < 0.25 and avg_pnl < -2.0 and trades >= 5:
        log.info(f"🧠 Skipping {symbol} — poor history (WR={win_rate:.0%}, avg={avg_pnl:.2f}%)")
        return False
    return True


# ── Best entry time? ──────────────────────────────────────────────────────────
def get_hour_confidence(agent: dict) -> float:
    """Return a confidence multiplier based on current hour's historical performance."""
    hour = str(datetime.now().hour)
    hour_perf = agent["hour_performance"].get(hour, {})
    trades = hour_perf.get("trades", 0)
    if trades < 3:
        return 1.0  # no data, neutral
    wins = hour_perf.get("wins", 0)
    wr   = wins / trades
    # Scale: 0.4 win rate = 0.85x, 0.6 win rate = 1.15x
    return max(0.7, min(1.3, 0.5 + wr))


# ── Summary log ──────────────────────────────────────────────────────────────
def log_summary(agent: dict):
    """Print agent status to logs."""
    n = agent["total_trades_learned"]
    if n == 0:
        log.info("🧠 Agent: no trades learned yet — using defaults")
        return

    wr7  = agent.get("win_rate_7d",  0.5)
    wr30 = agent.get("win_rate_30d", 0.5)
    pnl  = agent.get("avg_pnl_7d",  0.0)
    sh   = agent.get("sharpe_approx", 0.0)
    thr  = agent["thresholds"]
    wts  = agent["signal_weights"]

    log.info(f"🧠 Agent | Learned={n} trades | WR 7d={wr7:.0%} 30d={wr30:.0%} | AvgPnL={pnl:+.2f}% | Sharpe≈{sh:.2f}")
    log.info(f"🧠 Thresholds | RSI OS={thr['rsi_oversold']} OB={thr['rsi_overbought']} | MinScore={thr['signal_min']}")
    log.info(f"🧠 Weights | MA={wts['ma_cross']:.2f} RSI={wts['rsi']:.2f} Vol={wts['volume']:.2f}")

    # Top symbols
    sym_list = [(s, m) for s, m in agent["symbol_memory"].items() if m.get("trades", 0) >= 3]
    if sym_list:
        best  = sorted(sym_list, key=lambda x: x[1].get("avg_pnl", 0), reverse=True)[:3]
        worst = sorted(sym_list, key=lambda x: x[1].get("avg_pnl", 0))[:3]
        best_str  = ", ".join(s + "(" + str(round(m["avg_pnl"], 1)) + "%)" for s, m in best)
        worst_str = ", ".join(s + "(" + str(round(m["avg_pnl"], 1)) + "%)" for s, m in worst)
        log.info("🧠 Best symbols:  " + best_str)
        log.info("🧠 Worst symbols: " + worst_str)
