"""
CryptoTrader v3 - 24/7 Smart Crypto Trading Bot
Built-in learning agent that adapts every 5 trades
Trades 14 crypto pairs on Alpaca, runs every 10 minutes
"""

import os
import json
import math
import logging
import warnings
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict
from pathlib import Path

import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# Logging
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/crypto_bot.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)
for noisy in ("urllib3", "requests", "alpaca_trade_api"):
    logging.getLogger(noisy).setLevel(logging.CRITICAL)

# Alpaca credentials
ALPACA_API_KEY    = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]
ALPACA_BASE_URL   = os.environ.get("ALPACA_BASE_URL", "https://api.alpaca.markets")

# Crypto pairs
CRYPTO_UNIVERSE = [
    "BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD",
    "AVAX/USD", "LINK/USD", "XRP/USD", "LTC/USD",
    "BCH/USD", "DOT/USD", "UNI/USD", "AAVE/USD",
    "SHIB/USD", "PEPE/USD",
]

# Strategy config (agent tunes these over time)
SHORT_WINDOW          = 3
LONG_WINDOW           = 9
RSI_PERIOD            = 5
MAX_POSITIONS         = 8
POSITION_SIZE         = 0.12
STOP_LOSS_PCT         = 0.03
TAKE_PROFIT_PCT       = 0.02
TRAILING_STOP_PCT     = 0.02
TRAILING_ACTIVATE_PCT = 0.008
MIN_CASH_BUFFER       = 0.005
MIN_ORDER_USD         = 0.50
DEFAULT_RSI_OS        = 45
DEFAULT_RSI_OB        = 55
DEFAULT_SIG_MIN       = 1

STATE_FILE  = Path("logs/crypto_state.json")
AGENT_FILE  = Path("logs/crypto_agent.json")
REPORT_FILE = Path("logs/crypto_report.json")


# =============================================================================
# LEARNING AGENT
# =============================================================================

DEFAULT_AGENT = {
    "version": 1,
    "total_learned": 0,
    "wins": 0,
    "losses": 0,
    "win_rate_7d": 0.5,
    "win_rate_30d": 0.5,
    "avg_pnl": 0.0,
    "sharpe": 0.0,
    "thresholds": {
        "rsi_oversold":   45,
        "rsi_overbought": 55,
        "signal_min":     1,
        "take_profit":    0.02,
        "stop_loss":      0.03,
        "trailing_stop":  0.02,
    },
    "weights": {
        "ma_cross": 1.0,
        "rsi":      1.0,
        "macd":     1.0,
        "volume":   1.0,
    },
    "symbol_memory":     {},
    "hour_performance":  {},
    "open_trades":       {},
    "closed_trades":     [],
    "calibration_log":   [],
    "last_calibrated":   None,
}


def load_agent() -> dict:
    if AGENT_FILE.exists():
        try:
            data = json.loads(AGENT_FILE.read_text())
            for k, v in DEFAULT_AGENT.items():
                if k not in data:
                    data[k] = v
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        if kk not in data[k]:
                            data[k][kk] = vv
            return data
        except Exception as e:
            log.warning("Agent load failed: " + str(e))
    return dict(DEFAULT_AGENT)


def save_agent(agent: dict):
    agent["closed_trades"]   = agent["closed_trades"][-200:]
    agent["calibration_log"] = agent["calibration_log"][-50:]
    AGENT_FILE.write_text(json.dumps(agent, indent=2, default=str))


def agent_record_open(agent: dict, symbol: str, price: float, indicators: dict):
    agent["open_trades"][symbol] = {
        "entry_price": price,
        "entry_time":  datetime.now().isoformat(),
        "entry_hour":  datetime.now().hour,
        "buy_score":   indicators.get("buy_score", 0),
        "rsi":         indicators.get("rsi", 50),
        "vol_ratio":   indicators.get("vol_ratio", 1.0),
    }
    log.info("🧠 Agent: opened " + symbol + " @ $" + str(round(price, 4)))


def agent_record_close(agent: dict, symbol: str, exit_price: float, reason: str):
    if symbol not in agent["open_trades"]:
        return
    trade   = agent["open_trades"].pop(symbol)
    entry   = trade["entry_price"]
    pnl_pct = (exit_price - entry) / entry * 100
    won     = pnl_pct > 0

    mem = agent["symbol_memory"].setdefault(symbol, {
        "wins": 0, "losses": 0, "total_pnl": 0.0,
        "avg_pnl": 0.0, "win_rate": 0.5, "trades": 0,
    })
    mem["trades"]    += 1
    mem["total_pnl"] += pnl_pct
    mem["avg_pnl"]    = mem["total_pnl"] / mem["trades"]
    if won:
        mem["wins"]   += 1
        agent["wins"] += 1
    else:
        mem["losses"]   += 1
        agent["losses"] += 1
    mem["win_rate"] = mem["wins"] / mem["trades"]

    hour = str(trade.get("entry_hour", datetime.now().hour))
    hp   = agent["hour_performance"].setdefault(hour, {
        "wins": 0, "losses": 0, "total_pnl": 0.0, "trades": 0
    })
    hp["trades"]    += 1
    hp["total_pnl"] += pnl_pct
    if won:
        hp["wins"] += 1
    else:
        hp["losses"] += 1

    agent["closed_trades"].append({
        "symbol":     symbol,
        "entry":      entry,
        "exit":       exit_price,
        "pnl_pct":    pnl_pct,
        "won":        won,
        "reason":     reason,
        "buy_score":  trade.get("buy_score", 0),
        "rsi":        trade.get("rsi", 50),
        "vol_ratio":  trade.get("vol_ratio", 1.0),
        "time":       datetime.now().isoformat(),
    })
    agent["total_learned"] += 1
    log.info("🧠 Closed " + symbol + " PnL=" + str(round(pnl_pct, 2)) + "% | " + reason)

    if agent["total_learned"] % 5 == 0:
        agent_calibrate(agent)


def agent_calibrate(agent: dict):
    trades = agent["closed_trades"]
    if len(trades) < 5:
        return

    r20     = trades[-20:]
    r7      = trades[-7:]
    wr20    = sum(1 for t in r20 if t["won"]) / len(r20)
    wr7     = sum(1 for t in r7  if t["won"]) / len(r7)
    avg_pnl = sum(t["pnl_pct"] for t in r20) / len(r20)

    agent["win_rate_7d"]  = wr7
    agent["win_rate_30d"] = wr20
    agent["avg_pnl"]      = avg_pnl

    thr     = agent["thresholds"]
    wts     = agent["weights"]
    changes = []

    # RSI overbought
    hi_rsi = [t for t in r20 if t.get("rsi", 50) > 60]
    if len(hi_rsi) >= 3:
        if sum(1 for t in hi_rsi if t["won"]) / len(hi_rsi) < 0.35:
            old = thr["rsi_overbought"]
            thr["rsi_overbought"] = max(55, old - 2)
            changes.append("RSI OB " + str(old) + "->" + str(thr["rsi_overbought"]))

    # RSI oversold
    lo_rsi = [t for t in r20 if t.get("rsi", 50) < 40]
    if len(lo_rsi) >= 3:
        lr_wr = sum(1 for t in lo_rsi if t["won"]) / len(lo_rsi)
        old   = thr["rsi_oversold"]
        if lr_wr > 0.65:
            thr["rsi_oversold"] = min(42, old + 2)
            changes.append("RSI OS " + str(old) + "->" + str(thr["rsi_oversold"]) + " (winning)")
        elif lr_wr < 0.35:
            thr["rsi_oversold"] = max(25, old - 2)
            changes.append("RSI OS " + str(old) + "->" + str(thr["rsi_oversold"]) + " (losing)")

    # Signal minimum
    if wr20 < 0.35 and avg_pnl < -1.0:
        old = thr["signal_min"]
        thr["signal_min"] = min(7, old + 1)
        changes.append("SigMin " + str(old) + "->" + str(thr["signal_min"]) + " (poor WR)")
    elif wr20 > 0.65 and avg_pnl > 1.0:
        old = thr["signal_min"]
        thr["signal_min"] = max(2, old - 1)
        changes.append("SigMin " + str(old) + "->" + str(thr["signal_min"]) + " (strong WR)")

    # Take profit
    if wr20 > 0.60 and avg_pnl > 1.5:
        old = thr["take_profit"]
        thr["take_profit"] = min(0.08, round(old + 0.005, 3))
        changes.append("TP " + str(old) + "->" + str(thr["take_profit"]) + " (let it run)")
    elif wr20 < 0.40:
        old = thr["take_profit"]
        thr["take_profit"] = max(0.02, round(old - 0.005, 3))
        changes.append("TP " + str(old) + "->" + str(thr["take_profit"]) + " (bank faster)")

    # Volume weight
    hi_vol = [t for t in r20 if t.get("vol_ratio", 1) > 1.5]
    if len(hi_vol) >= 3:
        hv_wr = sum(1 for t in hi_vol if t["won"]) / len(hi_vol)
        if hv_wr > 0.65:
            wts["volume"] = min(1.8, round(wts["volume"] + 0.1, 2))
            changes.append("Vol wt->" + str(wts["volume"]) + " (vol trades winning)")
        elif hv_wr < 0.35:
            wts["volume"] = max(0.5, round(wts["volume"] - 0.1, 2))
            changes.append("Vol wt->" + str(wts["volume"]) + " (vol trades losing)")

    # Sharpe
    if len(trades) >= 10:
        pnls  = [t["pnl_pct"] for t in trades[-30:]]
        mean  = sum(pnls) / len(pnls)
        var   = sum((p - mean) ** 2 for p in pnls) / len(pnls)
        std   = math.sqrt(var) if var > 0 else 1
        agent["sharpe"] = round(mean / std, 3)

    agent["last_calibrated"] = datetime.now().isoformat()

    if changes:
        agent["calibration_log"].append({
            "time":    datetime.now().isoformat(),
            "changes": changes,
            "wr20":    round(wr20, 3),
            "avg_pnl": round(avg_pnl, 3),
        })
        for c in changes:
            log.info("🧠 Calibrated: " + c)
    else:
        log.info("🧠 No calibration needed WR=" + str(round(wr20 * 100)) + "% avg=" + str(round(avg_pnl, 2)) + "%")


def agent_score(agent: dict, symbol: str, indicators: dict) -> dict:
    wts       = agent["weights"]
    thr       = agent["thresholds"]
    mem       = agent["symbol_memory"].get(symbol, {})
    buy_score = indicators.get("buy_score",  0)
    sell_score = indicators.get("sell_score", 0)
    rsi       = indicators.get("rsi", 50)
    vol_ratio = indicators.get("vol_ratio", 1.0)

    ma_w  = wts.get("ma_cross", 1.0)
    rsi_w = wts.get("rsi",      1.0)
    mac_w = wts.get("macd",     1.0)
    vol_w = wts.get("volume",   1.0)

    factor     = 0.35 * ma_w + 0.30 * rsi_w + 0.20 * mac_w + 0.15
    buy_score  = buy_score  * factor
    sell_score = sell_score * factor

    if vol_ratio > 1.5:
        boost      = (vol_w - 1.0) * 0.25
        buy_score  = buy_score  * (1 + boost)
        sell_score = sell_score * (1 + boost)

    if rsi < thr["rsi_oversold"]:
        buy_score  += rsi_w * 2
    elif rsi > thr["rsi_overbought"]:
        sell_score += rsi_w * 2

    if mem.get("trades", 0) >= 3:
        sym_wr = mem.get("win_rate", 0.5)
        if sym_wr > 0.65:
            buy_score = buy_score * 1.2
        elif sym_wr < 0.30:
            buy_score = buy_score * 0.7

    hour  = str(datetime.now().hour)
    hperf = agent["hour_performance"].get(hour, {})
    if hperf.get("trades", 0) >= 3:
        hour_wr   = hperf["wins"] / hperf["trades"]
        buy_score = buy_score * max(0.7, min(1.3, 0.5 + hour_wr))

    sig_min    = thr.get("signal_min", DEFAULT_SIG_MIN)
    signal     = (
        "BUY"     if buy_score  >= sig_min else
        "SELL"    if sell_score >= sig_min else
        "BULLISH" if buy_score  >  sell_score else
        "BEARISH" if sell_score >  buy_score else "HOLD"
    )
    confidence = min(1.0, (buy_score / 10.0) * 0.6 + agent.get("win_rate_7d", 0.5) * 0.4)

    return {
        **indicators,
        "buy_score":  round(buy_score,  2),
        "sell_score": round(sell_score, 2),
        "confidence": round(confidence, 3),
        "signal":     signal,
    }


def agent_should_trade(agent: dict, symbol: str) -> bool:
    mem = agent["symbol_memory"].get(symbol, {})
    if (mem.get("trades", 0) >= 5
            and mem.get("win_rate", 0.5) < 0.25
            and mem.get("avg_pnl",  0.0) < -2.0):
        log.info("🧠 Skipping " + symbol + " WR=" + str(round(mem["win_rate"] * 100)) + "%")
        return False
    return True


def agent_log_summary(agent: dict):
    n = agent["total_learned"]
    if n == 0:
        log.info("🧠 Agent: no trades learned yet — using defaults")
        return
    thr = agent["thresholds"]
    wts = agent["weights"]
    log.info("🧠 Agent | Learned=" + str(n) +
             " | WR 7d=" + str(round(agent["win_rate_7d"] * 100)) + "%" +
             " 30d=" + str(round(agent["win_rate_30d"] * 100)) + "%" +
             " | AvgPnL=" + str(round(agent["avg_pnl"], 2)) + "%" +
             " | Sharpe=" + str(agent["sharpe"]))
    log.info("🧠 Thresholds | RSI OS=" + str(thr["rsi_oversold"]) +
             " OB=" + str(thr["rsi_overbought"]) +
             " | SigMin=" + str(thr["signal_min"]) +
             " | TP=" + str(round(thr["take_profit"] * 100, 1)) + "%" +
             " SL=" + str(round(thr["stop_loss"] * 100, 1)) + "%")
    log.info("🧠 Weights | MA=" + str(wts["ma_cross"]) +
             " RSI=" + str(wts["rsi"]) +
             " Vol=" + str(wts["volume"]))

    syms = [(s, m) for s, m in agent["symbol_memory"].items() if m.get("trades", 0) >= 2]
    if syms:
        best  = sorted(syms, key=lambda x: x[1]["avg_pnl"], reverse=True)[:3]
        worst = sorted(syms, key=lambda x: x[1]["avg_pnl"])[:3]
        log.info("🧠 Best:  " + " | ".join(s + " " + str(round(m["avg_pnl"], 1)) + "%" for s, m in best))
        log.info("🧠 Worst: " + " | ".join(s + " " + str(round(m["avg_pnl"], 1)) + "%" for s, m in worst))


# =============================================================================
# STATE
# =============================================================================

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {
        "trades": [], "signals": {}, "last_run": None,
        "daily_trades": 0, "trade_date": None,
        "peak_prices": {}, "total_trades": 0,
        "wins": 0, "losses": 0,
    }


def save_state(state: dict):
    state["trades"] = state["trades"][-500:]
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


# =============================================================================
# MARKET DATA
# =============================================================================

def fetch_bars(api, symbols: List[str]) -> Dict[str, pd.DataFrame]:
    result = {}
    end    = datetime.now(timezone.utc)
    start  = end - timedelta(days=35)
    for sym in symbols:
        try:
            bars = api.get_crypto_bars(
                sym,
                tradeapi.rest.TimeFrame.Hour,
                start=start.isoformat(),
                end=end.isoformat(),
            ).df
            if bars is not None and not bars.empty and len(bars) >= LONG_WINDOW + 5:
                result[sym] = bars
        except Exception as e:
            log.debug("Bars " + sym + ": " + str(e))
    return result


def compute_indicators(df: pd.DataFrame, agent: dict) -> Optional[dict]:
    try:
        close = df["close"].squeeze().astype(float)
        vol   = df["volume"].squeeze().astype(float)
        if len(close) < LONG_WINDOW + 5:
            return None

        thr    = agent["thresholds"]
        sma_s  = close.rolling(SHORT_WINDOW).mean()
        sma_l  = close.rolling(LONG_WINDOW).mean()
        delta  = close.diff()
        gain   = delta.clip(lower=0).rolling(RSI_PERIOD).mean()
        loss   = (-delta.clip(upper=0)).rolling(RSI_PERIOD).mean()
        rsi    = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
        ema12  = close.ewm(span=12, adjust=False).mean()
        ema26  = close.ewm(span=26, adjust=False).mean()
        macd   = ema12 - ema26
        macd_s = macd.ewm(span=9, adjust=False).mean()
        hist   = macd - macd_s
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_low = bb_mid - 2 * bb_std
        bb_up  = bb_mid + 2 * bb_std
        avg_v  = vol.rolling(20).mean()

        price    = float(close.iloc[-1])
        ss       = float(sma_s.iloc[-1]); ss_p = float(sma_s.iloc[-2])
        sl       = float(sma_l.iloc[-1]); sl_p = float(sma_l.iloc[-2])
        rsi_v    = float(rsi.iloc[-1])
        hist_v   = float(hist.iloc[-1]); hist_p = float(hist.iloc[-2])
        macd_v   = float(macd.iloc[-1]); msig_v = float(macd_s.iloc[-1])
        bbl      = float(bb_low.iloc[-1])
        bbu      = float(bb_up.iloc[-1])
        vol_r    = float(vol.iloc[-1] / avg_v.iloc[-1]) if float(avg_v.iloc[-1]) > 0 else 1.0

        buy = sell = 0

        if ss_p <= sl_p and ss > sl: buy  += 3
        if ss_p >= sl_p and ss < sl: sell += 3
        ma_gap = (ss - sl) / sl if sl > 0 else 0
        if ma_gap >  0.002: buy  += 1
        if ma_gap < -0.002: sell += 1

        if rsi_v < thr["rsi_oversold"]:   buy  += 2
        if rsi_v > thr["rsi_overbought"]: sell += 2

        if hist_v > 0 and hist_p <= 0: buy  += 2
        if hist_v < 0 and hist_p >= 0: sell += 2
        if macd_v > msig_v:            buy  += 1
        if macd_v < msig_v:            sell += 1

        if price <= bbl: buy  += 2
        if price >= bbu: sell += 2

        if vol_r > 1.5:
            buy  = int(buy  * 1.2)
            sell = int(sell * 1.2)

        sig_min = thr["signal_min"]
        signal  = (
            "BUY"     if buy  >= sig_min else
            "SELL"    if sell >= sig_min else
            "BULLISH" if buy  >  sell else
            "BEARISH" if sell >  buy  else "HOLD"
        )

        return {
            "price": price, "rsi": rsi_v,
            "sma_short": ss, "sma_long": sl,
            "macd": macd_v, "macd_signal": msig_v, "macd_hist": hist_v,
            "bb_low": bbl, "bb_up": bbu,
            "vol_ratio": vol_r,
            "buy_score": buy, "sell_score": sell, "signal": signal,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        log.debug("Indicators: " + str(e))
        return None


# =============================================================================
# ORDERS
# =============================================================================

def calc_qty(portfolio_value: float, cash: float, price: float) -> float:
    if cash <= 0 or price <= 0:
        return 0.0
    # Use 90% of available cash — buy one position at a time with what we have
    alloc = cash * 0.90
    if alloc < MIN_ORDER_USD:
        return 0.0
    qty = alloc / price
    if price > 10000: return round(qty, 8)
    if price > 1:     return round(qty, 6)
    return round(qty, 2)


def place_order(api, symbol: str, qty: float, side: str, reason: str = "") -> Optional[dict]:
    try:
        order = api.submit_order(
            symbol=symbol, qty=qty, side=side,
            type="market", time_in_force="gtc",
        )
        icon = "✅ BUY" if side == "buy" else "💰 SELL"
        log.info(icon + " " + str(qty) + " " + symbol + " | " + reason + " | id=" + order.id)
        return {
            "id": order.id, "symbol": symbol, "qty": qty,
            "side": side, "reason": reason,
            "time": datetime.now().isoformat(),
        }
    except Exception as e:
        log.error("❌ Order failed " + side + " " + symbol + ": " + str(e))
        return None


# =============================================================================
# EXITS
# =============================================================================

def check_exits(api, positions: list, peak_prices: dict, agent: dict, state: dict) -> tuple:
    exits = []
    thr   = agent["thresholds"]

    for pos in positions:
        symbol = pos.symbol
        base   = symbol.replace("USD", "").replace("USDT", "")
        pair   = base + "/USD"
        qty    = float(pos.qty)
        plpc   = float(pos.unrealized_plpc)
        price  = float(pos.current_price)
        entry  = float(pos.avg_entry_price)

        peak = max(peak_prices.get(symbol, entry), price)
        peak_prices[symbol] = peak

        reason = None
        if plpc <= -thr["stop_loss"]:
            reason = "stop_loss"
            log.warning("🛑 STOP LOSS " + symbol + ": " + str(round(plpc * 100, 1)) + "%")
        elif plpc >= thr["take_profit"]:
            reason = "take_profit"
            log.info("🎯 TAKE PROFIT " + symbol + ": " + str(round(plpc * 100, 1)) + "%")
        elif peak > entry * (1 + TRAILING_ACTIVATE_PCT):
            trail = peak * (1 - thr["trailing_stop"])
            if price <= trail:
                reason = "trailing_stop"
                log.info("📉 TRAILING STOP " + symbol + ": " + str(round(plpc * 100, 1)) + "%")

        if reason:
            order = place_order(api, pair, qty, "sell", reason)
            if order:
                agent_record_close(agent, symbol, price, reason)
                exits.append({**order, "symbol": symbol, "pnl_pct": plpc * 100})
                peak_prices.pop(symbol, None)
                if plpc > 0:
                    state["wins"]   = state.get("wins", 0) + 1
                else:
                    state["losses"] = state.get("losses", 0) + 1

    return exits, peak_prices


# =============================================================================
# MAIN
# =============================================================================

def run_bot():
    log.info("=" * 60)
    log.info("🪙 CryptoTrader v3 | " + datetime.now().strftime("%Y-%m-%d %H:%M UTC"))
    log.info("=" * 60)

    api   = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version="v2")
    state = load_state()
    agent = load_agent()
    agent_log_summary(agent)

    acc             = api.get_account()
    portfolio_value = float(acc.portfolio_value)
    # Use the lower of cash and buying_power to get truly spendable amount
    raw_cash        = float(acc.cash)
    buying_power    = float(acc.buying_power)
    non_marginable  = float(getattr(acc, 'non_marginable_buying_power', buying_power))
    cash            = min(raw_cash, buying_power, non_marginable)
    log.info("💵 Buying power breakdown: cash=" + str(round(raw_cash,2)) + 
             " bp=" + str(round(buying_power,2)) + 
             " spendable=" + str(round(cash,2)))
    log.info("💰 Portfolio: $" + str(round(portfolio_value, 2)) + " | Cash: $" + str(round(cash, 2)))

    if acc.status != "ACTIVE":
        log.error("Account not active — aborting")
        return

    today = datetime.now().strftime("%Y-%m-%d")
    if state.get("trade_date") != today:
        state["daily_trades"] = 0
        state["trade_date"]   = today

    all_pos    = api.list_positions()
    crypto_pos = [p for p in all_pos if "USD" in p.symbol and len(p.symbol) <= 8]
    n_pos      = len(crypto_pos)
    peak_prices = state.get("peak_prices", {})
    log.info("📊 Open positions: " + str(n_pos) + "/" + str(MAX_POSITIONS))

    exits, peak_prices = check_exits(api, crypto_pos, peak_prices, agent, state)
    if exits:
        state["trades"].extend(exits)
        state["daily_trades"] += len(exits)
        state["total_trades"]  = state.get("total_trades", 0) + len(exits)
        crypto_pos  = [p for p in api.list_positions() if "USD" in p.symbol and len(p.symbol) <= 8]
        n_pos       = len(crypto_pos)
        cash        = float(api.get_account().cash)

    held = {p.symbol for p in crypto_pos}

    log.info("🔍 Scanning " + str(len(CRYPTO_UNIVERSE)) + " pairs...")
    bars = fetch_bars(api, CRYPTO_UNIVERSE)
    log.info("📡 Data for " + str(len(bars)) + " pairs")

    all_signals = {}
    all_buys    = []
    all_sells   = []

    for pair, df in bars.items():
        ind = compute_indicators(df, agent)
        if ind is None:
            continue
        pos_sym = pair.replace("/", "")
        if not agent_should_trade(agent, pos_sym):
            continue
        ind         = agent_score(agent, pos_sym, ind)
        ind["symbol"] = pair
        all_signals[pair] = ind

        if ind["signal"] == "BUY" and pos_sym not in held:
            all_buys.append(ind)
        elif ind["signal"] in ("SELL", "BEARISH") and pos_sym in held:
            all_sells.append(ind)

    log.info("📊 Signals | Buys=" + str(len(all_buys)) + " Sells=" + str(len(all_sells)))

    for sig in all_sells:
        pair    = sig["symbol"]
        pos_sym = pair.replace("/", "")
        pos     = next((p for p in crypto_pos if p.symbol == pos_sym), None)
        if not pos:
            continue
        qty   = float(pos.qty)
        plpc  = float(pos.unrealized_plpc)
        order = place_order(api, pair, qty, "sell", "signal_sell score=" + str(sig["sell_score"]))
        if order:
            agent_record_close(agent, pos_sym, sig["price"], "signal_sell")
            state["trades"].append({**order, "symbol": pos_sym})
            state["daily_trades"] += 1
            state["total_trades"]  = state.get("total_trades", 0) + 1
            if plpc > 0: state["wins"]   = state.get("wins", 0) + 1
            else:        state["losses"] = state.get("losses", 0) + 1
            held.discard(pos_sym)
            n_pos -= 1
            peak_prices.pop(pos_sym, None)

    slots    = MAX_POSITIONS - n_pos
    min_cash = portfolio_value * MIN_CASH_BUFFER
    all_buys.sort(key=lambda x: x.get("confidence", 0), reverse=True)

    for sig in all_buys[:slots]:
        pair  = sig["symbol"]
        price = sig["price"]
        qty   = calc_qty(portfolio_value, cash, price)
        cost  = qty * price if qty > 0 else 0

        log.info("💡 Considering " + pair + " | price=$" + str(round(price, 6)) +
                 " qty=" + str(qty) + " cost=$" + str(round(cost, 4)) +
                 " cash=$" + str(round(cash, 2)))

        if qty == 0.0:
            log.info("⚠️  Skip " + pair + " — qty is zero")
            continue
        if cost < MIN_ORDER_USD:
            log.info("⚠️  Skip " + pair + " — cost $" + str(round(cost, 4)) + " below min $" + str(MIN_ORDER_USD))
            continue
        if (cash - cost) < min_cash:
            log.info("⚠️  Skip " + pair + " — low cash buffer")
            continue

        order = place_order(api, pair, qty, "buy",
                            "signal_buy score=" + str(sig["buy_score"]) + " conf=" + str(sig.get("confidence", 0)))
        if order:
            pos_sym = pair.replace("/", "")
            agent_record_open(agent, pos_sym, price, sig)
            state["trades"].append({**order, "symbol": pos_sym, "price": price})
            state["daily_trades"] += 1
            state["total_trades"]  = state.get("total_trades", 0) + 1
            peak_prices[pos_sym]   = price
            cash  -= cost
            n_pos += 1

    state["signals"]     = all_signals
    state["last_run"]    = datetime.now().isoformat()
    state["peak_prices"] = peak_prices
    save_state(state)
    save_agent(agent)

    wins   = state.get("wins", 0)
    losses = state.get("losses", 0)
    total  = wins + losses

    try:
        live_pos = [
            {
                "symbol":          p.symbol,
                "qty":             float(p.qty),
                "avg_entry":       float(p.avg_entry_price),
                "current_price":   float(p.current_price),
                "market_value":    float(p.market_value),
                "unrealized_pl":   float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc),
            }
            for p in api.list_positions()
            if "USD" in p.symbol and len(p.symbol) <= 8
        ]
    except Exception:
        live_pos = []

    report = {
        "generated_at": datetime.now().isoformat(),
        "account": {
            "portfolio_value": portfolio_value,
            "cash":            cash,
            "buying_power":    float(acc.buying_power),
            "status":          acc.status,
        },
        "last_run":  state["last_run"],
        "positions": live_pos,
        "trades":    state["trades"][-50:],
        "signals":   all_signals,
        "agent": {
            "total_learned":   agent["total_learned"],
            "win_rate_7d":     agent["win_rate_7d"],
            "win_rate_30d":    agent["win_rate_30d"],
            "avg_pnl":         agent["avg_pnl"],
            "sharpe":          agent["sharpe"],
            "thresholds":      agent["thresholds"],
            "weights":         agent["weights"],
            "last_calibrated": agent["last_calibrated"],
            "calibration_log": agent["calibration_log"][-5:],
        },
        "performance": {
            "total_trades":   state.get("total_trades", 0),
            "daily_trades":   state.get("daily_trades", 0),
            "wins":           wins,
            "losses":         losses,
            "win_rate":       wins / total if total > 0 else 0,
            "open_positions": n_pos,
        },
    }

    REPORT_FILE.write_text(json.dumps(report, indent=2, default=str))
    log.info("✅ Done | Daily=" + str(state["daily_trades"]) +
             " | W/L=" + str(wins) + "/" + str(losses) +
             " | Positions=" + str(n_pos))
    log.info("=" * 60)


if __name__ == "__main__":
    run_bot()
