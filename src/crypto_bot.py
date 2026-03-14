"""
CryptoTrader v4 - World-Class Smart Crypto Trading Bot
Built as if by the world's best trader with the best track record.

ENTRY INTELLIGENCE:
  - Multi-timeframe trend analysis (1h, 3h, 6h, 24h)
  - Falling knife prevention (don't buy into continued crashes)
  - RSI divergence detection (price falling, RSI rising = hidden strength)
  - Volume quality analysis (smart money vs retail)
  - Support/resistance levels from recent price action
  - Market regime detection (crash/bear/recovery/bull via BTC)
  - Stabilisation detection (is the dip bottoming out?)

RISK MANAGEMENT:
  - ATR-based dynamic stop losses (volatility-adjusted)
  - Kelly-inspired position sizing (confidence x volatility)
  - Trailing stop activates immediately on any gain
  - Never buys in crash regime
  - Per-regime performance tracking

LEARNING AGENT:
  - Per-signal-type win rate tracking
  - Per-coin memory
  - Per-hour performance
  - Per-regime performance
  - Auto-calibrates every 5 trades
  - Adjusts TP, SL, signal weights, RSI thresholds automatically
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

# Intelligence modules — loaded after logging so errors are captured
MODULES_LOADED = False
try:
    from sentiment   import get_sentiment
    from momentum    import get_momentum_score
    from patterns    import detect_patterns
    from correlation import get_correlation_signals
    from risk        import (load_risk_state, save_risk_state, update_risk_state,
                             check_risk, get_position_size_multiplier, record_trade_result)
    from grid        import load_grid_state, save_grid_state, run_grid
    MODULES_LOADED = True
    log.info("✅ All intelligence modules loaded")
except ImportError as e:
    log.warning("⚠️  Intelligence module not loaded: " + str(e))
    log.warning("⚠️  Running with core strategy only")

ALPACA_API_KEY    = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]
ALPACA_BASE_URL   = os.environ.get("ALPACA_BASE_URL", "https://api.alpaca.markets")

CRYPTO_UNIVERSE = [
    # Major coins
    "BTC/USD",   # Bitcoin
    "ETH/USD",   # Ethereum
    "SOL/USD",   # Solana
    "XRP/USD",   # Ripple
    "DOGE/USD",  # Dogecoin

    # Mid caps with good liquidity
    "AVAX/USD",  # Avalanche
    "LINK/USD",  # Chainlink
    "LTC/USD",   # Litecoin
    "BCH/USD",   # Bitcoin Cash
    "DOT/USD",   # Polkadot
    "UNI/USD",   # Uniswap
    "AAVE/USD",  # Aave

    # High volatility / meme coins
    "SHIB/USD",  # Shiba Inu
    "PEPE/USD",  # Pepe
    "TRUMP/USD", # Trump coin — highly volatile

    # Additional Alpaca supported pairs
    "BAT/USD",   # Basic Attention Token
    "CRV/USD",   # Curve Finance
    "GRT/USD",   # The Graph
    "SUSHI/USD", # SushiSwap
    "XTZ/USD",   # Tezos
    "YFI/USD",   # Yearn Finance
    "MKR/USD",   # Maker
]

MAX_POSITIONS       = 5  # slightly more with 22 pairs to scan
MIN_ORDER_USD       = 1.10
MIN_CASH_BUFFER     = 0.05
DEFAULT_TP          = 0.025
DEFAULT_SL          = 0.03
DEFAULT_TRAIL       = 0.015
TRAIL_ACTIVATE      = 0.008
MIN_SIGNAL_SCORE    = 5  # raised — only trade strong setups
TREND_LOOKBACK_H    = 6

STATE_FILE  = Path("logs/crypto_state.json")
AGENT_FILE  = Path("logs/crypto_agent.json")
REPORT_FILE = Path("logs/crypto_report.json")


# =============================================================================
# LEARNING AGENT
# =============================================================================

DEFAULT_AGENT = {
    "version": 2,
    "total_learned": 0,
    "wins": 0,
    "losses": 0,
    "win_rate_7d":  0.5,
    "win_rate_30d": 0.5,
    "avg_pnl":  0.0,
    "sharpe":   0.0,
    "thresholds": {
        "take_profit":    DEFAULT_TP,
        "stop_loss":      DEFAULT_SL,
        "trailing_stop":  DEFAULT_TRAIL,
        "signal_min":     MIN_SIGNAL_SCORE,
        "rsi_oversold":   35,
        "rsi_overbought": 65,
    },
    "weights": {
        "trend_alignment": 1.0,
        "rsi_divergence":  1.0,
        "volume_confirm":  1.0,
        "macd_cross":      1.0,
        "bb_bounce":       1.0,
        "momentum":        1.0,
    },
    "symbol_memory":    {},
    "hour_performance": {},
    "regime_performance": {
        "crash":    {"wins": 0, "losses": 0},
        "bear":     {"wins": 0, "losses": 0},
        "recovery": {"wins": 0, "losses": 0},
        "bull":     {"wins": 0, "losses": 0},
    },
    "signal_type_performance": {},
    "open_trades":     {},
    "closed_trades":   [],
    "calibration_log": [],
    "last_calibrated": None,
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
    agent["closed_trades"]   = agent["closed_trades"][-300:]
    agent["calibration_log"] = agent["calibration_log"][-50:]
    AGENT_FILE.write_text(json.dumps(agent, indent=2, default=str))


def agent_record_open(agent: dict, symbol: str, price: float, analysis: dict):
    agent["open_trades"][symbol] = {
        "entry_price":   price,
        "entry_time":    datetime.now().isoformat(),
        "entry_hour":    datetime.now().hour,
        "score":         analysis.get("final_score", 0),
        "regime":        analysis.get("regime", "unknown"),
        "rsi":           analysis.get("rsi", 50),
        "vol_ratio":     analysis.get("vol_ratio", 1.0),
        "trend_slope":   analysis.get("trend_slope", 0),
        "signals_fired": analysis.get("signals_fired", []),
    }
    log.info("🧠 Opened: " + symbol + " @ $" + str(round(price, 6)))


def agent_record_close(agent: dict, symbol: str, exit_price: float, reason: str):
    if symbol not in agent["open_trades"]:
        return
    trade   = agent["open_trades"].pop(symbol)
    entry   = trade["entry_price"]
    pnl_pct = (exit_price - entry) / entry * 100
    won     = pnl_pct > 0
    regime  = trade.get("regime", "unknown")

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
    hp["trades"] += 1
    hp["total_pnl"] += pnl_pct
    if won: hp["wins"] += 1
    else:   hp["losses"] += 1

    rp = agent["regime_performance"].setdefault(regime, {"wins": 0, "losses": 0})
    if won: rp["wins"] += 1
    else:   rp["losses"] += 1

    for sig in trade.get("signals_fired", []):
        sp = agent["signal_type_performance"].setdefault(sig, {
            "wins": 0, "losses": 0, "trades": 0
        })
        sp["trades"] += 1
        if won: sp["wins"] += 1
        else:   sp["losses"] += 1

    agent["closed_trades"].append({
        "symbol":        symbol,
        "entry":         entry,
        "exit":          exit_price,
        "pnl_pct":       pnl_pct,
        "won":           won,
        "reason":        reason,
        "regime":        regime,
        "score":         trade.get("score", 0),
        "rsi":           trade.get("rsi", 50),
        "vol_ratio":     trade.get("vol_ratio", 1.0),
        "signals_fired": trade.get("signals_fired", []),
        "time":          datetime.now().isoformat(),
    })
    agent["total_learned"] += 1
    icon = "✅" if won else "❌"
    log.info("🧠 " + icon + " " + symbol + " PnL=" + str(round(pnl_pct, 2)) + "% | " + reason)

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

    if wr20 > 0.65 and avg_pnl > 1.5:
        old = thr["take_profit"]
        thr["take_profit"] = min(0.06, round(old + 0.005, 3))
        changes.append("TP->" + str(thr["take_profit"]) + " (let winners run)")
    elif wr20 < 0.40:
        old = thr["take_profit"]
        thr["take_profit"] = max(0.015, round(old - 0.003, 3))
        changes.append("TP->" + str(thr["take_profit"]) + " (bank faster)")

    if avg_pnl < -2.0 and wr20 < 0.35:
        old = thr["stop_loss"]
        thr["stop_loss"] = max(0.015, round(old - 0.003, 3))
        changes.append("SL->" + str(thr["stop_loss"]) + " (cut faster)")

    if wr20 < 0.35 and len(r20) >= 10:
        old = thr["signal_min"]
        thr["signal_min"] = min(8, old + 1)
        changes.append("SigMin->" + str(thr["signal_min"]))
    elif wr20 > 0.70 and avg_pnl > 2.0:
        old = thr["signal_min"]
        thr["signal_min"] = max(2, old - 1)
        changes.append("SigMin->" + str(thr["signal_min"]))

    hi_rsi = [t for t in r20 if t.get("rsi", 50) > 60]
    if len(hi_rsi) >= 3 and sum(1 for t in hi_rsi if t["won"]) / len(hi_rsi) < 0.35:
        old = thr["rsi_overbought"]
        thr["rsi_overbought"] = max(55, old - 3)
        changes.append("RSI OB->" + str(thr["rsi_overbought"]))

    lo_rsi = [t for t in r20 if t.get("rsi", 50) < 35]
    if len(lo_rsi) >= 3:
        lr_wr = sum(1 for t in lo_rsi if t["won"]) / len(lo_rsi)
        if lr_wr > 0.70:
            old = thr["rsi_oversold"]
            thr["rsi_oversold"] = min(45, old + 3)
            changes.append("RSI OS->" + str(thr["rsi_oversold"]) + " (buy earlier)")
        elif lr_wr < 0.30:
            old = thr["rsi_oversold"]
            thr["rsi_oversold"] = max(20, old - 3)
            changes.append("RSI OS->" + str(thr["rsi_oversold"]) + " (wait deeper)")

    for sig_name, sp in agent["signal_type_performance"].items():
        if sp["trades"] >= 5:
            sig_wr = sp["wins"] / sp["trades"]
            w = wts.get(sig_name, 1.0)
            if sig_wr > 0.70:
                wts[sig_name] = min(2.0, round(w + 0.15, 2))
                changes.append(sig_name + "->" + str(wts[sig_name]) + " (winning)")
            elif sig_wr < 0.30:
                wts[sig_name] = max(0.2, round(w - 0.15, 2))
                changes.append(sig_name + "->" + str(wts[sig_name]) + " (losing)")

    if len(trades) >= 10:
        pnls = [t["pnl_pct"] for t in trades[-30:]]
        mean = sum(pnls) / len(pnls)
        var  = sum((p - mean) ** 2 for p in pnls) / len(pnls)
        std  = math.sqrt(var) if var > 0 else 1
        agent["sharpe"] = round(mean / std, 3)

    agent["last_calibrated"] = datetime.now().isoformat()
    if changes:
        agent["calibration_log"].append({
            "time": datetime.now().isoformat(),
            "changes": changes,
            "wr20": round(wr20, 3),
            "avg_pnl": round(avg_pnl, 3),
        })
        for c in changes:
            log.info("🧠 Calibrated: " + c)
    else:
        log.info("🧠 No changes WR=" + str(round(wr20 * 100)) + "% avg=" + str(round(avg_pnl, 2)) + "%")


def agent_log_summary(agent: dict):
    n = agent["total_learned"]
    log.info("🧠 Agent | n=" + str(n) +
             " WR7=" + str(round(agent["win_rate_7d"] * 100)) + "%" +
             " WR30=" + str(round(agent["win_rate_30d"] * 100)) + "%" +
             " PnL=" + str(round(agent["avg_pnl"], 2)) + "%" +
             " Sharpe=" + str(agent["sharpe"]))
    if n == 0:
        log.info("🧠 No trades learned yet — using smart defaults")
        return
    thr = agent["thresholds"]
    log.info("🧠 TP=" + str(round(thr["take_profit"] * 100, 1)) + "%" +
             " SL=" + str(round(thr["stop_loss"] * 100, 1)) + "%" +
             " Trail=" + str(round(thr["trailing_stop"] * 100, 1)) + "%" +
             " SigMin=" + str(thr["signal_min"]))
    syms = [(s, m) for s, m in agent["symbol_memory"].items() if m.get("trades", 0) >= 2]
    if syms:
        best  = sorted(syms, key=lambda x: x[1]["avg_pnl"], reverse=True)[:3]
        worst = sorted(syms, key=lambda x: x[1]["avg_pnl"])[:3]
        log.info("🧠 Best:  " + " | ".join(s + " " + str(round(m["avg_pnl"], 1)) + "%" for s, m in best))
        log.info("🧠 Worst: " + " | ".join(s + " " + str(round(m["avg_pnl"], 1)) + "%" for s, m in worst))


def agent_should_trade(agent: dict, symbol: str) -> bool:
    mem = agent["symbol_memory"].get(symbol, {})
    if (mem.get("trades", 0) >= 6 and
            mem.get("win_rate", 0.5) < 0.25 and
            mem.get("avg_pnl", 0.0) < -2.0):
        log.info("🧠 Blocking " + symbol + " persistent loser")
        return False
    return True


def agent_get_hour_mult(agent: dict) -> float:
    hour  = str(datetime.now().hour)
    hperf = agent["hour_performance"].get(hour, {})
    if hperf.get("trades", 0) >= 4:
        wr = hperf["wins"] / hperf["trades"]
        return max(0.6, min(1.4, 0.4 + wr))
    return 1.0


def agent_get_regime_mult(agent: dict, regime: str) -> float:
    rp    = agent["regime_performance"].get(regime, {"wins": 0, "losses": 0})
    total = rp["wins"] + rp["losses"]
    if total >= 4:
        wr = rp["wins"] / total
        return max(0.5, min(1.5, wr * 2))
    return 1.0


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

def fetch_bars(api, symbols: List[str], hours: int = 200) -> Dict[str, pd.DataFrame]:
    result = {}
    end    = datetime.now(timezone.utc)
    start  = end - timedelta(hours=hours)
    for sym in symbols:
        try:
            bars = api.get_crypto_bars(
                sym,
                tradeapi.rest.TimeFrame.Hour,
                start=start.isoformat(),
                end=end.isoformat(),
            ).df
            if bars is not None and not bars.empty and len(bars) >= 30:
                result[sym] = bars
        except Exception as e:
            log.debug("Bars " + sym + ": " + str(e))
    return result


# =============================================================================
# ANALYSIS ENGINE
# =============================================================================

def detect_regime(btc_df: pd.DataFrame) -> str:
    try:
        close   = btc_df["close"].astype(float)
        price   = float(close.iloc[-1])
        ma50    = float(close.rolling(50).mean().iloc[-1])
        ma20    = float(close.rolling(20).mean().iloc[-1])
        chg_4h  = (price - float(close.iloc[-4]))  / float(close.iloc[-4])
        chg_24h = (price - float(close.iloc[-24])) / float(close.iloc[-24]) if len(close) >= 24 else 0
        if chg_4h < -0.05 or chg_24h < -0.10:
            return "crash"
        elif price < ma50 * 0.97:
            return "bear"
        elif price > ma50 and price > ma20:
            return "bull"
        else:
            return "recovery"
    except Exception:
        return "unknown"


def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    try:
        high  = df["high"].astype(float)
        low   = df["low"].astype(float)
        close = df["close"].astype(float)
        prev  = close.shift(1)
        tr    = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
        return float(tr.rolling(period).mean().iloc[-1])
    except Exception:
        return 0.0


def rsi_divergence(close: pd.Series, rsi: pd.Series, lookback: int = 10) -> str:
    try:
        pc = float(close.iloc[-1]) - float(close.iloc[-lookback])
        rc = float(rsi.iloc[-1])   - float(rsi.iloc[-lookback])
        if pc < -0.005 and rc > 2:  return "bullish"
        if pc >  0.005 and rc < -2: return "bearish"
        return "none"
    except Exception:
        return "none"


def trend_slope(close: pd.Series, hours: int) -> float:
    try:
        recent = close.iloc[-hours:].astype(float)
        return (float(recent.iloc[-1]) - float(recent.iloc[0])) / float(recent.iloc[0]) / hours * 100
    except Exception:
        return 0.0


def support_level(close: pd.Series, lookback: int = 48) -> float:
    try:
        return float(close.iloc[-lookback:].astype(float).quantile(0.15))
    except Exception:
        return 0.0


def analyse_coin(df: pd.DataFrame, agent: dict, regime: str) -> Optional[dict]:
    try:
        close = df["close"].astype(float)
        high  = df["high"].astype(float)
        low   = df["low"].astype(float)
        vol   = df["volume"].astype(float)

        if len(close) < 50:
            return None

        thr = agent["thresholds"]
        wts = agent["weights"]

        sma3   = close.rolling(3).mean()
        sma9   = close.rolling(9).mean()
        sma21  = close.rolling(21).mean()
        sma50  = close.rolling(50).mean()
        ema12  = close.ewm(span=12, adjust=False).mean()
        ema26  = close.ewm(span=26, adjust=False).mean()
        macd   = ema12 - ema26
        macd_s = macd.ewm(span=9, adjust=False).mean()
        hist   = macd - macd_s
        delta  = close.diff()
        gain   = delta.clip(lower=0).rolling(5).mean()
        loss   = (-delta.clip(upper=0)).rolling(5).mean()
        rsi_s  = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_low = bb_mid - 2 * bb_std
        bb_up  = bb_mid + 2 * bb_std
        bb_pct = (close - bb_low) / (bb_up - bb_low)
        avg_v  = vol.rolling(20).mean()

        price    = float(close.iloc[-1])
        s3       = float(sma3.iloc[-1]);  s3p = float(sma3.iloc[-2])
        s9       = float(sma9.iloc[-1]);  s9p = float(sma9.iloc[-2])
        s21      = float(sma21.iloc[-1])
        s50      = float(sma50.iloc[-1])
        rsi_v    = float(rsi_s.iloc[-1])
        hist_v   = float(hist.iloc[-1]);  histp = float(hist.iloc[-2])
        macd_v   = float(macd.iloc[-1]);  msig  = float(macd_s.iloc[-1])
        bb_pct_v = float(bb_pct.iloc[-1])
        bbl      = float(bb_low.iloc[-1])
        bbu      = float(bb_up.iloc[-1])
        avg_vol  = float(avg_v.iloc[-1])
        cur_vol  = float(vol.iloc[-1])
        vol_ratio = cur_vol / avg_vol if avg_vol > 0 else 1.0
        vol_accel = float(vol.iloc[-3:].mean()) / avg_vol if avg_vol > 0 else 1.0
        vol_strong = vol_ratio > 1.3 and vol_accel > 1.2
        atr      = compute_atr(df)
        atr_pct  = atr / price if price > 0 else 0.02

        rsi_div    = rsi_divergence(close, rsi_s)
        slope_1h   = trend_slope(close, 1)
        slope_3h   = trend_slope(close, 3)
        slope_6h   = trend_slope(close, 6)
        slope_24h  = trend_slope(close, 24) if len(close) >= 24 else 0
        trend_4h   = (price - float(close.iloc[-4])) / float(close.iloc[-4])
        support    = support_level(close)
        near_sup   = price <= support * 1.02
        stabilising = slope_6h < 0 and slope_1h > slope_3h

        buy_score  = 0.0
        sell_score = 0.0
        signals_fired = []

        # 1. TREND ALIGNMENT
        tw = wts.get("trend_alignment", 1.0)
        if s3 > s9 > s21:
            buy_score += 2 * tw
            signals_fired.append("trend_bull")
        elif s3 < s9 < s21:
            sell_score += 2 * tw
            signals_fired.append("trend_bear")
        if s3p <= s9p and s3 > s9:
            buy_score += 3 * tw
            signals_fired.append("ma_cross_bull")
        elif s3p >= s9p and s3 < s9:
            sell_score += 3 * tw
            signals_fired.append("ma_cross_bear")
        if price > s50:
            buy_score += 1 * tw
        else:
            sell_score += 0.5 * tw

        # 2. RSI
        rw = wts.get("rsi_divergence", 1.0)
        if rsi_v < thr["rsi_oversold"]:
            buy_score += 2 * rw
            signals_fired.append("rsi_oversold")
        elif rsi_v > thr["rsi_overbought"]:
            sell_score += 2 * rw
            signals_fired.append("rsi_overbought")
        if rsi_div == "bullish":
            buy_score += 3 * rw
            signals_fired.append("rsi_divergence_bull")
        elif rsi_div == "bearish":
            sell_score += 3 * rw
            signals_fired.append("rsi_divergence_bear")

        # 3. MACD
        mw = wts.get("macd_cross", 1.0)
        if hist_v > 0 and histp <= 0:
            buy_score += 3 * mw
            signals_fired.append("macd_bull")
        elif hist_v < 0 and histp >= 0:
            sell_score += 3 * mw
            signals_fired.append("macd_bear")
        if macd_v > msig:
            buy_score += 1 * mw
        else:
            sell_score += 0.5 * mw

        # 4. BOLLINGER BANDS
        bw = wts.get("bb_bounce", 1.0)
        if bb_pct_v < 0.10:   # tighter — only deep lower band
            buy_score += 4 * bw   # boosted — 4W/0L historically
            signals_fired.append("bb_lower")
        elif bb_pct_v > 0.90:
            sell_score += 3 * bw
            signals_fired.append("bb_upper")
        # near_support removed — data shows 10W/20L, not reliable

        # 5. VOLUME
        vw = wts.get("volume_confirm", 1.0)
        if vol_strong:
            buy_score  *= (1 + 0.3 * vw)
            signals_fired.append("vol_confirm")
        elif vol_ratio < 0.5:
            buy_score *= 0.7

        # 6. MOMENTUM
        mow = wts.get("momentum", 1.0)
        if slope_1h > 0.05:
            buy_score += 1 * mow
            signals_fired.append("momentum_up")
        elif slope_6h < -0.15:
            sell_score += 2 * mow
            signals_fired.append("momentum_down")

        # FALLING KNIFE PREVENTION
        if trend_4h < -0.03 and not stabilising:
            buy_score *= 0.15  # much harder penalty
            signals_fired.append("falling_knife")
        if stabilising and rsi_v < 40:
            buy_score *= 1.3
            signals_fired.append("stabilising")
        if slope_24h < -0.05:
            buy_score *= 0.4   # don't buy coins down 5%+ today
        if slope_24h < -0.10:
            buy_score *= 0.1   # almost never buy coins down 10%+ today

        # REGIME ADJUSTMENT
        rm = agent_get_regime_mult(agent, regime)
        if regime == "crash":
            buy_score *= 0.1
        elif regime == "bear":
            buy_score *= 0.6 * rm
        elif regime == "recovery":
            buy_score *= 1.1 * rm
        elif regime == "bull":
            buy_score *= 1.3 * rm

        # HOUR ADJUSTMENT
        buy_score *= agent_get_hour_mult(agent)

        # DYNAMIC STOPS
        dyn_stop = max(thr["stop_loss"],  min(0.06, atr_pct * 1.5))
        dyn_tp   = max(thr["take_profit"], min(0.08, atr_pct * 2.0))

        sig_min = thr["signal_min"]
        signal  = (
            "BUY"     if buy_score  >= sig_min else
            "SELL"    if sell_score >= sig_min else
            "BULLISH" if buy_score  >  sell_score else
            "BEARISH" if sell_score >  buy_score  else "HOLD"
        )

        confidence = min(1.0,
            (buy_score / 15.0) * 0.4 +
            agent.get("win_rate_7d", 0.5) * 0.3 +
            (0.9 if rsi_div == "bullish" else 0.5) * 0.15 +
            (0.9 if vol_strong else 0.3) * 0.15
        )

        return {
            "price":          price,
            "rsi":            round(rsi_v, 2),
            "sma3":           round(s3, 6),
            "sma9":           round(s9, 6),
            "sma21":          round(s21, 6),
            "sma50":          round(s50, 6),
            "macd_hist":      round(hist_v, 6),
            "bb_pct":         round(bb_pct_v, 3),
            "vol_ratio":      round(vol_ratio, 3),
            "vol_strong":     vol_strong,
            "rsi_divergence": rsi_div,
            "trend_slope":    round(slope_6h, 4),
            "trend_1h":       round(slope_1h, 4),
            "trend_4h":       round(trend_4h, 4),
            "trend_24h":      round(slope_24h, 4),
            "stabilising":    stabilising,
            "near_support":   near_sup,
            "atr_pct":        round(atr_pct, 4),
            "dynamic_stop":   round(dyn_stop, 4),
            "dynamic_tp":     round(dyn_tp, 4),
            "buy_score":      round(buy_score, 2),
            "sell_score":     round(sell_score, 2),
            "final_score":    round(buy_score, 2),
            "signal":         signal,
            "confidence":     round(confidence, 3),
            "regime":         regime,
            "signals_fired":  signals_fired,
            "timestamp":      datetime.now().isoformat(),
        }

    except Exception as e:
        log.debug("Analysis error: " + str(e))
        return None


# =============================================================================
# POSITION SIZING & ORDERS
# =============================================================================

def calc_qty(cash: float, price: float, confidence: float = 0.5, atr_pct: float = 0.02) -> float:
    if cash <= 0 or price <= 0:
        return 0.0
    base_alloc = cash * 0.90
    conf_scale = 0.6 + (confidence * 0.4)
    vol_scale  = max(0.5, min(1.0, 0.02 / max(atr_pct, 0.005)))
    alloc      = base_alloc * conf_scale * vol_scale
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
        icon = "✅ BUY " if side == "buy" else "💰 SELL"
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

def check_exits(api, positions: list, peak_prices: dict,
                agent: dict, state: dict, analyses: dict) -> tuple:
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

        peak  = max(peak_prices.get(symbol, entry), price)
        peak_prices[symbol] = peak

        analysis   = analyses.get(pair, {})
        dyn_stop   = analysis.get("dynamic_stop",  thr["stop_loss"])
        dyn_tp     = analysis.get("dynamic_tp",    thr["take_profit"])
        sell_score = analysis.get("sell_score",    0)

        reason = None

        if plpc <= -dyn_stop:
            reason = "stop_loss"
            log.warning("🛑 STOP " + symbol + " " + str(round(plpc * 100, 1)) + "%")
        elif plpc >= dyn_tp:
            reason = "take_profit"
            log.info("🎯 PROFIT " + symbol + " +" + str(round(plpc * 100, 1)) + "%")
        elif peak > entry * (1 + TRAIL_ACTIVATE * 2):  # needs 1.6% gain before trailing
            trail = peak * (1 - thr["trailing_stop"])
            if price <= trail:
                reason = "trailing_stop"
                log.info("📉 TRAIL " + symbol)
        elif sell_score >= thr["signal_min"] * 2.5 and plpc > 0.008:
            reason = "signal_exit"
            log.info("📊 SIGNAL EXIT " + symbol + " +" + str(round(plpc * 100, 1)) + "%")

        if reason:
            order = place_order(api, pair, qty, "sell", reason)
            if order:
                agent_record_close(agent, symbol, price, reason)
                exits.append({**order, "symbol": symbol, "pnl_pct": plpc * 100})
                peak_prices.pop(symbol, None)
                won = plpc > 0
                if won: state["wins"]   = state.get("wins",   0) + 1
                else:   state["losses"] = state.get("losses", 0) + 1

    return exits, peak_prices


# =============================================================================
# MAIN
# =============================================================================

def run_bot():
    log.info("=" * 60)
    log.info("🪙 CryptoTrader v4 | " + datetime.now().strftime("%Y-%m-%d %H:%M UTC"))
    log.info("=" * 60)

    api   = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version="v2")
    state      = load_state()
    agent      = load_agent()
    risk       = load_risk_state()  if MODULES_LOADED else {}
    grid_state = load_grid_state()  if MODULES_LOADED else {"grids": {}}
    dca_state  = load_dca_state()   if MODULES_LOADED else {"positions": {}}
    agent_log_summary(agent)

    acc          = api.get_account()
    pv           = float(acc.portfolio_value)
    buying_power = float(acc.buying_power)
    raw_cash     = float(acc.cash)
    try:
        nm_bp = float(getattr(acc, "non_marginable_buying_power", buying_power))
    except Exception:
        nm_bp = buying_power
    cash = min(raw_cash, buying_power, nm_bp)

    log.info("💰 Portfolio=$" + str(round(pv, 2)) +
             " Spendable=$" + str(round(cash, 2)) +
             " BP=$" + str(round(buying_power, 2)))

    if acc.status != "ACTIVE":
        log.error("Account not active")
        return

    today = datetime.now().strftime("%Y-%m-%d")
    if state.get("trade_date") != today:
        state["daily_trades"] = 0
        state["trade_date"]   = today

    log.info("📡 Fetching market data for " + str(len(CRYPTO_UNIVERSE)) + " pairs...")
    all_bars = fetch_bars(api, CRYPTO_UNIVERSE, hours=200)
    log.info("📡 Got " + str(len(all_bars)) + " pairs")

    regime = "unknown"
    if "BTC/USD" in all_bars:
        regime = detect_regime(all_bars["BTC/USD"])
    log.info("🌍 Regime: " + regime.upper())
    if regime == "crash":
        log.warning("🚨 CRASH mode — very limited buying")
    market_bearish = False  # will be set after analysis

    # ── Intelligence Modules ──
    sentiment_data   = None
    correlation_data = None
    sentiment_mult   = 1.0
    lagging_syms     = set()

    if MODULES_LOADED:
        try:
            sentiment_data = get_sentiment()
            sentiment_mult = sentiment_data.get("multiplier", 1.0)
            log.info("😨 Sentiment: " + sentiment_data.get("regime", "unknown") +
                     " score=" + str(sentiment_data.get("score", 50)) +
                     " mult=" + str(sentiment_mult))
        except Exception as e:
            log.debug("Sentiment error: " + str(e))

        try:
            if "BTC/USD" in all_bars:
                correlation_data = get_correlation_signals(all_bars["BTC/USD"], all_bars)
                lagging          = correlation_data.get("lagging_alts", [])
                lagging_syms     = {x["symbol"].replace("/", "") for x in lagging}
        except Exception as e:
            log.debug("Correlation error: " + str(e))

        # Risk check
        try:
            wins   = state.get("wins", 0)
            losses = state.get("losses", 0)
            total  = wins + losses
            recent_wr = wins / total if total > 0 else 0.5
            risk  = update_risk_state(risk, pv)
            risk_check = check_risk(risk, pv, recent_wr, len(crypto_pos))
            if not risk_check["can_trade"]:
                log.warning("🛑 Risk blocked: " + risk_check["reason"])
                save_risk_state(risk)
                # Still check exits but skip buys
                market_bearish = True
        except Exception as e:
            log.debug("Risk check error: " + str(e))
            risk_check = {"can_trade": True, "warnings": []}

    all_pos    = api.list_positions()
    crypto_pos = [p for p in all_pos if "USD" in p.symbol and len(p.symbol) <= 8]
    n_pos      = len(crypto_pos)
    peak_prices = state.get("peak_prices", {})
    log.info("📊 Positions: " + str(n_pos) + "/" + str(MAX_POSITIONS))

    all_analyses = {}
    btc_df = all_bars.get("BTC/USD")

    for pair, df in all_bars.items():
        analysis = analyse_coin(df, agent, regime)
        if not analysis:
            continue

        # Add momentum signals
        if MODULES_LOADED:
            try:
                mom = get_momentum_score(df, btc_df, pair)
                analysis["buy_score"]  += mom["momentum_score"] * 0.5
                analysis["final_score"] = analysis["buy_score"]
                analysis["momentum"]    = mom
                for sig in mom.get("signals_fired", []):
                    if sig not in analysis["signals_fired"]:
                        analysis["signals_fired"].append(sig)
            except Exception as e:
                log.debug("Momentum error " + pair + ": " + str(e))

            # Add pattern signals
            try:
                pat = detect_patterns(df)
                if pat["score"] > 0:
                    analysis["buy_score"]  += pat["score"] * 0.4
                    analysis["final_score"] = analysis["buy_score"]
                    for p in pat.get("patterns", []):
                        analysis["signals_fired"].append("pattern_" + p)
                elif pat["score"] < 0:
                    analysis["sell_score"] += abs(pat["score"]) * 0.4
                analysis["patterns"] = pat
            except Exception as e:
                log.debug("Pattern error " + pair + ": " + str(e))

            # Apply sentiment multiplier
            try:
                analysis["buy_score"]   *= sentiment_mult
                analysis["final_score"]  = analysis["buy_score"]
            except Exception:
                pass

            # Boost lagging alts (BTC leading signal)
            pos_sym = pair.replace("/", "")
            if pos_sym in lagging_syms:
                analysis["buy_score"]  *= 1.3
                analysis["final_score"] = analysis["buy_score"]
                analysis["signals_fired"].append("btc_lagging_boost")

        # Update signal based on new scores
        sig_min = agent["thresholds"]["signal_min"]
        if analysis["buy_score"] >= sig_min:
            analysis["signal"] = "BUY"
        elif analysis["sell_score"] >= sig_min:
            analysis["signal"] = "SELL"

        analysis["symbol"] = pair
        all_analyses[pair] = analysis

    # Check broad market health using fresh analysis data
    bearish_count  = sum(1 for a in all_analyses.values() if a.get("trend_24h", 0) < -0.05)
    market_bearish = bearish_count >= 8
    if market_bearish:
        log.warning("📉 BROAD MARKET DOWN (" + str(bearish_count) + "/14 coins -5%+ today) — pausing buys")
    elif regime == "crash":
        market_bearish = True

    exits, peak_prices = check_exits(api, crypto_pos, peak_prices, agent, state, all_analyses)
    if exits:
        state["trades"].extend(exits)
        state["daily_trades"] += len(exits)
        state["total_trades"]  = state.get("total_trades", 0) + len(exits)
        crypto_pos  = [p for p in api.list_positions() if "USD" in p.symbol and len(p.symbol) <= 8]
        n_pos       = len(crypto_pos)
        acc2        = api.get_account()
        cash        = min(float(acc2.cash), float(acc2.buying_power))

    held      = {p.symbol for p in crypto_pos}
    all_buys  = []
    all_sells = []

    for pair, analysis in all_analyses.items():
        pos_sym = pair.replace("/", "")
        if not agent_should_trade(agent, pos_sym):
            continue
        if analysis["signal"] == "BUY" and pos_sym not in held:
            all_buys.append(analysis)
        elif analysis["signal"] in ("SELL", "BEARISH") and pos_sym in held:
            all_sells.append(analysis)

    all_buys.sort(key=lambda x: (x.get("confidence", 0), x.get("final_score", 0)), reverse=True)
    if market_bearish:
        all_buys = []  # no new buys when market broadly declining
    log.info("📊 Candidates | Buys=" + str(len(all_buys)) + " Sells=" + str(len(all_sells)))

    for a in all_buys[:3]:
        log.info("  🎯 " + a["symbol"] +
                 " score=" + str(a["final_score"]) +
                 " conf=" + str(a["confidence"]) +
                 " RSI=" + str(a["rsi"]) +
                 " slope=" + str(a["trend_slope"]) +
                 " sigs=" + str(a.get("signals_fired", [])))

    for analysis in all_sells:
        pair    = analysis["symbol"]
        pos_sym = pair.replace("/", "")
        pos     = next((p for p in crypto_pos if p.symbol == pos_sym), None)
        if not pos:
            continue
        qty  = float(pos.qty)
        plpc = float(pos.unrealized_plpc)
        order = place_order(api, pair, qty, "sell",
                            "signal_sell score=" + str(round(analysis["sell_score"], 1)))
        if order:
            agent_record_close(agent, pos_sym, analysis["price"], "signal_sell")
            state["trades"].append({**order, "symbol": pos_sym})
            state["daily_trades"] += 1
            state["total_trades"]  = state.get("total_trades", 0) + 1
            if plpc > 0: state["wins"]   = state.get("wins",   0) + 1
            else:        state["losses"] = state.get("losses", 0) + 1
            held.discard(pos_sym)
            n_pos -= 1
            peak_prices.pop(pos_sym, None)
            cash += float(pos.market_value)

    slots    = MAX_POSITIONS - n_pos
    min_cash = pv * MIN_CASH_BUFFER

    # ── Grid Trading ──
    if MODULES_LOADED and not market_bearish:
        try:
            grid_orders, grid_state = run_grid(api, all_bars, cash, held, grid_state)
            if grid_orders:
                state["trades"].extend(grid_orders)
                state["daily_trades"] += len(grid_orders)
                state["total_trades"]  = state.get("total_trades", 0) + len(grid_orders)
                log.info("🔲 Grid executed " + str(len(grid_orders)) + " orders")
        except Exception as e:
            log.debug("Grid error: " + str(e))

    # ── DCA Trading ──
    if MODULES_LOADED and not market_bearish:
        try:
            dca_orders, dca_state = run_dca(api, all_bars, all_analyses, cash, held, dca_state)
            if dca_orders:
                state["trades"].extend(dca_orders)
                state["daily_trades"] += len(dca_orders)
                state["total_trades"]  = state.get("total_trades", 0) + len(dca_orders)
                log.info("📊 DCA executed " + str(len(dca_orders)) + " orders")
        except Exception as e:
            log.debug("DCA error: " + str(e))

    # Get risk sizing multiplier
    risk_size_mult = 1.0
    if MODULES_LOADED:
        try:
            risk_size_mult = get_position_size_multiplier(
                risk, pv, atr_pct=0.02
            )
            if risk_size_mult < 1.0:
                log.info("🛡️  Risk reducing position size to " + str(round(risk_size_mult * 100)) + "%")
        except Exception:
            pass

    for analysis in all_buys[:slots]:
        pair  = analysis["symbol"]
        price = analysis["price"]
        conf  = analysis.get("confidence", 0.5)
        atr   = analysis.get("atr_pct",    0.02)

        qty  = calc_qty(cash * risk_size_mult, price, conf, atr)
        cost = qty * price if qty > 0 else 0

        log.info("💡 " + pair +
                 " qty=" + str(qty) +
                 " cost=$" + str(round(cost, 2)) +
                 " avail=$" + str(round(cash, 2)))

        if qty == 0.0:
            log.info("⚠️  Skip " + pair + " — zero qty")
            continue
        if cost < MIN_ORDER_USD:
            log.info("⚠️  Skip " + pair + " — $" + str(round(cost, 2)) + " < min")
            continue
        if (cash - cost) < min_cash:
            log.info("⚠️  Skip " + pair + " — cash buffer")
            continue

        order = place_order(api, pair, qty, "buy",
                            "score=" + str(analysis["final_score"]) +
                            " conf=" + str(conf) +
                            " " + regime)
        if order:
            pos_sym = pair.replace("/", "")
            agent_record_open(agent, pos_sym, price, analysis)
            state["trades"].append({**order, "symbol": pos_sym, "price": price})
            state["daily_trades"] += 1
            state["total_trades"]  = state.get("total_trades", 0) + 1
            peak_prices[pos_sym]   = price
            cash  -= cost
            n_pos += 1

    state["signals"]     = all_analyses
    state["last_run"]    = datetime.now().isoformat()
    state["peak_prices"] = peak_prices
    save_state(state)
    save_agent(agent)
    if MODULES_LOADED:
        try:
            save_risk_state(risk)
            save_grid_state(grid_state)
            save_dca_state(dca_state)
        except Exception:
            pass

    wins   = state.get("wins",   0)
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
        "generated_at":  datetime.now().isoformat(),
        "sentiment":     sentiment_data,
        "correlation":   correlation_data,
        "account": {
            "portfolio_value": pv,
            "cash":            cash,
            "buying_power":    buying_power,
            "status":          acc.status,
        },
        "market_regime": regime,
        "last_run":      state["last_run"],
        "positions":     live_pos,
        "trades":        state["trades"][-50:],
        "signals":       all_analyses,
        "agent": {
            "total_learned":      agent["total_learned"],
            "win_rate_7d":        agent["win_rate_7d"],
            "win_rate_30d":       agent["win_rate_30d"],
            "avg_pnl":            agent["avg_pnl"],
            "sharpe":             agent["sharpe"],
            "thresholds":         agent["thresholds"],
            "weights":            agent["weights"],
            "last_calibrated":    agent["last_calibrated"],
            "calibration_log":    agent["calibration_log"][-5:],
            "signal_performance": agent.get("signal_type_performance", {}),
        },
        "grid": {
            "active_grids": list(grid_state.get("grids", {}).keys()),
            "total_grid_buys":  sum(g.get("total_buys", 0)  for g in grid_state.get("grids", {}).values()),
            "total_grid_sells": sum(g.get("total_sells", 0) for g in grid_state.get("grids", {}).values()),
            "estimated_pnl":    sum(g.get("estimated_pnl", 0) for g in grid_state.get("grids", {}).values()),
        },
        "dca": {
            "active_positions": list(dca_state.get("positions", {}).keys()),
            "total_dca_positions": len(dca_state.get("positions", {})),
        },
        "grid": {
            "total_profit": grid_state.get("total_profit", 0) if MODULES_LOADED else 0,
            "active_grids": len(grid_state.get("grids", {})) if MODULES_LOADED else 0,
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
             " W/L=" + str(wins) + "/" + str(losses) +
             " Positions=" + str(n_pos) +
             " Regime=" + regime)
    log.info("=" * 60)


if __name__ == "__main__":
    run_bot()
