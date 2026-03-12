"""
CryptoTrader — 24/7 Crypto Trading Bot
Uses Alpaca Crypto API
Trades BTC, ETH, SOL, DOGE, AVAX, LINK, XRP and more
Runs every 10 minutes around the clock
Shorter timeframes, tighter stops, faster profits
"""

import os
import json
import logging
import time
import warnings
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict
import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Logging ───────────────────────────────────────────────────────────────────
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

# ── Config ────────────────────────────────────────────────────────────────────
ALPACA_API_KEY    = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]
ALPACA_BASE_URL   = os.environ.get("ALPACA_BASE_URL", "https://api.alpaca.markets")

# Crypto pairs to trade — all USD pairs supported by Alpaca
CRYPTO_UNIVERSE = [
    "BTC/USD",   # Bitcoin
    "ETH/USD",   # Ethereum
    "SOL/USD",   # Solana
    "DOGE/USD",  # Dogecoin
    "AVAX/USD",  # Avalanche
    "LINK/USD",  # Chainlink
    "XRP/USD",   # Ripple
    "LTC/USD",   # Litecoin
    "BCH/USD",   # Bitcoin Cash
    "DOT/USD",   # Polkadot
    "UNI/USD",   # Uniswap
    "AAVE/USD",  # Aave
    "SHIB/USD",  # Shiba Inu
    "PEPE/USD",  # Pepe
]

# Strategy — tuned for crypto's faster moves
SHORT_WINDOW      = 6       # faster MA for crypto
LONG_WINDOW       = 18      # shorter long MA
RSI_PERIOD        = 8       # faster RSI
RSI_OVERSOLD      = 35
RSI_OVERBOUGHT    = 68
MAX_POSITIONS     = 6       # max concurrent crypto positions
POSITION_SIZE     = 0.18    # 18% per position
STOP_LOSS_PCT     = 0.04    # 4% stop loss (crypto moves fast)
TAKE_PROFIT_PCT   = 0.04    # 4% take profit (quick gains)
TRAILING_STOP_PCT = 0.03    # 3% trailing stop
MIN_CASH_BUFFER   = 0.05
SIGNAL_THRESHOLD  = 3
LOOKBACK_DAYS     = 30      # crypto only needs 30 days history
MIN_ORDER_USD     = 1.0     # minimum $1 order

STATE_FILE  = Path("logs/crypto_state.json")
REPORT_FILE = Path("logs/crypto_report.json")


# ── State ─────────────────────────────────────────────────────────────────────
def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "trades": [], "signals": {}, "last_run": None,
        "daily_trades": 0, "trade_date": None,
        "peak_prices": {}, "total_trades": 0,
        "wins": 0, "losses": 0,
    }


def save_state(state: dict):
    STATE_FILE.parent.mkdir(exist_ok=True)
    state["trades"] = state["trades"][-500:]
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ── API ───────────────────────────────────────────────────────────────────────
def get_api():
    return tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version="v2")


def get_account(api) -> dict:
    acc = api.get_account()
    return {
        "portfolio_value": float(acc.portfolio_value),
        "cash":            float(acc.cash),
        "buying_power":    float(acc.buying_power),
        "equity":          float(acc.equity),
        "status":          acc.status,
    }


# ── Market Data ───────────────────────────────────────────────────────────────
def fetch_crypto_bars(api, symbols: List[str], days: int = 35) -> Dict[str, pd.DataFrame]:
    """Fetch daily bars for crypto symbols using Alpaca crypto data feed."""
    result = {}
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    for sym in symbols:
        try:
            bars = api.get_crypto_bars(
                sym,
                tradeapi.rest.TimeFrame.Hour,  # hourly bars for crypto = more signals
                start=start.isoformat(),
                end=end.isoformat(),
            ).df

            if bars is None or bars.empty:
                continue

            bars.index = pd.to_datetime(bars.index)
            if len(bars) >= LONG_WINDOW + 5:
                result[sym] = bars

        except Exception as e:
            log.debug(f"Bars error {sym}: {e}")

    return result


def get_latest_price(api, symbol: str) -> Optional[float]:
    """Get latest crypto price."""
    try:
        quote = api.get_latest_crypto_quote(symbol)
        if quote:
            return float(quote.ap)  # ask price
    except Exception:
        pass
    try:
        bars = api.get_crypto_bars(
            symbol,
            tradeapi.rest.TimeFrame.Minute,
            start=(datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat(),
            end=datetime.now(timezone.utc).isoformat(),
        ).df
        if not bars.empty:
            return float(bars["close"].iloc[-1])
    except Exception:
        pass
    return None


# ── Indicators ────────────────────────────────────────────────────────────────
def compute_indicators(df: pd.DataFrame) -> Optional[dict]:
    try:
        close = df["close"].squeeze().astype(float)
        vol   = df["volume"].squeeze().astype(float)

        if len(close) < LONG_WINDOW + 5:
            return None

        sma_short = close.rolling(SHORT_WINDOW).mean()
        sma_long  = close.rolling(LONG_WINDOW).mean()

        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(RSI_PERIOD).mean()
        loss  = (-delta.clip(upper=0)).rolling(RSI_PERIOD).mean()
        rs    = gain / loss.replace(0, np.nan)
        rsi   = 100 - (100 / (1 + rs))

        ema12  = close.ewm(span=12, adjust=False).mean()
        ema26  = close.ewm(span=26, adjust=False).mean()
        macd   = ema12 - ema26
        macd_s = macd.ewm(span=9, adjust=False).mean()
        hist   = macd - macd_s

        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_up  = bb_mid + 2 * bb_std
        bb_low = bb_mid - 2 * bb_std

        avg_vol   = vol.rolling(20).mean()
        vol_ratio = float(vol.iloc[-1] / avg_vol.iloc[-1]) if float(avg_vol.iloc[-1]) > 0 else 1.0

        price     = float(close.iloc[-1])
        ss        = float(sma_short.iloc[-1])
        sl        = float(sma_long.iloc[-1])
        ss_prev   = float(sma_short.iloc[-2])
        sl_prev   = float(sma_long.iloc[-2])
        rsi_val   = float(rsi.iloc[-1])
        hist_val  = float(hist.iloc[-1])
        hist_prev = float(hist.iloc[-2])
        macd_val  = float(macd.iloc[-1])
        macd_sval = float(macd_s.iloc[-1])
        bb_low_v  = float(bb_low.iloc[-1])
        bb_up_v   = float(bb_up.iloc[-1])

        buy_score = sell_score = 0

        # MA crossover
        if ss_prev <= sl_prev and ss > sl: buy_score  += 3
        if ss_prev >= sl_prev and ss < sl: sell_score += 3

        ma_gap = (ss - sl) / sl if sl > 0 else 0
        if ma_gap >  0.002: buy_score  += 1
        if ma_gap < -0.002: sell_score += 1

        # RSI
        if rsi_val < RSI_OVERSOLD:   buy_score  += 2
        if rsi_val > RSI_OVERBOUGHT: sell_score += 2

        # MACD
        if hist_val > 0 and hist_prev <= 0: buy_score  += 2
        if hist_val < 0 and hist_prev >= 0: sell_score += 2
        if macd_val > macd_sval:            buy_score  += 1
        if macd_val < macd_sval:            sell_score += 1

        # Bollinger
        if price <= bb_low_v: buy_score  += 2
        if price >= bb_up_v:  sell_score += 2

        # Volume
        if vol_ratio > 1.5:
            buy_score  = int(buy_score  * 1.2)
            sell_score = int(sell_score * 1.2)

        if buy_score >= SIGNAL_THRESHOLD:
            signal = "BUY"
        elif sell_score >= SIGNAL_THRESHOLD:
            signal = "SELL"
        elif buy_score > sell_score:
            signal = "BULLISH"
        elif sell_score > buy_score:
            signal = "BEARISH"
        else:
            signal = "HOLD"

        return {
            "price": price, "sma_short": ss, "sma_long": sl,
            "rsi": rsi_val, "macd": macd_val, "macd_signal": macd_sval,
            "macd_hist": hist_val, "bb_low": bb_low_v, "bb_up": bb_up_v,
            "vol_ratio": vol_ratio, "buy_score": buy_score,
            "sell_score": sell_score, "signal": signal,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        log.debug(f"Indicator error: {e}")
        return None


# ── Orders ────────────────────────────────────────────────────────────────────
def calc_qty(portfolio_value: float, cash: float, price: float) -> float:
    """Calculate order quantity using notional sizing."""
    alloc = min(portfolio_value * POSITION_SIZE, cash * 0.92)
    if alloc < MIN_ORDER_USD or price <= 0:
        return 0.0
    qty = alloc / price
    # Round to appropriate decimal places per coin
    if price > 1000:    # BTC, ETH
        return round(qty, 6)
    elif price > 1:     # SOL, AVAX, etc
        return round(qty, 4)
    else:               # DOGE, SHIB, PEPE
        return round(qty, 2)


def place_order(api, symbol: str, qty: float, side: str, reason: str = "") -> Optional[dict]:
    try:
        # Crypto uses gtc not day
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type="market",
            time_in_force="gtc",
        )
        log.info(f"✅ {side.upper()} {qty} {symbol} | {reason} | id={order.id}")
        return {
            "id": order.id, "symbol": symbol, "qty": qty,
            "side": side, "reason": reason,
            "time": datetime.now().isoformat(),
        }
    except Exception as e:
        log.error(f"❌ Order failed {side} {symbol}: {e}")
        return None


# ── Exits ─────────────────────────────────────────────────────────────────────
def check_exits(api, positions: list, peak_prices: dict, state: dict) -> tuple:
    exits = []
    for pos in positions:
        symbol = pos.symbol  # Alpaca returns BTCUSD format for positions
        # Convert to trading pair format for orders
        base   = symbol.replace("USD", "").replace("USDT", "")
        pair   = f"{base}/USD"

        qty   = float(pos.qty)
        plpc  = float(pos.unrealized_plpc)
        price = float(pos.current_price)

        # Update peak
        prev_peak = peak_prices.get(symbol, float(pos.avg_entry_price))
        peak      = max(prev_peak, price)
        peak_prices[symbol] = peak

        # Hard stop
        if plpc <= -STOP_LOSS_PCT:
            log.warning(f"🛑 STOP LOSS {symbol}: {plpc:.1%}")
            order = place_order(api, pair, qty, "sell", "stop_loss")
            if order:
                exits.append({**order, "symbol": symbol, "pnl_pct": plpc * 100})
                state["losses"] = state.get("losses", 0) + 1
                peak_prices.pop(symbol, None)
            continue

        # Take profit
        if plpc >= TAKE_PROFIT_PCT:
            log.info(f"🎯 TAKE PROFIT {symbol}: {plpc:.1%}")
            order = place_order(api, pair, qty, "sell", "take_profit")
            if order:
                exits.append({**order, "symbol": symbol, "pnl_pct": plpc * 100})
                state["wins"] = state.get("wins", 0) + 1
                peak_prices.pop(symbol, None)
            continue

        # Trailing stop (activates after 1.5% gain — crypto moves fast)
        entry = float(pos.avg_entry_price)
        if peak > entry * 1.015:
            trail = peak * (1 - TRAILING_STOP_PCT)
            if price <= trail:
                log.info(f"📉 TRAILING STOP {symbol}: peak={peak:.4f} now={price:.4f}")
                order = place_order(api, pair, qty, "sell", "trailing_stop")
                if order:
                    exits.append({**order, "symbol": symbol, "pnl_pct": plpc * 100})
                    if plpc > 0:
                        state["wins"] = state.get("wins", 0) + 1
                    else:
                        state["losses"] = state.get("losses", 0) + 1
                    peak_prices.pop(symbol, None)

    return exits, peak_prices


# ── Main ──────────────────────────────────────────────────────────────────────
def run_bot():
    log.info("=" * 60)
    log.info(f"🪙 CryptoTrader | {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")
    log.info("=" * 60)

    api   = get_api()
    state = load_state()

    account = get_account(api)
    log.info(f"💰 Portfolio: ${account['portfolio_value']:,.2f} | Cash: ${account['cash']:,.2f}")

    if account["status"] != "ACTIVE":
        log.error("Account not active — aborting")
        return

    # Reset daily counter
    today = datetime.now().strftime("%Y-%m-%d")
    if state.get("trade_date") != today:
        state["daily_trades"] = 0
        state["trade_date"]   = today

    # Get positions (crypto positions show as BTCUSD etc)
    positions   = api.list_positions()
    crypto_pos  = [p for p in positions if "/" not in p.symbol and len(p.symbol) > 4]
    # More reliable: check asset class
    try:
        crypto_pos = [p for p in positions if "USD" in p.symbol and not p.symbol.startswith("USD")]
    except Exception:
        pass

    held_symbols = {p.symbol for p in crypto_pos}
    n_positions  = len(crypto_pos)
    peak_prices  = state.get("peak_prices", {})

    log.info(f"📊 Open crypto positions: {n_positions}/{MAX_POSITIONS}")

    # Check exits
    exits, peak_prices = check_exits(api, crypto_pos, peak_prices, state)
    state["peak_prices"] = peak_prices
    if exits:
        state["trades"].extend(exits)
        state["daily_trades"] += len(exits)
        state["total_trades"]  = state.get("total_trades", 0) + len(exits)
        crypto_pos   = [p for p in api.list_positions() if "USD" in p.symbol]
        held_symbols = {p.symbol for p in crypto_pos}
        n_positions  = len(crypto_pos)

    # Fetch bars for all crypto
    log.info(f"🔍 Scanning {len(CRYPTO_UNIVERSE)} crypto pairs...")
    bars = fetch_crypto_bars(api, CRYPTO_UNIVERSE)
    log.info(f"📡 Got data for {len(bars)} pairs")

    all_signals = {}
    all_buys    = []
    all_sells   = []

    for pair, df in bars.items():
        ind = compute_indicators(df)
        if ind is None:
            continue
        ind["symbol"] = pair
        all_signals[pair] = ind

        # Convert pair to position symbol format (BTC/USD -> BTCUSD)
        pos_symbol = pair.replace("/", "")

        if ind["signal"] == "BUY" and pos_symbol not in held_symbols:
            all_buys.append(ind)
        elif ind["signal"] in ("SELL", "BEARISH") and pos_symbol in held_symbols:
            # Find the position
            for p in crypto_pos:
                if p.symbol == pos_symbol:
                    plpc = float(p.unrealized_plpc)
                    if ind["signal"] == "SELL" or plpc > 0.005:
                        all_sells.append(ind)
                    break

    log.info(f"📊 Signals: Buys={len(all_buys)} Sells={len(all_sells)}")

    # Execute sells
    for sig in all_sells:
        pair       = sig["symbol"]
        pos_symbol = pair.replace("/", "")
        pos        = next((p for p in crypto_pos if p.symbol == pos_symbol), None)
        if not pos:
            continue
        qty   = float(pos.qty)
        order = place_order(api, pair, qty, "sell", f"signal_sell(score={sig['sell_score']})")
        if order:
            state["trades"].append({**order, "symbol": pos_symbol})
            state["daily_trades"] += 1
            state["total_trades"]  = state.get("total_trades", 0) + 1
            held_symbols.discard(pos_symbol)
            n_positions -= 1

    # Execute buys
    slots    = MAX_POSITIONS - n_positions
    cash     = account["cash"]
    min_cash = account["portfolio_value"] * MIN_CASH_BUFFER

    if slots > 0 and all_buys:
        all_buys.sort(key=lambda x: x["buy_score"], reverse=True)
        for sig in all_buys[:slots]:
            pair  = sig["symbol"]
            price = sig["price"]
            qty   = calc_qty(account["portfolio_value"], cash, price)

            if qty == 0.0:
                log.info(f"⚠️  Skip {pair} — insufficient funds")
                continue

            cost = qty * price
            if (cash - cost) < min_cash:
                log.info(f"⚠️  Skip {pair} — low cash buffer")
                continue

            order = place_order(api, pair, qty, "buy", f"signal_buy(score={sig['buy_score']})")
            if order:
                pos_symbol = pair.replace("/", "")
                state["trades"].append({**order, "symbol": pos_symbol, "price": price})
                state["daily_trades"] += 1
                state["total_trades"]  = state.get("total_trades", 0) + 1
                peak_prices[pos_symbol] = price
                cash -= cost
                n_positions += 1
    elif slots <= 0:
        log.info(f"Positions full ({n_positions}/{MAX_POSITIONS})")
    else:
        log.info("No buy signals found")

    state["signals"]     = all_signals
    state["last_run"]    = datetime.now().isoformat()
    state["account"]     = account
    state["peak_prices"] = peak_prices
    save_state(state)

    # Save report
    wins   = state.get("wins", 0)
    losses = state.get("losses", 0)
    total  = wins + losses
    report = {
        "generated_at":   datetime.now().isoformat(),
        "account":        account,
        "last_run":       state["last_run"],
        "positions":      [],
        "trades":         state["trades"][-50:],
        "signals":        all_signals,
        "performance": {
            "total_trades":   state.get("total_trades", 0),
            "daily_trades":   state.get("daily_trades", 0),
            "wins":           wins,
            "losses":         losses,
            "win_rate":       wins / total if total > 0 else 0,
            "open_positions": n_positions,
        }
    }

    # Add live positions
    try:
        live_pos = api.list_positions()
        report["positions"] = [
            {
                "symbol":          p.symbol,
                "qty":             float(p.qty),
                "avg_entry":       float(p.avg_entry_price),
                "current_price":   float(p.current_price),
                "market_value":    float(p.market_value),
                "unrealized_pl":   float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc),
            }
            for p in live_pos if "USD" in p.symbol
        ]
    except Exception:
        pass

    REPORT_FILE.parent.mkdir(exist_ok=True)
    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2, default=str)

    log.info(f"✅ Done | Daily trades={state['daily_trades']} | W/L={wins}/{losses}")
    log.info("=" * 60)


if __name__ == "__main__":
    run_bot()
