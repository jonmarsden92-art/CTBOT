"""
risk.py - Professional Risk Management with Kelly Criterion
============================================================
Combines:
  - Proper Kelly Criterion position sizing (Quarter Kelly)
  - Daily loss limit protection
  - Maximum drawdown protection
  - Consecutive loss guard
  - Volatility-adjusted sizing
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

RISK_FILE = Path("logs/risk_state.json")

# Risk limits - AGGRESSIVE SETTINGS
DAILY_LOSS_LIMIT     = 0.10   # allow 10% daily loss
MAX_DRAWDOWN         = 0.25   # allow 25% drawdown
MAX_CONSECUTIVE_LOSS = 3      # resume trading after 3 losses
MIN_PORTFOLIO_USD    = 15.0   # never trade below this


def load_risk_state() -> dict:
    if RISK_FILE.exists():
        try:
            return json.loads(RISK_FILE.read_text())
        except Exception:
            pass
    return {
        "peak_portfolio":     0.0,
        "daily_start_value":  0.0,
        "trade_date":         None,
        "consecutive_losses": 0,
        "consecutive_wins":   0,
        "paused_until":       None,
        "pause_reason":       None,
        "daily_pnl":          0.0,
        "total_risk_blocks":  0,
    }


def save_risk_state(risk: dict):
    RISK_FILE.parent.mkdir(exist_ok=True)
    RISK_FILE.write_text(json.dumps(risk, indent=2, default=str))


def calculate_kelly_size(balance: float, agent: dict) -> float:
    """
    Proper Kelly Criterion position sizing.
    Uses Quarter Kelly (0.25x) to avoid over-leveraging in volatile crypto.
    Falls back to fixed 8% if win rate data is insufficient.
    """
    try:
        win_rate = agent.get("win_rate_30d", 0.5)
        tp = agent.get("thresholds", {}).get("take_profit", 0.025)
        sl = agent.get("thresholds", {}).get("stop_loss",   0.03)

        total_learned = agent.get("total_learned", 0)
        if total_learned < 10:
            return round(balance * 0.08, 2)

        b = tp / sl if sl > 0 else 1.0
        kelly_pct = (win_rate * (b + 1) - 1) / b
        safe_kelly = max(0.0, kelly_pct * 0.25)
        final_pct = min(0.15, safe_kelly)
        if kelly_pct > 0:
            final_pct = max(0.03, final_pct)

        size = round(balance * final_pct, 2)
        log.debug(f"Kelly: WR={round(win_rate*100)}% b={round(b,2)} kelly={round(kelly_pct*100,1)}% quarter={round(final_pct*100,1)}% size=${size}")
        return size
    except Exception as e:
        log.error(f"Kelly error: {e}")
        return round(balance * 0.08, 2)


def get_position_size(agent: dict, signal_score: float, cash: float) -> float:
    """
    Final position size combining Kelly + signal confidence.
    Returns dollar amount to allocate to trade.
    """
    try:
        base_size = calculate_kelly_size(cash, agent)
        confidence_mult = max(0.5, min(1.3, signal_score / 7.0))
        final_size = base_size * confidence_mult
        final_size = min(final_size, cash * 0.90)
        return round(final_size, 2)
    except Exception as e:
        log.error(f"Sizing error: {e}")
        return round(cash * 0.08, 2)


def update_risk_state(risk: dict, portfolio_value: float,
                      last_trade_won: Optional[bool] = None) -> dict:
    today = datetime.now().strftime("%Y-%m-%d")

    if risk.get("trade_date") != today:
        risk["daily_start_value"] = portfolio_value
        risk["daily_pnl"]         = 0.0
        risk["trade_date"]        = today
        log.info(f"📅 New trading day — start=${round(portfolio_value, 2)}")

    if portfolio_value > risk.get("peak_portfolio", 0):
        risk["peak_portfolio"] = portfolio_value

    if risk["daily_start_value"] > 0:
        risk["daily_pnl"] = (portfolio_value - risk["daily_start_value"]) / risk["daily_start_value"]

    if last_trade_won is True:
        risk["consecutive_losses"] = 0
        risk["consecutive_wins"]   = risk.get("consecutive_wins", 0) + 1
    elif last_trade_won is False:
        risk["consecutive_wins"]   = 0
        risk["consecutive_losses"] = risk.get("consecutive_losses", 0) + 1

    return risk


def check_risk(risk: dict, portfolio_value: float,
               recent_win_rate: float, n_positions: int) -> dict:
    warnings  = []
    can_trade = True
    reason    = ""

    if risk.get("paused_until"):
        try:
            pause_dt = datetime.fromisoformat(risk["paused_until"])
            if datetime.now() < pause_dt:
                can_trade = False
                reason = f"Paused until {pause_dt.strftime('%H:%M')} ({risk.get('pause_reason', 'risk')})"
                return {"can_trade": False, "reason": reason, "warnings": warnings}
            else:
                risk["paused_until"] = None
                risk["pause_reason"] = None
                log.info("✅ Risk pause lifted")
        except Exception:
            risk["paused_until"] = None

    if portfolio_value < MIN_PORTFOLIO_USD:
        can_trade = False
        reason = f"Portfolio ${round(portfolio_value,2)} below minimum"

    daily_pnl = risk.get("daily_pnl", 0)
    if daily_pnl <= -DAILY_LOSS_LIMIT:
        can_trade = False
        reason = f"Daily loss limit: {round(daily_pnl * 100, 1)}%"
        tomorrow = datetime.now().replace(hour=1, minute=0, second=0) + timedelta(days=1)
        risk["paused_until"] = tomorrow.isoformat()
        risk["pause_reason"] = "daily_loss_limit"
        log.warning("🛑 Daily loss limit hit")

    peak = risk.get("peak_portfolio", portfolio_value)
    drawdown = (peak - portfolio_value) / peak if peak > 0 else 0
    if drawdown >= MAX_DRAWDOWN:
        can_trade = False
        reason = f"Max drawdown: -{round(drawdown * 100, 1)}%"
        log.warning("🛑 Max drawdown hit")
    elif drawdown > MAX_DRAWDOWN * 0.7:
        warnings.append("drawdown_warning")

    consec_loss = risk.get("consecutive_losses", 0)
    if consec_loss >= MAX_CONSECUTIVE_LOSS:
        can_trade = False
        reason = f"{consec_loss} consecutive losses — cooling down"
        resume = datetime.now() + timedelta(minutes=45)
        risk["paused_until"] = resume.isoformat()
        risk["pause_reason"] = "consecutive_losses"
        log.warning(f"🛑 {consec_loss} consecutive losses — 45 min pause")
    elif consec_loss >= 3:
        warnings.append(f"losing_streak_{consec_loss}")

    if can_trade:
        log.info(f"🛡️  Risk OK | DailyPnL={round(daily_pnl * 100, 1)}% | ConsecLoss={risk.get('consecutive_losses',0)} | Drawdown={round(drawdown * 100, 1)}%")

    return {
        "can_trade": can_trade,
        "reason":    reason,
        "warnings":  warnings,
        "daily_pnl": daily_pnl,
        "drawdown":  round(drawdown, 4),
    }


def get_position_size_multiplier(risk: dict, portfolio_value: float,
                                  atr_pct: float = 0.02) -> float:
    mult = 1.0
    consec_loss = risk.get("consecutive_losses", 0)
    if consec_loss >= 4:
        mult *= 0.4
    elif consec_loss >= 3:
        mult *= 0.6
    elif consec_loss >= 2:
        mult *= 0.8

    peak = risk.get("peak_portfolio", portfolio_value)
    drawdown = (peak - portfolio_value) / peak if peak > 0 else 0
    if drawdown > 0.08:
        mult *= 0.5
    elif drawdown > 0.04:
        mult *= 0.75

    if atr_pct > 0.05:
        mult *= 0.6
    elif atr_pct > 0.03:
        mult *= 0.8

    daily_pnl = risk.get("daily_pnl", 0)
    if daily_pnl < -0.03:
        mult *= 0.5
    elif daily_pnl < -0.015:
        mult *= 0.75

    return max(0.3, min(1.0, round(mult, 2)))


def record_trade_result(risk: dict, won: bool, pnl_pct: float) -> dict:
    if won:
        risk["consecutive_losses"] = 0
        risk["consecutive_wins"]   = risk.get("consecutive_wins", 0) + 1
    else:
        risk["consecutive_wins"]   = 0
        risk["consecutive_losses"] = risk.get("consecutive_losses", 0) + 1
    return risk
