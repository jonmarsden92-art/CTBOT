"""
risk.py - Professional Risk Management
========================================
The most important module. Even the best signals fail without proper risk control.

Rules used by professional traders:
  1. Daily loss limit — stop trading if down X% today
  2. Maximum drawdown — stop trading if portfolio drops X% from peak
  3. Position correlation risk — don't hold too many correlated coins
  4. Volatility-adjusted sizing — smaller positions in volatile markets
  5. Win rate guard — if recent win rate drops below threshold, pause
  6. Consecutive loss guard — 3 losses in a row = pause and reassess
  7. Time-based rules — avoid trading in the first/last 30 mins of sessions
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

RISK_FILE = Path("logs/risk_state.json")

# Risk limits
DAILY_LOSS_LIMIT      = 0.05   # stop trading if down 5% today
MAX_DRAWDOWN          = 0.12   # stop trading if down 12% from peak
MIN_WIN_RATE_RECENT   = 0.25   # pause if last 10 trades below 25% win rate
MAX_CONSECUTIVE_LOSS  = 4      # pause after 4 losses in a row
MAX_CORRELATED_POS    = 3      # max positions in same market direction
MIN_PORTFOLIO_USD     = 20.0   # never trade below this portfolio value


def load_risk_state() -> dict:
    if RISK_FILE.exists():
        try:
            return json.loads(RISK_FILE.read_text())
        except Exception:
            pass
    return {
        "peak_portfolio":      0.0,
        "daily_start_value":   0.0,
        "trade_date":          None,
        "consecutive_losses":  0,
        "consecutive_wins":    0,
        "paused_until":        None,
        "pause_reason":        None,
        "daily_pnl":           0.0,
        "total_risk_blocks":   0,
    }


def save_risk_state(risk: dict):
    RISK_FILE.parent.mkdir(exist_ok=True)
    RISK_FILE.write_text(json.dumps(risk, indent=2, default=str))


def update_risk_state(risk: dict, portfolio_value: float,
                      last_trade_won: Optional[bool] = None) -> dict:
    """Update risk state with current portfolio value and last trade result."""
    today = datetime.now().strftime("%Y-%m-%d")

    # Reset daily values at start of new day
    if risk.get("trade_date") != today:
        risk["daily_start_value"] = portfolio_value
        risk["daily_pnl"]         = 0.0
        risk["trade_date"]        = today
        log.info("📅 New trading day — reset daily P&L. Start=$" + str(round(portfolio_value, 2)))

    # Update peak
    if portfolio_value > risk.get("peak_portfolio", 0):
        risk["peak_portfolio"] = portfolio_value

    # Update daily P&L
    if risk["daily_start_value"] > 0:
        risk["daily_pnl"] = (portfolio_value - risk["daily_start_value"]) / risk["daily_start_value"]

    # Update consecutive loss/win counter
    if last_trade_won is True:
        risk["consecutive_losses"] = 0
        risk["consecutive_wins"]   = risk.get("consecutive_wins", 0) + 1
    elif last_trade_won is False:
        risk["consecutive_wins"]   = 0
        risk["consecutive_losses"] = risk.get("consecutive_losses", 0) + 1

    return risk


def check_risk(risk: dict, portfolio_value: float,
               recent_win_rate: float, n_positions: int) -> dict:
    """
    Check all risk rules. Returns dict with:
    - can_trade: bool
    - reason: str (why blocked if can_trade=False)
    - warnings: list of non-blocking warnings
    """
    warnings  = []
    can_trade = True
    reason    = ""

    # Check if currently paused
    if risk.get("paused_until"):
        try:
            pause_dt = datetime.fromisoformat(risk["paused_until"])
            if datetime.now() < pause_dt:
                can_trade = False
                reason    = "Paused until " + pause_dt.strftime("%H:%M") + " (" + risk.get("pause_reason", "risk limit") + ")"
                return {"can_trade": False, "reason": reason, "warnings": warnings}
            else:
                risk["paused_until"] = None
                risk["pause_reason"] = None
                log.info("✅ Risk pause lifted")
        except Exception:
            risk["paused_until"] = None

    # 1. Minimum portfolio value
    if portfolio_value < MIN_PORTFOLIO_USD:
        can_trade = False
        reason    = "Portfolio $" + str(round(portfolio_value, 2)) + " below minimum $" + str(MIN_PORTFOLIO_USD)

    # 2. Daily loss limit
    daily_pnl = risk.get("daily_pnl", 0)
    if daily_pnl <= -DAILY_LOSS_LIMIT:
        can_trade = False
        reason    = "Daily loss limit hit: " + str(round(daily_pnl * 100, 1)) + "% (limit " + str(DAILY_LOSS_LIMIT * 100) + "%)"
        risk["total_risk_blocks"] = risk.get("total_risk_blocks", 0) + 1
        # Pause until next day
        from datetime import timedelta
        tomorrow = datetime.now().replace(hour=0, minute=30, second=0) + timedelta(days=1)
        risk["paused_until"] = tomorrow.isoformat()
        risk["pause_reason"] = "daily_loss_limit"
        log.warning("🛑 DAILY LOSS LIMIT — pausing until tomorrow")

    # 3. Maximum drawdown from peak
    peak = risk.get("peak_portfolio", portfolio_value)
    if peak > 0:
        drawdown = (peak - portfolio_value) / peak
        if drawdown >= MAX_DRAWDOWN:
            can_trade = False
            reason    = "Max drawdown hit: -" + str(round(drawdown * 100, 1)) + "% from peak"
            log.warning("🛑 MAX DRAWDOWN HIT — " + str(round(drawdown * 100, 1)) + "%")
        elif drawdown > MAX_DRAWDOWN * 0.7:
            warnings.append("drawdown_warning_" + str(round(drawdown * 100, 1)) + "pct")

    # 4. Consecutive losses
    consec_loss = risk.get("consecutive_losses", 0)
    if consec_loss >= MAX_CONSECUTIVE_LOSS:
        can_trade = False
        reason    = str(consec_loss) + " consecutive losses — cooling down"
        # 30 minute pause
        from datetime import timedelta
        resume = datetime.now() + timedelta(minutes=30)
        risk["paused_until"] = resume.isoformat()
        risk["pause_reason"] = "consecutive_losses"
        log.warning("🛑 " + str(consec_loss) + " consecutive losses — 30 min pause")
    elif consec_loss >= 2:
        warnings.append("consecutive_losses_" + str(consec_loss))

    # 5. Recent win rate too low
    if recent_win_rate < MIN_WIN_RATE_RECENT and recent_win_rate > 0:
        warnings.append("low_win_rate_" + str(round(recent_win_rate * 100)) + "pct")
        if recent_win_rate < 0.20:
            can_trade = False
            reason    = "Win rate critically low: " + str(round(recent_win_rate * 100)) + "%"

    # Log risk status
    if can_trade:
        status_parts = [
            "DailyPnL=" + str(round(daily_pnl * 100, 1)) + "%",
            "ConsecLoss=" + str(risk.get("consecutive_losses", 0)),
            "WR=" + str(round(recent_win_rate * 100)) + "%",
        ]
        if warnings:
            log.warning("⚠️  Risk warnings: " + " | ".join(warnings))
        log.info("🛡️  Risk OK | " + " | ".join(status_parts))
    else:
        log.warning("🛑 TRADING BLOCKED: " + reason)

    return {
        "can_trade": can_trade,
        "reason":    reason,
        "warnings":  warnings,
        "daily_pnl": daily_pnl,
        "drawdown":  round((risk.get("peak_portfolio", portfolio_value) - portfolio_value) /
                          max(risk.get("peak_portfolio", portfolio_value), 1), 4),
    }


def get_position_size_multiplier(risk: dict, portfolio_value: float,
                                 atr_pct: float = 0.02) -> float:
    """
    Dynamic position sizing multiplier based on risk state.
    Returns 0.3 to 1.0 multiplier applied to base position size.
    """
    mult = 1.0

    # Scale down if on a losing streak
    consec_loss = risk.get("consecutive_losses", 0)
    if consec_loss >= 3:
        mult *= 0.5
    elif consec_loss >= 2:
        mult *= 0.7
    elif consec_loss >= 1:
        mult *= 0.85

    # Scale down if drawdown is significant
    peak     = risk.get("peak_portfolio", portfolio_value)
    drawdown = (peak - portfolio_value) / peak if peak > 0 else 0
    if drawdown > 0.06:
        mult *= 0.6
    elif drawdown > 0.03:
        mult *= 0.8

    # Scale down in high volatility (high ATR)
    if atr_pct > 0.05:
        mult *= 0.6
    elif atr_pct > 0.03:
        mult *= 0.8

    # Scale down if daily loss approaching limit
    daily_pnl = risk.get("daily_pnl", 0)
    if daily_pnl < -0.03:
        mult *= 0.5
    elif daily_pnl < -0.01:
        mult *= 0.8

    return max(0.3, min(1.0, round(mult, 2)))


def record_trade_result(risk: dict, won: bool, pnl_pct: float) -> dict:
    """Record the result of a closed trade and update risk state."""
    if won:
        risk["consecutive_losses"] = 0
        risk["consecutive_wins"]   = risk.get("consecutive_wins", 0) + 1
    else:
        risk["consecutive_wins"]   = 0
        risk["consecutive_losses"] = risk.get("consecutive_losses", 0) + 1
    return risk
