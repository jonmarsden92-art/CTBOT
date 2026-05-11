"""
predictor.py - XGBoost Machine Learning Predictor
===================================================
[Full docstring unchanged]
"""

import json
import logging
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TRAINING_FILE  = Path("logs/training_data.json")
MODEL_FILE     = Path("logs/xgb_model.json")
MIN_TRAIN_ROWS = 200
RETRAIN_EVERY  = 50
OUTCOME_HOURS  = 2
PROFIT_TARGET  = 0.015

def load_training_data() -> list:
    if TRAINING_FILE.exists():
        try:
            return json.loads(TRAINING_FILE.read_text())
        except Exception:
            pass
    return []

def save_training_data(data: list):
    TRAINING_FILE.parent.mkdir(exist_ok=True)
    data = data[-5000:]
    TRAINING_FILE.write_text(json.dumps(data, default=str))

def extract_features(analysis: dict, sentiment_score: float = 50) -> dict:
    """
    Extract ML features from a coin analysis dict.
    Uses .get() with safe defaults to avoid KeyError.
    """
    return {
        # Price momentum
        "rsi":           analysis.get("rsi", 50),
        "rsi_norm":      (analysis.get("rsi", 50) - 50) / 50,
        "bb_pct":        analysis.get("bb_pct", 0.5),
        "macd_hist":     float(analysis.get("macd_hist", 0)),
        "trend_1h":      analysis.get("trend_1h", 0),
        "trend_slope":   analysis.get("trend_slope", 0),
        "trend_4h":      analysis.get("trend_4h", 0),
        "trend_24h":     analysis.get("trend_24h", 0),

        # Volume
        "vol_ratio":     min(analysis.get("vol_ratio", 1), 5),
        "vol_strong":    1 if analysis.get("vol_strong", False) else 0,

        # Pattern signals
        "stabilising":   1 if analysis.get("stabilising", False) else 0,
        "near_support":  1 if analysis.get("near_support", False) else 0,
        "rsi_div_bull":  1 if analysis.get("rsi_divergence") == "bullish" else 0,
        "rsi_div_bear":  1 if analysis.get("rsi_divergence") == "bearish" else 0,

        # MA alignment - safe default 0 if not provided
        "sma3_vs_sma9":  analysis.get("sma3_vs_sma9", 0),
        "sma9_vs_sma21": analysis.get("sma9_vs_sma21", 0),

        # ATR / volatility
        "atr_pct":       analysis.get("atr_pct", 0.02),

        # Sentiment
        "fear_greed":    sentiment_score / 100,
        "fear_extreme":  1 if sentiment_score < 25 else 0,
        "greed_extreme": 1 if sentiment_score > 75 else 0,

        # Time features
        "hour":          datetime.now().hour / 24,
        "hour_sin":      math.sin(2 * math.pi * datetime.now().hour / 24),
        "hour_cos":      math.cos(2 * math.pi * datetime.now().hour / 24),

        # Score from rule-based system (as a feature, not the final answer)
        "rule_buy_score":  min(analysis.get("buy_score", 0) / 10, 2),
        "rule_sell_score": min(analysis.get("sell_score", 0) / 10, 2),
    }

def record_training_example(training_data: list, symbol: str,
                              analysis: dict, sentiment_score: float = 50):
    features = extract_features(analysis, sentiment_score)
    example  = {
        "symbol":    symbol,
        "time":      datetime.now().isoformat(),
        "price":     analysis.get("price", 0),
        "features":  features,
        "outcome":   None,
        "signal":    analysis.get("signal", "HOLD"),
    }
    training_data.append(example)
    return training_data

def update_outcomes(training_data: list, all_bars: dict) -> tuple:
    n_updated  = 0
    cutoff     = datetime.now() - timedelta(hours=OUTCOME_HOURS)

    for example in training_data:
        if example.get("outcome") is not None:
            continue

        try:
            example_time = datetime.fromisoformat(example["time"])
        except Exception:
            example["outcome"] = -1
            continue

        if example_time > cutoff:
            continue

        symbol  = example["symbol"]
        pair    = symbol if "/" in symbol else symbol[:-3] + "/USD"
        entry   = example.get("price", 0)

        if pair in all_bars and entry > 0:
            current_price = float(all_bars[pair]["close"].iloc[-1])
            pnl_pct       = (current_price - entry) / entry
            outcome       = 1 if pnl_pct >= PROFIT_TARGET else 0
            example["outcome"]   = outcome
            example["actual_pnl"] = round(pnl_pct * 100, 3)
            n_updated += 1
        else:
            example["outcome"] = -1

    return training_data, n_updated

def get_training_arrays(training_data: list):
    valid = [e for e in training_data
             if e.get("outcome") in (0, 1)
             and e.get("features")]

    if len(valid) < MIN_TRAIN_ROWS:
        return None, None, len(valid)

    feature_keys = sorted(valid[0]["features"].keys())
    X = np.array([[e["features"].get(k, 0) for k in feature_keys] for e in valid])
    y = np.array([e["outcome"] for e in valid])

    return X, y, len(valid)

def train_model(training_data: list) -> Optional[object]:
    try:
        import xgboost as xgb
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler

        X, y, n = get_training_arrays(training_data)
        if X is None:
            log.info("🤖 ML: need " + str(MIN_TRAIN_ROWS - n) + " more examples before training (have " + str(n) + ")")
            return None

        pos_rate = float(y.mean())
        log.info("🤖 ML: training on " + str(n) + " examples | positive rate=" + str(round(pos_rate * 100, 1)) + "%")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=(1 - pos_rate) / max(pos_rate, 0.01),
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
        model.fit(X_scaled, y)

        try:
            cv_scores = cross_val_score(model, X_scaled, y, cv=3, scoring="roc_auc")
            log.info("🤖 ML: CV AUC=" + str(round(cv_scores.mean(), 3)) +
                     " ± " + str(round(cv_scores.std(), 3)))
        except Exception:
            cv_scores = None

        feature_keys = sorted(valid[0]["features"].keys())
        importances  = model.feature_importances_
        top_features = sorted(zip(feature_keys, importances), key=lambda x: x[1], reverse=True)[:5]
        log.info("🤖 ML top features: " + " | ".join(f + "=" + str(round(imp, 3)) for f, imp in top_features))

        model_data = {
            "trained_at":   datetime.now().isoformat(),
            "n_examples":   n,
            "feature_keys": feature_keys,
            "cv_auc":       round(float(cv_scores.mean()), 3) if cv_scores is not None else 0,
            "top_features": [(f, round(float(imp), 3)) for f, imp in top_features],
            "scaler_mean":  scaler.mean_.tolist(),
            "scaler_std":   scaler.scale_.tolist(),
        }

        model.save_model(str(MODEL_FILE))
        meta_file = MODEL_FILE.with_suffix(".meta.json")
        meta_file.write_text(json.dumps(model_data, indent=2))

        log.info("🤖 ML: model saved")
        return (model, scaler, feature_keys)

    except ImportError:
        log.warning("🤖 ML: xgboost/sklearn not installed — skipping training")
        return None
    except Exception as e:
        log.error("🤖 ML training error: " + str(e))
        return None

def load_model() -> Optional[tuple]:
    try:
        import xgboost as xgb
        from sklearn.preprocessing import StandardScaler
        import numpy as np

        if not MODEL_FILE.exists():
            return None

        meta_file = MODEL_FILE.with_suffix(".meta.json")
        if not meta_file.exists():
            return None

        meta    = json.loads(meta_file.read_text())
        model   = xgb.XGBClassifier()
        model.load_model(str(MODEL_FILE))

        scaler        = StandardScaler()
        scaler.mean_  = np.array(meta["scaler_mean"])
        scaler.scale_ = np.array(meta["scaler_std"])
        scaler.var_   = scaler.scale_ ** 2
        scaler.n_features_in_ = len(meta["feature_keys"])

        trained_at = meta.get("trained_at", "unknown")
        n_examples = meta.get("n_examples", 0)
        cv_auc     = meta.get("cv_auc", 0)
        log.info("🤖 ML model loaded | trained=" + trained_at[:16] +
                 " n=" + str(n_examples) +
                 " AUC=" + str(cv_auc))

        return (model, scaler, meta["feature_keys"])

    except Exception as e:
        log.debug("🤖 ML load error: " + str(e))
        return None

def predict_probability(model_tuple: tuple, analysis: dict,
                         sentiment_score: float = 50) -> float:
    if model_tuple is None:
        return 0.5

    try:
        model, scaler, feature_keys = model_tuple
        features = extract_features(analysis, sentiment_score)
        X = np.array([[features.get(k, 0) for k in feature_keys]])
        X_scaled = scaler.transform(X)
        prob = float(model.predict_proba(X_scaled)[0][1])
        return round(prob, 3)
    except Exception as e:
        log.debug("🤖 Prediction error: " + str(e))
        return 0.5

def run_predictor(training_data: list, all_bars: dict,
                   model_tuple: Optional[tuple]) -> tuple:
    training_data, n_new = update_outcomes(training_data, all_bars)
    if n_new > 0:
        log.info("🤖 ML: " + str(n_new) + " new outcomes recorded")
        save_training_data(training_data)

    labelled = sum(1 for e in training_data if e.get("outcome") in (0, 1))

    should_retrain = (
        model_tuple is None and labelled >= MIN_TRAIN_ROWS or
        labelled > 0 and labelled % RETRAIN_EVERY == 0 and n_new > 0
    )

    if should_retrain:
        log.info("🤖 ML: retraining on " + str(labelled) + " examples...")
        new_model = train_model(training_data)
        if new_model:
            model_tuple = new_model
    elif labelled < MIN_TRAIN_ROWS:
        remaining = MIN_TRAIN_ROWS - labelled
        log.info("🤖 ML: collecting data — " + str(labelled) + "/" +
                 str(MIN_TRAIN_ROWS) + " examples (" + str(remaining) + " more needed)")

    return training_data, model_tuple
