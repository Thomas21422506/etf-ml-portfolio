"""
Backtesting strategies with risk management
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import logging

from .utils import TRAIN_TEST_SPLIT

logger = logging.getLogger(__name__)

# ============================================================================
# PART 3: ML TRADING STRATEGY (BASE VERSION)
# ============================================================================

def realistic_ml_strategy(features, targets, weekly_returns, best_models=None, retrain_every=26):
    """
    ML allocation strategy using pre-trained models from Part 2.

    Parameters:
    -----------
    best_models : dict, optional
        Dictionary with ETF names as keys and trained models as values.
        If provided, uses these models instead of retraining.
    """
    common_idx = features.index.intersection(weekly_returns.index)
    features = features.loc[common_idx]
    weekly_returns = weekly_returns.loc[common_idx]
    targets = targets.loc[common_idx]

    split_idx = int(len(features) * TRAIN_TEST_SPLIT)
    portfolio_returns = []
    dates = []

    etf_list = ['SPY', 'QQQ', 'EEM', 'TLT', 'GLD']

    if best_models is not None:
        print(f"[PART 3] Using pre-trained ML models from Part 2")
        current_models = best_models
        use_pre_trained = True
    else:
        print(f"[PART 3] No pre-trained models provided, using periodic retraining")
        current_models = {}
        use_pre_trained = False

    print(f"[PART 3] Backtesting base ML strategy over {len(features) - split_idx} weeks")

    for i in range(split_idx, len(features) - 1):
        current_date = features.index[i]
        next_date = features.index[i + 1]

        if not use_pre_trained:
            if (i - split_idx) % retrain_every == 0 or not current_models:
                print(f"[PART 3] Retraining models as of {current_date.strftime('%Y-%m-%d')}")
                current_models = {}

                for etf in etf_list:
                    target_col = f'{etf}_target_next_week'
                    X_train = features.iloc[:i]
                    y_train = targets[target_col].iloc[:i]

                    if len(y_train) < 52:
                        continue

                    n_up = y_train.sum()
                    n_down = len(y_train) - n_up

                    if n_up > n_down:
                        class_weights = {1: 1.0, 0: n_up/n_down}
                    else:
                        class_weights = {1: n_down/n_up, 0: 1.0}

                    model = LogisticRegression(
                        random_state=42,
                        max_iter=1000,
                        class_weight=class_weights,
                        C=0.1,
                        penalty='l2'
                    )
                    model.fit(X_train, y_train)
                    current_models[etf] = model

        weekly_weights = {}
        all_predictions = []

        for etf in etf_list:
            if etf not in current_models:
                weekly_weights[etf] = 0
                continue

            model = current_models[etf]
            X_current = features.iloc[i:i+1]

            try:
                prob_up = model.predict_proba(X_current)[0, 1]

                if prob_up >= 0.50:
                    weekly_weights[etf] = prob_up
                else:
                    weekly_weights[etf] = 0

                all_predictions.append((etf, prob_up))

            except Exception as e:
                logger.warning(f"Prediction failed for {etf} at {current_date}: {str(e)}")
                weekly_weights[etf] = 0


        total_weight = sum(weekly_weights.values())

        if total_weight > 0:
            weekly_weights = {k: v/total_weight for k, v in weekly_weights.items()}

            etfs_above_threshold = [etf for etf, weight in weekly_weights.items() if weight > 0]

            if len(etfs_above_threshold) == 1:
                single_etf = etfs_above_threshold[0]
                other_probs = [(etf, prob) for etf, prob in all_predictions if etf != single_etf]

                if other_probs:
                    second_best = max(other_probs, key=lambda x: x[1])
                    if second_best[1] >= 0.40:
                        weekly_weights[second_best[0]] = 0.3
                        weekly_weights[single_etf] = 0.7
                        total = sum(weekly_weights.values())
                        weekly_weights = {k: v/total for k, v in weekly_weights.items()}
        else:
            if all_predictions:
                filtered_probs = [(etf, prob) for etf, prob in all_predictions if prob >= 0.35]

                if len(filtered_probs) >= 2:
                    top_etfs = sorted(filtered_probs, key=lambda x: x[1], reverse=True)[:3]
                    n_selected = len(top_etfs)
                    equal_weight = 1.0 / n_selected
                    weekly_weights = {etf: equal_weight for etf, _ in top_etfs}
                else:
                    weekly_weights = {etf: 0 for etf in etf_list}
            else:
                weekly_weights = {etf: 0 for etf in etf_list}

        week_return = 0

        if next_date in weekly_returns.index:
            for etf, weight in weekly_weights.items():
                if weight > 0.01:
                    return_next_week = weekly_returns.loc[next_date, etf]
                    week_return += weight * return_next_week

        portfolio_returns.append(week_return)
        dates.append(current_date)

    strategy_returns = pd.Series(portfolio_returns, index=dates)
    non_zero_returns = len([r for r in portfolio_returns if abs(r) > 0.001])
    print(f"[PART 3] Base ML strategy: {len(strategy_returns)} periods, "
          f"Trades: {non_zero_returns}, Mean return: {strategy_returns.mean():.4f}")

    return strategy_returns

# ============================================================================
# PART 3: RISK MANAGEMENT & POSITION SIZING
# ============================================================================

def calculate_kelly_criterion(prob_win, avg_win, avg_loss):
    """
    Computes optimal position size using the Kelly criterion.

    Kelly% = (p * b - q) / b
    where:
    - p = win probability
    - q = loss probability (1 - p)
    - b = average win / average loss
    """
    if avg_loss == 0:
        return 0

    b = abs(avg_win / avg_loss)
    q = 1 - prob_win

    kelly_pct = (prob_win * b - q) / b

    # Use half-Kelly and cap at 25% per position
    kelly_pct = max(0, min(kelly_pct * 0.5, 0.25))

    return kelly_pct

def calculate_risk_parity_weights(returns_history, lookback=52):
    """
    Risk parity allocation: equal risk contribution per asset.

    weight_i ∝ 1 / volatility_i
    """
    if len(returns_history) < lookback:
        n_assets = len(returns_history.columns)
        return {col: 1/n_assets for col in returns_history.columns}

    recent_returns = returns_history.iloc[-lookback:]
    volatilities = recent_returns.std()

    volatilities = volatilities.replace(0, 1e-6)

    inverse_vol = 1 / volatilities
    weights = inverse_vol / inverse_vol.sum()

    return weights.to_dict()

def calculate_var_cvar(returns, confidence=0.95):
    """
    Computes Value at Risk (VaR) and Conditional Value at Risk (CVaR).

    VaR: loss threshold at a given confidence level.
    CVaR (expected shortfall): average loss beyond the VaR.
    """
    if len(returns) == 0:
        return 0, 0

    sorted_returns = np.sort(returns)
    index = int((1 - confidence) * len(sorted_returns))

    var = abs(sorted_returns[index]) if index < len(sorted_returns) else 0
    cvar = abs(sorted_returns[:index].mean()) if index > 0 else 0

    return var, cvar

def apply_volatility_targeting(base_weights, returns_history, target_vol=0.12, lookback=26):
    """
    Adjusts portfolio weights to target an annualized volatility.

    If realized vol > target: reduce exposure.
    If realized vol < target: increase exposure (within limits).
    """
    if len(returns_history) < lookback:
        return base_weights

    recent_returns = returns_history.iloc[-lookback:]
    portfolio_returns = sum(base_weights.get(col, 0) * recent_returns[col]
                           for col in recent_returns.columns)

    realized_vol = portfolio_returns.std() * np.sqrt(52)

    if realized_vol == 0:
        return base_weights

    scaling_factor = target_vol / realized_vol
    scaling_factor = max(0.5, min(scaling_factor, 1.5))

    adjusted_weights = {k: v * scaling_factor for k, v in base_weights.items()}

    total_weight = sum(adjusted_weights.values())
    if total_weight > 1.0:
        adjusted_weights = {k: v/total_weight for k, v in adjusted_weights.items()}

    return adjusted_weights

def calculate_maximum_drawdown_constraint(cumulative_returns, max_allowed_dd=0.20):
    """
    Checks whether current drawdown exceeds a maximum allowed threshold.

    Returns an exposure reduction factor if necessary.
    """
    if len(cumulative_returns) == 0:
        return 1.0

    cumulative = (1 + cumulative_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative / running_max - 1)
    current_dd = drawdown.iloc[-1]

    if abs(current_dd) > max_allowed_dd:
        reduction_factor = max_allowed_dd / abs(current_dd)
        return max(0.3, reduction_factor)

    return 1.0

def enhanced_ml_strategy_with_risk_management(features, targets, weekly_returns, best_models=None, retrain_every=26):
    """
    Machine learning allocation strategy with advanced risk management.
    Uses pre-trained models from Part 2 if available.
    """
    common_idx = features.index.intersection(weekly_returns.index)
    features = features.loc[common_idx]
    weekly_returns = weekly_returns.loc[common_idx]
    targets = targets.loc[common_idx]

    split_idx = int(len(features) * TRAIN_TEST_SPLIT)
    portfolio_returns = []
    dates = []
    weights_history = []

    etf_list = ['SPY', 'QQQ', 'EEM', 'TLT', 'GLD']

    if best_models is not None:
        print(f"[PART 3B] Using pre-trained ML models from Part 2")
        current_models = best_models
        use_pre_trained = True
    else:
        print(f"[PART 3B] No pre-trained models provided, using periodic retraining")
        current_models = {}
        use_pre_trained = False

    print(f"[PART 3B] Backtesting ML strategy with risk management over {len(features) - split_idx} weeks")

    for i in range(split_idx, len(features) - 1):
        current_date = features.index[i]
        next_date = features.index[i + 1]

        if not use_pre_trained:
            if (i - split_idx) % retrain_every == 0 or not current_models:
                print(f"[PART 3B] Retraining models as of {current_date.strftime('%Y-%m-%d')}")
                current_models = {}

                for etf in etf_list:
                    target_col = f'{etf}_target_next_week'
                    X_train = features.iloc[:i]
                    y_train = targets[target_col].iloc[:i]

                    if len(y_train) < 52:
                        continue

                    n_up = y_train.sum()
                    n_down = len(y_train) - n_up

                    if n_up > n_down:
                        class_weights = {1: 1.0, 0: n_up/n_down}
                    else:
                        class_weights = {1: n_down/n_up, 0: 1.0}

                    model = LogisticRegression(
                        random_state=42,
                        max_iter=1000,
                        class_weight=class_weights,
                        C=0.1,
                        penalty='l2'
                    )
                    model.fit(X_train, y_train)
                    current_models[etf] = model


        ml_weights = {}
        all_predictions = []

        for etf in etf_list:
            if etf not in current_models:
                ml_weights[etf] = 0
                continue

            model = current_models[etf]
            X_current = features.iloc[i:i+1]

            try:
                prob_up = model.predict_proba(X_current)[0, 1]

                if prob_up >= 0.50:
                    ml_weights[etf] = prob_up
                else:
                    ml_weights[etf] = 0

                all_predictions.append((etf, prob_up))

            except Exception as e:
                logger.warning(f"Prediction failed for {etf} at {current_date}: {str(e)}")
                ml_weights[etf] = 0

        total_ml_weight = sum(ml_weights.values())

        if total_ml_weight > 0:
            ml_weights = {k: v/total_ml_weight for k, v in ml_weights.items()}
        else:
            if all_predictions:
                filtered_probs = [(etf, prob) for etf, prob in all_predictions if prob >= 0.35]

                if len(filtered_probs) >= 2:
                    top_etfs = sorted(filtered_probs, key=lambda x: x[1], reverse=True)[:3]
                    n_selected = len(top_etfs)
                    ml_weights = {etf: 1.0 / n_selected for etf, _ in top_etfs}
                else:
                    ml_weights = {etf: 0 for etf in etf_list}
            else:
                ml_weights = {etf: 0 for etf in etf_list}

        if i > split_idx + 52:
            risk_parity_weights = calculate_risk_parity_weights(
                weekly_returns.iloc[i-52:i],
                lookback=52
            )

            blended_weights = {}
            for etf in etf_list:
                ml_w = ml_weights.get(etf, 0)
                rp_w = risk_parity_weights.get(etf, 0)

                if ml_w > 0:
                    blended_weights[etf] = 0.7 * ml_w + 0.3 * rp_w
                else:
                    blended_weights[etf] = 0

            total = sum(blended_weights.values())
            if total > 0:
                blended_weights = {k: v/total for k, v in blended_weights.items()}
            else:
                blended_weights = ml_weights
        else:
            blended_weights = ml_weights

        if i > split_idx + 26:
            blended_weights = apply_volatility_targeting(
                blended_weights,
                weekly_returns.iloc[i-26:i],
                target_vol=0.12,
                lookback=26
            )

        if len(portfolio_returns) > 0:
            recent_portfolio_returns = pd.Series(
                portfolio_returns[-52:] if len(portfolio_returns) >= 52 else portfolio_returns
            )
            dd_factor = calculate_maximum_drawdown_constraint(
                recent_portfolio_returns, max_allowed_dd=0.20
            )

            if dd_factor < 1.0:
                print(f"[PART 3B] Drawdown constraint active: exposure reduced to {dd_factor:.1%}")
                blended_weights = {k: v * dd_factor for k, v in blended_weights.items()}

        week_return = 0

        if next_date in weekly_returns.index:
            for etf, weight in blended_weights.items():
                if weight > 0.01:
                    return_next_week = weekly_returns.loc[next_date, etf]
                    week_return += weight * return_next_week

        portfolio_returns.append(week_return)
        dates.append(current_date)
        weights_history.append(blended_weights.copy())

    strategy_returns = pd.Series(portfolio_returns, index=dates)

    var_95, cvar_95 = calculate_var_cvar(strategy_returns, confidence=0.95)

    print(f"\n[PART 3B] ML strategy with risk management summary:")
    print(f"  Periods: {len(strategy_returns)}")
    print(f"  Mean weekly return: {strategy_returns.mean():.4f}")
    print(f"  VaR 95%: {var_95:.4f}")
    print(f"  CVaR 95%: {cvar_95:.4f}")

    return strategy_returns, weights_history


def calculate_benchmarks(weekly_returns, test_dates):
    """Computes benchmark returns: Equal-weight and buy-and-hold SPY."""
    equal_returns = []
    spy_returns = []
    benchmark_dates = []

    for current_date in test_dates:
        if current_date in weekly_returns.index:
            week_return = 0
            for etf in weekly_returns.columns:
                week_return += 0.2 * weekly_returns.loc[current_date, etf]
            equal_returns.append(week_return)

            spy_ret = weekly_returns.loc[current_date, 'SPY']
            spy_returns.append(spy_ret)

            benchmark_dates.append(current_date)

    return (pd.Series(equal_returns, index=benchmark_dates),
            pd.Series(spy_returns, index=benchmark_dates))