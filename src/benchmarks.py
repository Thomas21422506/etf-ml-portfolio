"""
Benchmark strategies: Momentum and Markowitz optimization
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize

from .utils import TRAIN_TEST_SPLIT


# ============================================================================
# PART 4: MARKOWITZ AND MOMENTUM BENCHMARKS
# ============================================================================

def momentum_benchmark(weekly_returns, lookback=4, hold_top=3):
    """
    Momentum benchmark without look-ahead bias.
    """
    split_idx = int(len(weekly_returns) * TRAIN_TEST_SPLIT)
    test_dates = weekly_returns.index[split_idx:]
    momentum_returns = []
    momentum_dates = []

    print(f"[PART 4] Momentum benchmark over {len(test_dates)} weeks")

    for current_date in test_dates:
        current_idx = weekly_returns.index.get_loc(current_date)

        if current_idx < lookback:
            momentum_returns.append(0)
            momentum_dates.append(current_date)
            continue

        lookback_start = current_idx - lookback
        lookback_end = current_idx - 1

        momentum_scores = {}
        for etf in weekly_returns.columns:
            past_returns = weekly_returns[etf].iloc[lookback_start:lookback_end]

            if len(past_returns) > 0:
                momentum_scores[etf] = (1 + past_returns).prod() - 1

        if momentum_scores:
            top_etfs = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)[:hold_top]
            weights = {etf: 1/hold_top for etf, _ in top_etfs}

            week_return = sum(weights.get(etf, 0) * weekly_returns.loc[current_date, etf]
                              for etf in weekly_returns.columns)
            momentum_returns.append(week_return)
            momentum_dates.append(current_date)
        else:
            momentum_returns.append(0)
            momentum_dates.append(current_date)

    momentum_series = pd.Series(momentum_returns, index=momentum_dates)
    print(f"[PART 4] Momentum benchmark: {len(momentum_series)} periods, "
          f"mean return: {momentum_series.mean():.4f}")

    return momentum_series


def markowitz_optimization(returns, optimization_type='sharpe', risk_free_rate=0.02):
    """Markowitz mean-variance optimization."""
    expected_returns = returns.mean() * 52
    covariance_matrix = returns.cov() * 52
    n_assets = len(expected_returns)

    def portfolio_return(weights):
        return np.dot(weights, expected_returns)

    def portfolio_variance(weights):
        return np.dot(weights.T, np.dot(covariance_matrix, weights))

    def portfolio_sharpe(weights):
        ret = portfolio_return(weights)
        vol = np.sqrt(portfolio_variance(weights))
        return (ret - risk_free_rate) / vol if vol > 0 else 0

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))
    initial_weights = np.array([1/n_assets] * n_assets)

    if optimization_type == 'sharpe':
        objective = lambda w: -portfolio_sharpe(w)
    elif optimization_type == 'min_variance':
        objective = portfolio_variance

    try:
        result = minimize(
            objective, initial_weights,
            method='SLSQP', bounds=bounds, constraints=constraints,
            options={'ftol': 1e-9, 'disp': False}
        )

        if result.success:
            weights = result.x
            weights[weights < 0.01] = 0
            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                weights = np.array([1/n_assets] * n_assets)
            return {etf: weight for etf, weight in zip(returns.columns, weights) if weight > 0.001}
    except Exception as e:
        print(f"[PART 4] Markowitz optimization error: {e}")

    return {etf: 1/n_assets for etf in returns.columns}


def calculate_markowitz_benchmarks(weekly_returns, test_dates, lookback=104):
    """Computes Markowitz-based benchmark strategies WITH ROLLING OPTIMIZATION."""
    test_dates = [d for d in test_dates if d in weekly_returns.index]

    if not test_dates:
        print("[PART 4] No valid test dates for Markowitz benchmarks.")
        return {}

    print(f"[PART 4] Computing ROLLING Markowitz benchmarks (lookback={lookback} weeks)")

    markowitz_returns = {}

    strategies = [
        ('Markowitz_Sharpe', 'sharpe'),
        ('Markowitz_MinVariance', 'min_variance')
    ]

    for strategy_name, optimization_type in strategies:
        strategy_returns = []
        strategy_dates = []

        print(f"\n[PART 4] Computing {strategy_name}...")

        for i, current_date in enumerate(test_dates):
            try:
                current_idx = weekly_returns.index.get_loc(current_date)
            except:
                continue

            train_start_idx = max(0, current_idx - lookback)
            train_end_idx = current_idx

            if train_end_idx - train_start_idx < 52:
                strategy_returns.append(0)
                strategy_dates.append(current_date)
                continue

            train_returns = weekly_returns.iloc[train_start_idx:train_end_idx]

            weights = markowitz_optimization(train_returns, optimization_type)

            week_return = 0
            for etf, weight in weights.items():
                if etf in weekly_returns.columns:
                    week_return += weight * weekly_returns.loc[current_date, etf]

            strategy_returns.append(week_return)
            strategy_dates.append(current_date)

            if i % 20 == 0 and i > 0:
                print(f"  Processed {i}/{len(test_dates)} dates...")

        if strategy_returns:
            markowitz_returns[strategy_name] = pd.Series(strategy_returns, index=strategy_dates)

            total_return = (1 + markowitz_returns[strategy_name]).prod() - 1
            annual_return = (1 + total_return) ** (52/len(strategy_returns)) - 1
            volatility = markowitz_returns[strategy_name].std() * np.sqrt(52)
            sharpe = annual_return / volatility if volatility > 0 else 0

            print(f"[PART 4] {strategy_name}: {len(strategy_returns)} periods, "
                  f"mean return: {markowitz_returns[strategy_name].mean():.4f}")
            print(f"          Annualized return: {annual_return:.2%}, "
                  f"Sharpe: {sharpe:.2f}")

    return markowitz_returns