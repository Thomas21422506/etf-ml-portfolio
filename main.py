"""
ETF ML Portfolio Allocation - Main Execution Script
"""


# Imports standards
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Imports src/
from src.data_loader import find_data_file, load_data_properly, create_weekly_data
from src.features import combine_all_features, verify_temporal_integrity
from src.models import (
    train_and_evaluate_models,
    temporal_stratified_evaluation,
    display_ml_metrics,
    visualize_ml_performance
)
from src.backtest import (
    realistic_ml_strategy,
    enhanced_ml_strategy_with_risk_management
)
from src.backtest import calculate_benchmarks
from src.benchmarks import (
    momentum_benchmark,
    calculate_markowitz_benchmarks
)
from src.evaluation import (
    comprehensive_performance_analysis,
    generate_comprehensive_visualizations,
    generate_portfolio_evolution_chart,
    walk_forward_optimization,
    visualize_walk_forward_results,
    analyze_regime_performance,
    statistical_significance_analysis,
    information_content_analysis,
    generate_prediction_quality_plots
)
from src.utils import TRAIN_TEST_SPLIT, PROJECT_PATH

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("ETF ML PORTFOLIO PROJECT - CORRECTED VERSION")
    print("=" * 70)
    print("Key corrections:")
    print("  • Target = next week (t+1)")
    print("  • Correct temporal alignment in backtesting")
    print("  • No look-ahead bias")
    print(f"  • Unified split at {TRAIN_TEST_SPLIT:.0%}")
    print("=" * 70)

    # PART 1: Feature Engineering
    print("\n[PART 1] DATA LOADING AND FEATURE ENGINEERING")
    print("=" * 60)

    DATA_PATH = find_data_file()
    if DATA_PATH is None:
        print("[CRITICAL] No data file found.")
        return

    df = load_data_properly(DATA_PATH)
    print(f"[PART 1] Raw data loaded: {df.shape}")

    weekly_returns = create_weekly_data(df)
    print(f"[PART 1] Weekly returns: {weekly_returns.shape}")

    features, targets = combine_all_features(weekly_returns)
    verify_temporal_integrity(features, targets, weekly_returns)

    # Class balance check
    print("\n[PART 1] CLASS BALANCE CHECK")
    print("-" * 60)
    for etf in ['SPY', 'QQQ', 'EEM', 'TLT', 'GLD']:
        up_proportion = targets[f'{etf}_target_next_week'].mean()
        status = "Imbalanced" if (up_proportion > 0.60 or up_proportion < 0.40) else "Reasonably balanced"
        print(f"{etf}: {up_proportion:.1%} positive weeks ({status})")

    # Save intermediate data
    features.to_csv(f'{PROJECT_PATH}etf_features.csv')
    targets.to_csv(f'{PROJECT_PATH}etf_targets.csv')
    weekly_returns.to_csv(f'{PROJECT_PATH}etf_weekly_returns.csv')
    print("\n[PART 1] Features, targets and weekly returns saved to CSV.")

    # PART 2: ML Training
    print("\n[PART 2] ML MODEL TRAINING")
    print("=" * 60)

    etf_performance = {}
    best_model_names = {}

    print("[PART 2] Phase 1: Model selection and evaluation")
    print("-" * 60)

    for etf in ['SPY', 'QQQ', 'EEM', 'TLT', 'GLD']:
        print(f"  Evaluating models for {etf}...")
        results, X_test, y_test, best_name, _ = train_and_evaluate_models(features, targets, etf)
        etf_performance[etf] = results
        best_model_names[etf] = best_name
        print(f"    → Best model: {best_name}")

    print(f"\n[PART 2] Best models selected: {best_model_names}")

    temporal_results = temporal_stratified_evaluation(features, targets, best_model_names)

    print("\n[PART 2] Phase 2: Training best models on full training data")
    print("=" * 60)

    split_idx = int(len(features) * TRAIN_TEST_SPLIT)
    final_best_models = {}

    for etf in ['SPY', 'QQQ', 'EEM', 'TLT', 'GLD']:
        best_model_name = best_model_names[etf]
        target_col = f'{etf}_target_next_week'
        X_train_full = features.iloc[:split_idx]
        y_train_full = targets[target_col].iloc[:split_idx]

        n_up = y_train_full.sum()
        n_down = len(y_train_full) - n_up

        if n_up > n_down:
            class_weights = {1: 1.0, 0: n_up/n_down}
        else:
            class_weights = {1: n_down/n_up, 0: 1.0}

        if best_model_name == 'Random Forest':
            model = RandomForestClassifier(
                n_estimators=50, random_state=42, max_depth=5, class_weight=class_weights
            )
        elif best_model_name == 'XGBoost':
            scale_pos_weight = n_down/n_up if n_up > 0 else 1.0
            model = XGBClassifier(
                random_state=42, n_estimators=50, max_depth=3,
                learning_rate=0.1, scale_pos_weight=scale_pos_weight, eval_metric='logloss'
            )
        else:
            model = LogisticRegression(
                random_state=42, max_iter=1000, C=0.1, penalty='l2', class_weight=class_weights
            )

        model.fit(X_train_full, y_train_full)
        final_best_models[etf] = model
        print(f"  {etf}: {best_model_name} trained on {len(X_train_full)} samples")

    print("\n[PART 2] Final models ready for backtesting")

    ml_metrics_df = display_ml_metrics(etf_performance)
    visualize_ml_performance(ml_metrics_df)

    # PART 3: Trading Strategy
    print("\n[PART 3] ML TRADING STRATEGY")
    print("=" * 60)

    ml_returns = realistic_ml_strategy(features, targets, weekly_returns, final_best_models)
    print(f"[PART 3] Base ML strategy computed: {len(ml_returns)} periods.")

    print("\n[PART 3B] ML STRATEGY WITH RISK MANAGEMENT")
    print("=" * 60)
    ml_returns_rm, weights_history = enhanced_ml_strategy_with_risk_management(
        features, targets, weekly_returns, final_best_models
    )
    print(f"[PART 3B] ML + risk management strategy computed: {len(ml_returns_rm)} periods.")

    test_dates = features.index[split_idx:]
    equal_returns, spy_returns = calculate_benchmarks(weekly_returns, test_dates)
    print("[PART 3] Benchmarks (Equal weight, Buy & Hold SPY) computed.")

    # PART 4: Markowitz
    print("\n[PART 4] MARKOWITZ BENCHMARKS")
    print("=" * 60)

    markowitz_returns = calculate_markowitz_benchmarks(weekly_returns, test_dates, lookback=104)
    print("[PART 4] Markowitz strategies computed.")

    # PART 5: Complete Backtest
    print("\n[PART 5] COMPLETE BACKTESTING")
    print("=" * 60)

    all_returns = {
        'ML_Strategy': ml_returns,
        'ML_Strategy_RiskMgmt': ml_returns_rm,
        'Equal_Weight': equal_returns,
        'Buy_Hold_SPY': spy_returns,
        'Momentum_4W': momentum_benchmark(weekly_returns)
    }
    all_returns.update(markowitz_returns)

    performance_df = comprehensive_performance_analysis(all_returns)

    print("\n[PART 5] FINAL STRATEGY COMPARISON:")
    print("=" * 80)
    print(performance_df.round(4).to_string())

    generate_comprehensive_visualizations(all_returns, performance_df)
    generate_portfolio_evolution_chart(all_returns, performance_df)
    performance_df.to_csv(f'{PROJECT_PATH}final_performance_results.csv', index=False)
    
    # Save all strategy returns for dashboard
    print("\n[PART 5] Saving strategy returns for dashboard...")
    
    # Find common index (shortest length)
    min_length = min(len(v) for v in all_returns.values())
    
    # Align all series on same index
    aligned_returns = {}
    for key, series in all_returns.items():
        if len(series) > min_length:
            aligned_returns[key] = series.iloc[-min_length:]
        else:
            aligned_returns[key] = series
    
    all_returns_df = pd.DataFrame(aligned_returns)
    all_returns_df.to_csv(f'{PROJECT_PATH}all_strategy_returns.csv')
    print(f"[PART 5] Strategy returns saved to: all_strategy_returns.csv ({len(all_returns_df)} periods)")

    # PART 5B: Walk-forward analysis
    print("\n[PART 5B] WALK-FORWARD ANALYSIS")
    print("=" * 60)

    wf_results = walk_forward_optimization(
        features, targets, weekly_returns, best_model_names,
        train_window=104, test_window=26, step_size=26
    )
    visualize_walk_forward_results(wf_results)
    wf_results_with_regime = analyze_regime_performance(wf_results, weekly_returns)
    wf_results_with_regime.to_csv(f'{PROJECT_PATH}walk_forward_results.csv', index=False)

    # PART 6: Statistical Analysis
    print("\n[PART 6] STATISTICAL QUALITY ANALYSIS")
    print("=" * 60)

    statistical_significance_analysis(features, targets, best_model_names)
    information_content_analysis(features, targets)
    generate_prediction_quality_plots(features, targets, best_model_names)

    print("\nPROJECT COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    main()