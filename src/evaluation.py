"""
Performance evaluation, walk-forward analysis, and statistical tests
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, mutual_info_score
from sklearn.model_selection import TimeSeriesSplit
from scipy import stats
import logging

from .utils import TRAIN_TEST_SPLIT, PROJECT_PATH
from .backtest import realistic_ml_strategy, enhanced_ml_strategy_with_risk_management, calculate_benchmarks
from .benchmarks import momentum_benchmark, calculate_markowitz_benchmarks

logger = logging.getLogger(__name__)


# ============================================================================
# PART 5: COMPLETE BACKTESTING
# ============================================================================

def load_all_returns():
    """Loads all strategy return series from disk."""
    strategies = {}

    features = pd.read_csv(f'{PROJECT_PATH}etf_features.csv', index_col=0, parse_dates=True)
    targets = pd.read_csv(f'{PROJECT_PATH}etf_targets.csv', index_col=0, parse_dates=True)
    weekly_returns = pd.read_csv(f'{PROJECT_PATH}etf_weekly_returns.csv', index_col=0, parse_dates=True)

    test_dates = features.index[int(len(features) * TRAIN_TEST_SPLIT):]

    strategies['ML_Strategy'] = realistic_ml_strategy(features, targets, weekly_returns)
    
    ml_returns_rm, _ = enhanced_ml_strategy_with_risk_management(features, targets, weekly_returns)
    strategies['ML_Strategy_RiskMgmt'] = ml_returns_rm

    equal_returns, spy_returns = calculate_benchmarks(weekly_returns, test_dates)
    strategies['Equal_Weight'] = equal_returns
    strategies['Buy_Hold_SPY'] = spy_returns

    strategies['Momentum_4W'] = momentum_benchmark(weekly_returns)

    markowitz_returns = calculate_markowitz_benchmarks(weekly_returns, test_dates)
    strategies.update(markowitz_returns)

    return strategies


def comprehensive_performance_analysis(all_returns):
    """Computes a full performance summary for all strategies."""
    performance_data = []

    for strategy_name, returns in all_returns.items():
        cumulative = (1 + returns).cumprod()
        total_return = cumulative.iloc[-1] - 1
        annual_return = (1 + total_return) ** (52/len(returns)) - 1
        volatility = returns.std() * np.sqrt(52)
        sharpe = annual_return / volatility if volatility > 0 else 0
        max_drawdown = (cumulative / cumulative.expanding().max() - 1).min()
        win_rate = (returns > 0).mean()

        performance_data.append({
            'Strategy': strategy_name,
            'Total_Return': total_return,
            'Annual_Return': annual_return,
            'Volatility': volatility,
            'Sharpe_Ratio': sharpe,
            'Max_Drawdown': max_drawdown,
            'Win_Rate': win_rate
        })

    performance_df = pd.DataFrame(performance_data)
    performance_df = performance_df.sort_values('Sharpe_Ratio', ascending=False)
    return performance_df


def generate_comprehensive_visualizations(all_returns, performance_df):
    """Generates the main set of performance charts."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    for strategy, returns in all_returns.items():
        (1 + returns).cumprod().plot(ax=axes[0, 0], label=strategy, linewidth=2)
    axes[0, 0].set_title('Cumulative Performance', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    performance_df.set_index('Strategy')['Sharpe_Ratio'].plot(
        kind='barh', ax=axes[0, 1], alpha=0.7, color='steelblue'
    )
    axes[0, 1].set_title('Sharpe Ratio', fontsize=14, fontweight='bold')
    axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=1)

    for _, row in performance_df.iterrows():
        axes[1, 0].scatter(row['Volatility'], row['Annual_Return'], s=100,
                           label=row['Strategy'], alpha=0.7)
    axes[1, 0].set_xlabel('Annualized Volatility', fontsize=12)
    axes[1, 0].set_ylabel('Annualized Return', fontsize=12)
    axes[1, 0].set_title('Risk-Return Profile', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    performance_df.set_index('Strategy')['Win_Rate'].plot(
        kind='barh', ax=axes[1, 1], color='green', alpha=0.7
    )
    axes[1, 1].set_title('Win Rate', fontsize=14, fontweight='bold')
    axes[1, 1].axvline(x=0.5, color='red', linestyle='--', linewidth=1, label='50%')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig('outputs/figures/comprehensive_backtesting_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def generate_portfolio_evolution_chart(all_returns, performance_df):
    """Generates a detailed chart of ML portfolio evolution with comparison to benchmarks."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0:2, :])

    for strategy, returns in all_returns.items():
        cumulative = (1 + returns).cumprod() * 100

        if strategy == 'ML_Strategy':
            ax1.plot(cumulative.index, cumulative.values,
                     label=strategy, linewidth=3, color='darkblue', alpha=0.9)
        else:
            ax1.plot(cumulative.index, cumulative.values,
                     label=strategy, linewidth=1.5, alpha=0.6, linestyle='--')

    ax1.set_title('Portfolio Evolution - Value of 100 Units Invested',
                  fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Portfolio Value', fontsize=12)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=100, color='red', linestyle=':', linewidth=1, alpha=0.5, label='Initial capital')

    if 'ML_Strategy' in all_returns:
        ml_returns = all_returns['ML_Strategy']
        final_value = (1 + ml_returns).cumprod().iloc[-1] * 100
        total_return = (final_value - 100) / 100

        textstr = 'ML Strategy:\n'
        textstr += f'Final value: {final_value:.2f}\n'
        textstr += f'Total return: {total_return:.1%}\n'
        textstr += f'Period: {len(ml_returns)} weeks'

        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
        ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=11,
                 verticalalignment='top', bbox=props)

    ax2 = fig.add_subplot(gs[2, 0])
    if 'ML_Strategy' in all_returns:
        ml_returns = all_returns['ML_Strategy']
        cumulative = (1 + ml_returns).cumprod()
        drawdown = (cumulative / cumulative.expanding().max() - 1) * 100

        ax2.fill_between(drawdown.index, drawdown.values, 0,
                         color='red', alpha=0.3)
        ax2.plot(drawdown.index, drawdown.values,
                 color='darkred', linewidth=1.5)
        ax2.set_title('ML Strategy Drawdown', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=10)
        ax2.set_ylabel('Drawdown (%)', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linewidth=1)

        max_dd = drawdown.min()
        ax2.text(0.02, 0.02, f'Max drawdown: {max_dd:.1f}%',
                 transform=ax2.transAxes, fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax3 = fig.add_subplot(gs[2, 1])
    if 'ML_Strategy' in all_returns:
        ml_returns = all_returns['ML_Strategy']
        returns_pct = ml_returns * 100

        ax3.hist(returns_pct, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax3.axvline(x=returns_pct.mean(), color='green', linestyle='--',
                    linewidth=2, label=f'Mean: {returns_pct.mean():.2f}%')
        ax3.set_title('Distribution of Weekly Returns',
                      fontsize=12, fontweight='bold')
        ax3.set_xlabel('Weekly return (%)', fontsize=10)
        ax3.set_ylabel('Frequency', fontsize=10)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')

        positive_weeks = (returns_pct > 0).sum()
        total_weeks = len(returns_pct)
        win_rate = positive_weeks / total_weeks * 100

        textstr = f'Positive weeks: {positive_weeks}/{total_weeks}\n'
        textstr += f'Win rate: {win_rate:.1f}%\n'
        textstr += f'Median: {returns_pct.median():.2f}%'

        ax3.text(0.98, 0.98, textstr, transform=ax3.transAxes, fontsize=9,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.savefig('outputs/figures/portfolio_evolution_detailed.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("[PART 5] Detailed portfolio chart saved to: portfolio_evolution_detailed.png")


# ============================================================================
# PART 5B: WALK-FORWARD ANALYSIS
# ============================================================================

def walk_forward_optimization(features, targets, weekly_returns, best_model_names,
                              train_window=104, test_window=26, step_size=26):
    """Walk-forward analysis to test robustness across multiple rolling periods."""
    print("\n" + "="*80)
    print("[PART 5B] WALK-FORWARD ANALYSIS")
    print("="*80)
    print("Configuration:")
    print(f"  • Train window: {train_window} weeks ({train_window/52:.1f} years)")
    print(f"  • Test window: {test_window} weeks ({test_window/52:.1f} years)")
    print(f"  • Step size: {step_size} weeks")
    print(f"  • Using best models: {best_model_names}\n")

    etf_list = ['SPY', 'QQQ', 'EEM', 'TLT', 'GLD']
    all_results = []

    common_idx = features.index.intersection(weekly_returns.index).intersection(targets.index)
    features = features.loc[common_idx]
    weekly_returns = weekly_returns.loc[common_idx]
    targets = targets.loc[common_idx]

    total_periods = len(features)
    n_windows = (total_periods - train_window) // step_size

    print(f"Total windows in walk-forward analysis: {n_windows}\n")

    for window_idx in range(n_windows):
        train_start = window_idx * step_size
        train_end = train_start + train_window
        test_start = train_end
        test_end = min(test_start + test_window, total_periods)

        if test_end - test_start < 4:
            break

        train_dates = features.index[train_start:train_end]
        test_dates = features.index[test_start:test_end]

        print(f"Window {window_idx + 1}/{n_windows}:")
        print(f"  Train: {train_dates[0].strftime('%Y-%m-%d')} → {train_dates[-1].strftime('%Y-%m-%d')}")
        print(f"  Test:  {test_dates[0].strftime('%Y-%m-%d')} → {test_dates[-1].strftime('%Y-%m-%d')}")

        window_models = {}

        for etf in etf_list:
            target_col = f'{etf}_target_next_week'
            X_train = features.iloc[train_start:train_end]
            y_train = targets[target_col].iloc[train_start:train_end]

            if len(y_train) < 26:
                continue

            n_up = y_train.sum()
            n_down = len(y_train) - n_up

            if n_up > n_down:
                class_weights = {1: 1.0, 0: n_up/n_down}
            else:
                class_weights = {1: n_down/n_up, 0: 1.0}

            model_name = best_model_names[etf]

            if model_name == 'Random Forest':
                model = RandomForestClassifier(
                    n_estimators=50,
                    random_state=42,
                    max_depth=5,
                    class_weight=class_weights
                )
            elif model_name == 'XGBoost':
                scale_pos_weight = n_down/n_up if n_up > 0 else 1.0
                model = XGBClassifier(
                    random_state=42,
                    n_estimators=50,
                    max_depth=3,
                    learning_rate=0.1,
                    scale_pos_weight=scale_pos_weight,
                    eval_metric='logloss'
                )
            else:
                model = LogisticRegression(
                    random_state=42,
                    max_iter=1000,
                    class_weight=class_weights,
                    C=0.1,
                    penalty='l2'
                )

            model.fit(X_train, y_train)
            window_models[etf] = model

        window_returns = []

        for i in range(test_start, test_end - 1):
            current_date = features.index[i]
            next_date = features.index[i + 1]

            weekly_weights = {}
            all_predictions = []

            for etf in etf_list:
                if etf not in window_models:
                    weekly_weights[etf] = 0
                    continue

                model = window_models[etf]
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
            else:
                if all_predictions:
                    filtered_probs = [(etf, prob) for etf, prob in all_predictions if prob >= 0.35]

                    if len(filtered_probs) >= 2:
                        top_etfs = sorted(filtered_probs, key=lambda x: x[1], reverse=True)[:3]
                        n_selected = len(top_etfs)
                        weekly_weights = {etf: 1.0/n_selected for etf, _ in top_etfs}
                    else:
                        weekly_weights = {etf: 0 for etf in etf_list}
                else:
                    weekly_weights = {etf: 0 for etf in etf_list}

            week_return = 0
            if next_date in weekly_returns.index:
                for etf, weight in weekly_weights.items():
                    if weight > 0.01:
                        week_return += weight * weekly_returns.loc[next_date, etf]

            window_returns.append(week_return)

        window_returns_series = pd.Series(window_returns)

        if len(window_returns_series) > 0:
            total_return = (1 + window_returns_series).prod() - 1
            annual_return = (1 + total_return) ** (52/len(window_returns_series)) - 1
            volatility = window_returns_series.std() * np.sqrt(52)
            sharpe = annual_return / volatility if volatility > 0 else 0

            cumulative = (1 + window_returns_series).cumprod()
            max_dd = (cumulative / cumulative.expanding().max() - 1).min()

            win_rate = (window_returns_series > 0).mean()

            all_results.append({
                'window': window_idx + 1,
                'train_start': train_dates[0],
                'train_end': train_dates[-1],
                'test_start': test_dates[0],
                'test_end': test_dates[-1],
                'n_test_weeks': len(window_returns_series),
                'total_return': total_return,
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd,
                'win_rate': win_rate
            })

            print(f"  → Total return: {total_return:+.2%} | Sharpe: {sharpe:.2f} | Max DD: {max_dd:.2%}")

        print()

    results_df = pd.DataFrame(all_results)

    print("\n" + "="*80)
    print("[PART 5B] AGGREGATED WALK-FORWARD STATISTICS")
    print("="*80)
    print(f"Number of windows: {len(results_df)}")
    print(f"\nAnnualized return:")
    print(f"  Mean:     {results_df['annual_return'].mean():+.2%}")
    print(f"  Median:   {results_df['annual_return'].median():+.2%}")
    print(f"  Std dev:  {results_df['annual_return'].std():.2%}")
    print(f"  Min:      {results_df['annual_return'].min():+.2%}")
    print(f"  Max:      {results_df['annual_return'].max():+.2%}")

    print(f"\nSharpe ratio:")
    print(f"  Mean:     {results_df['sharpe_ratio'].mean():.2f}")
    print(f"  Median:   {results_df['sharpe_ratio'].median():.2f}")
    print(f"  Min:      {results_df['sharpe_ratio'].min():.2f}")
    print(f"  Max:      {results_df['sharpe_ratio'].max():.2f}")

    print(f"\nMaximum drawdown:")
    print(f"  Mean:     {results_df['max_drawdown'].mean():.2%}")
    print(f"  Worst:    {results_df['max_drawdown'].min():.2%}")

    print(f"\nWin rate:")
    print(f"  Mean:     {results_df['win_rate'].mean():.1%}")
    print(f"  Median:   {results_df['win_rate'].median():.1%}")

    positive_windows = (results_df['total_return'] > 0).sum()
    consistency = positive_windows / len(results_df)
    print(f"\nConsistency (percentage of positive windows): "
          f"{consistency:.1%} ({positive_windows}/{len(results_df)})")

    return results_df


def visualize_walk_forward_results(wf_results_df):
    """Visualizes walk-forward results."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    ax1 = axes[0, 0]
    wf_results_df['total_return'].plot(kind='bar', ax=ax1, color='steelblue', alpha=0.7)
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax1.set_title('Total Return per Walk-Forward Window', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Window #', fontsize=11)
    ax1.set_ylabel('Total Return', fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')

    ax2 = axes[0, 1]
    wf_results_df['sharpe_ratio'].plot(kind='bar', ax=ax2, color='green', alpha=0.7)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax2.axhline(y=wf_results_df['sharpe_ratio'].mean(), color='orange',
                linestyle='--', linewidth=2,
                label=f"Mean: {wf_results_df['sharpe_ratio'].mean():.2f}")
    ax2.set_title('Sharpe Ratio per Window', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Window #', fontsize=11)
    ax2.set_ylabel('Sharpe Ratio', fontsize=11)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    ax3 = axes[1, 0]
    wf_results_df['max_drawdown'].plot(kind='bar', ax=ax3, color='red', alpha=0.7)
    ax3.set_title('Maximum Drawdown per Window', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Window #', fontsize=11)
    ax3.set_ylabel('Max Drawdown', fontsize=11)
    ax3.grid(True, alpha=0.3, axis='y')

    ax4 = axes[1, 1]
    wf_results_df['annual_return'].hist(
        bins=15, ax=ax4, color='purple', alpha=0.7, edgecolor='black'
    )
    ax4.axvline(
        x=wf_results_df['annual_return'].mean(), color='orange',
        linestyle='--', linewidth=2,
        label=f"Mean: {wf_results_df['annual_return'].mean():.2%}"
    )
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=1)
    ax4.set_title('Distribution of Annualized Returns', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Annualized Return', fontsize=11)
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('outputs/figures/walk_forward_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("[PART 5B] Walk-forward charts saved to: walk_forward_analysis.png")


def analyze_regime_performance(wf_results_df, weekly_returns):
    """Analyzes performance depending on market regime."""
    if 'SPY' not in weekly_returns.columns:
        print("[PART 5B] SPY not available for regime analysis.")
        return

    print("\n" + "="*80)
    print("[PART 5B] MARKET REGIME ANALYSIS")
    print("="*80)

    regimes = []

    for _, row in wf_results_df.iterrows():
        test_start = row['test_start']
        test_end = row['test_end']

        spy_returns = weekly_returns.loc[test_start:test_end, 'SPY']
        spy_momentum = spy_returns.mean()

        if spy_momentum > 0.001:
            regime = 'Bull'
        elif spy_momentum < -0.001:
            regime = 'Bear'
        else:
            regime = 'Sideways'

        regimes.append(regime)

    wf_results_df['regime'] = regimes

    for regime in ['Bull', 'Bear', 'Sideways']:
        regime_data = wf_results_df[wf_results_df['regime'] == regime]

        if len(regime_data) == 0:
            continue

        print(f"\n{regime.upper()} regime ({len(regime_data)} windows):")
        print(f"  Mean annualized return: {regime_data['annual_return'].mean():+.2%}")
        print(f"  Mean Sharpe:            {regime_data['sharpe_ratio'].mean():.2f}")
        print(f"  Mean max drawdown:      {regime_data['max_drawdown'].mean():.2%}")
        print(f"  Mean win rate:          {regime_data['win_rate'].mean():.1%}")

    return wf_results_df


# ============================================================================
# PART 6: STATISTICAL QUALITY ANALYSIS
# ============================================================================

def statistical_significance_analysis(features, targets, best_model_names):
    """Binomial tests to assess statistical significance of predictive accuracy."""
    print("\n[PART 6] STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 60)

    for etf in ['SPY', 'QQQ', 'EEM', 'TLT', 'GLD']:
        target_col = f'{etf}_target_next_week'
        y = targets[target_col].values
        X = features.values

        tscv = TimeSeriesSplit(n_splits=3)
        accuracies = []
        p_values = []

        model_name = best_model_names[etf]

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if len(np.unique(y_train)) < 2:
                continue

            n_up = y_train.sum()
            n_down = len(y_train) - n_up

            if n_up > n_down:
                class_weights = {1: 1.0, 0: n_up/n_down}
            else:
                class_weights = {1: n_down/n_up, 0: 1.0}

            if model_name == 'Random Forest':
                model = RandomForestClassifier(
                    n_estimators=50,
                    random_state=42,
                    max_depth=5,
                    class_weight=class_weights
                )
            elif model_name == 'XGBoost':
                scale_pos_weight = n_down/n_up if n_up > 0 else 1.0
                model = XGBClassifier(
                    random_state=42,
                    n_estimators=50,
                    max_depth=3,
                    learning_rate=0.1,
                    scale_pos_weight=scale_pos_weight,
                    eval_metric='logloss'
                )
            else:
                model = LogisticRegression(
                    random_state=42,
                    max_iter=1000,
                    class_weight=class_weights
                )

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)

            n_correct = np.sum(y_pred == y_test)
            n_total = len(y_test)

            binom_result = stats.binomtest(n_correct, n_total, 0.5, alternative='greater')
            p_value = binom_result.pvalue
            p_values.append(p_value)

        if accuracies:
            mean_accuracy = np.mean(accuracies)
            mean_p_value = np.mean(p_values)

            significant = "SIGNIFICANT" if mean_p_value < 0.05 else "NOT SIGNIFICANT"

            print(f"{etf} ({model_name}): Accuracy={mean_accuracy:.3f}, p-value={mean_p_value:.4f} ({significant})")


def information_content_analysis(features, targets):
    """Mutual information analysis to assess feature information content."""
    print("\n[PART 6] INFORMATION CONTENT ANALYSIS")
    print("=" * 60)

    for etf in ['SPY', 'QQQ', 'EEM', 'TLT', 'GLD']:
        target_col = f'{etf}_target_next_week'
        y = targets[target_col]

        etf_features = [col for col in features.columns if etf in col]

        if not etf_features:
            continue

        mi_scores = []
        for feature in etf_features[:5]:
            x = features[feature].dropna()
            common_idx = x.index.intersection(y.index)
            if len(common_idx) > 50:
                x_common = x.loc[common_idx]
                y_common = y.loc[common_idx]

                x_binned = pd.cut(x_common, bins=5, labels=False, duplicates='drop')
                mi = mutual_info_score(x_binned, y_common)
                mi_scores.append((feature, mi))

        if mi_scores:
            top_3 = sorted(mi_scores, key=lambda x: x[1], reverse=True)[:3]
            print(f"\n{etf} - Top 3 features by mutual information:")
            for feature, mi in top_3:
                feature_short = feature.replace(f'{etf}_', '')
                print(f"  • {feature_short}: {mi:.4f}")


def generate_prediction_quality_plots(features, targets, best_model_names):
    """Generates plots summarizing prediction quality and class balance."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    etf_accuracies = []
    etf_names = ['SPY', 'QQQ', 'EEM', 'TLT', 'GLD']

    for etf in etf_names:
        target_col = f'{etf}_target_next_week'
        y = targets[target_col].values
        X = features.values

        split_idx = int(len(X) * TRAIN_TEST_SPLIT)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        if len(np.unique(y_train)) < 2:
            accuracy = 0.5
        else:
            n_up = y_train.sum()
            n_down = len(y_train) - n_up

            if n_up > n_down:
                class_weights = {1: 1.0, 0: n_up/n_down}
            else:
                class_weights = {1: n_down/n_up, 0: 1.0}

            model_name = best_model_names[etf]

            if model_name == 'Random Forest':
                model = RandomForestClassifier(
                    n_estimators=50,
                    random_state=42,
                    max_depth=5,
                    class_weight=class_weights
                )
            elif model_name == 'XGBoost':
                scale_pos_weight = n_down/n_up if n_up > 0 else 1.0
                model = XGBClassifier(
                    random_state=42,
                    n_estimators=50,
                    max_depth=3,
                    learning_rate=0.1,
                    scale_pos_weight=scale_pos_weight,
                    eval_metric='logloss'
                )
            else:
                model = LogisticRegression(
                    random_state=42,
                    max_iter=1000,
                    class_weight=class_weights
                )

            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)

        etf_accuracies.append(accuracy)

    axes[0].bar(etf_names, etf_accuracies, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].axhline(y=0.5, color='red', linestyle='--', label='Random (50%)', linewidth=2)
    axes[0].set_title('Prediction Accuracy by ETF (Using Best Models)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_ylim(0.4, 0.7)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    class_balance = []
    for etf in etf_names:
        target_col = f'{etf}_target_next_week'
        up_proportion = targets[target_col].mean()
        class_balance.append(up_proportion)

    axes[1].bar(etf_names, class_balance, color='green', alpha=0.7, edgecolor='black')
    axes[1].axhline(y=0.5, color='red', linestyle='--', label='Perfect balance (50%)', linewidth=2)
    axes[1].set_title('Proportion of Positive Weeks by ETF', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Proportion', fontsize=12)
    axes[1].set_ylim(0.4, 0.7)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('outputs/figures/prediction_quality_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n[PART 6] Prediction quality plots saved to: prediction_quality_analysis.png")