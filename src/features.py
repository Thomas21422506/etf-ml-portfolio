"""
Feature engineering functions
"""

import pandas as pd
import numpy as np


def create_temporally_correct_features(returns_df):
    """
    Temporal integrity: Target = NEXT week (t+1), not current week (t).

    Logic:
    - At the beginning of week t, we only have data up to week t-1
    - We predict the direction of week t+1
    - Features: built from returns up to t-1
    - Target: return sign of week t+1
    """
    features_list = []
    targets_list = []

    for etf in returns_df.columns:
        returns = returns_df[etf]

        # Target = direction of next week (t+1)
        target = (returns.shift(-1) > 0).astype(int)
        target.name = f'{etf}_target_next_week'

        etf_features = pd.DataFrame(index=returns.index)

        # Lagged returns
        etf_features[f'{etf}_lag_1w'] = returns.shift(1)
        etf_features[f'{etf}_lag_2w'] = returns.shift(2)
        etf_features[f'{etf}_lag_3w'] = returns.shift(3)
        etf_features[f'{etf}_lag_4w'] = returns.shift(4)

        # 4-week momentum
        etf_features[f'{etf}_mom_4w'] = returns.shift(1).rolling(4).mean()

        # 8-week volatility
        etf_features[f'{etf}_vol_8w'] = returns.shift(1).rolling(8).std()

        # 12-week long-term momentum
        etf_features[f'{etf}_mom_12w'] = returns.shift(1).rolling(12).mean()

        features_list.append(etf_features)
        targets_list.append(target)

    all_features = pd.concat(features_list, axis=1)
    all_targets = pd.concat(targets_list, axis=1)

    # Remove NaN rows
    valid_idx = all_features.dropna().index.intersection(all_targets.dropna().index)
    all_features = all_features.loc[valid_idx]
    all_targets = all_targets.loc[valid_idx]

    print(f"[PART 1] Temporally consistent features created: {all_features.shape}")
    print(f"[PART 1] Date range: {all_features.index.min()} to {all_features.index.max()}")

    return all_features, all_targets


def create_macro_features(returns_df):
    """
    Creates advanced macro features to capture market regime and sentiment.
    """
    macro_features = pd.DataFrame(index=returns_df.index)
    
    # 1. VIX proxy: volatility of GLD + TLT
    if 'GLD' in returns_df.columns and 'TLT' in returns_df.columns:
        safe_haven_returns = (returns_df['GLD'] + returns_df['TLT']) / 2
        vix_proxy = safe_haven_returns.rolling(4).std()
        macro_features['vix_proxy_4w'] = vix_proxy.shift(1)
        
        vix_change = vix_proxy.diff()
        macro_features['vix_proxy_change'] = vix_change.shift(1)
    
    # 2. Risk-on / Risk-off: SPY vs TLT
    if 'SPY' in returns_df.columns and 'TLT' in returns_df.columns:
        spy_mom = returns_df['SPY'].rolling(4).sum()
        tlt_mom = returns_df['TLT'].rolling(4).sum()
        
        risk_on_off = spy_mom - tlt_mom
        macro_features['risk_on_off_4w'] = risk_on_off.shift(1)
        
        risk_on_off_trend = risk_on_off.diff()
        macro_features['risk_on_off_trend'] = risk_on_off_trend.shift(1)
    
    # 3. Flight to safety: GLD vs SPY
    if 'GLD' in returns_df.columns and 'SPY' in returns_df.columns:
        gld_mom = returns_df['GLD'].rolling(4).sum()
        spy_mom = returns_df['SPY'].rolling(4).sum()
        
        flight_to_safety = gld_mom - spy_mom
        macro_features['flight_to_safety_4w'] = flight_to_safety.shift(1)
    
    # 4. Correlation regimes
    if 'SPY' in returns_df.columns and 'QQQ' in returns_df.columns:
        corr_spy_qqq = returns_df['SPY'].rolling(12).corr(returns_df['QQQ'])
        macro_features['corr_spy_qqq_12w'] = corr_spy_qqq.shift(1)
    
    if 'SPY' in returns_df.columns and 'EEM' in returns_df.columns:
        corr_spy_eem = returns_df['SPY'].rolling(12).corr(returns_df['EEM'])
        macro_features['corr_spy_eem_12w'] = corr_spy_eem.shift(1)
    
    # 5. Market breadth
    positive_returns = (returns_df > 0).astype(int)
    market_breadth = positive_returns.sum(axis=1) / len(returns_df.columns)
    macro_features['market_breadth'] = market_breadth.shift(1)
    
    # 6. Cross-sectional dispersion
    returns_dispersion = returns_df.std(axis=1)
    macro_features['returns_dispersion'] = returns_dispersion.shift(1)
    
    print(f"[PART 1] Macro features created: {macro_features.shape}")
    print(f"[PART 1] Macro feature set: {list(macro_features.columns)}")
    
    return macro_features


def combine_all_features(returns_df):
    """Combines technical and macro features into a single dataset."""
    technical_features, targets = create_temporally_correct_features(returns_df)
    macro_features = create_macro_features(returns_df)
    
    common_idx = technical_features.index.intersection(macro_features.index)
    technical_features = technical_features.loc[common_idx]
    macro_features = macro_features.loc[common_idx]
    targets = targets.loc[common_idx]
    
    all_features = pd.concat([technical_features, macro_features], axis=1)
    
    valid_idx = all_features.dropna().index
    all_features = all_features.loc[valid_idx]
    targets = targets.loc[valid_idx]
    
    print(f"\n[PART 1] FINAL FEATURE SET (Technical + Macro): {all_features.shape}")
    print(f"[PART 1] - Technical features: {technical_features.shape[1]}")
    print(f"[PART 1] - Macro features: {macro_features.shape[1]}")
    print(f"[PART 1] - Total features: {all_features.shape[1]}")
    
    return all_features, targets


def verify_temporal_integrity(features, targets, weekly_returns):
    """
    Numerical check of temporal integrity (no look-ahead).
    """
    print("\n[CHECK] TEMPORAL INTEGRITY VERIFICATION")
    print("=" * 50)

    lagged_features = 0
    total_features = len(features.columns)

    for col in features.columns:
        if any(term in col for term in ['lag_', 'mom_', 'vol_']):
            lagged_features += 1

    print(f"[CHECK] Features identified as lagged/derived: {lagged_features}/{total_features}")

    # Numerical lag check
    print("\n[CHECK] Numerical lag consistency:")
    for etf in ['SPY', 'QQQ', 'EEM', 'TLT', 'GLD']:
        if f'{etf}_lag_1w' in features.columns and etf in weekly_returns.columns:
            feature_lag1 = features[f'{etf}_lag_1w']
            actual_lag1 = weekly_returns[etf].shift(1)
            
            common_idx = feature_lag1.dropna().index.intersection(actual_lag1.dropna().index)
            if len(common_idx) > 0:
                corr = feature_lag1.loc[common_idx].corr(actual_lag1.loc[common_idx])
                
                if corr < 0.99:
                    print(f"[CHECK] {etf}_lag_1w: correlation = {corr:.3f} (should be ~1.0)")
                else:
                    print(f"[CHECK] {etf}_lag_1w: correlation = {corr:.3f} (OK)")

    common_dates = features.index.intersection(targets.index).intersection(weekly_returns.index)
    print(f"\n[CHECK] Common dates across features/targets/returns: {len(common_dates)} periods")
    print(f"[CHECK] Date range: {common_dates.min()} to {common_dates.max()}")

    return True