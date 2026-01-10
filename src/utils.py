"""
Utility functions and global configuration
"""

import os

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

# Reproducibility
RANDOM_STATE = 42

# Unified train/test split
TRAIN_TEST_SPLIT = 0.8

# Project path
PROJECT_PATH = "outputs/results/"

# ETF list
ETF_LIST = ['SPY', 'QQQ', 'EEM', 'TLT', 'GLD']

# Model hyperparameters (optional - if you want to centralize)
LOGISTIC_PARAMS = {
    'C': 0.1,
    'class_weight': 'balanced',
    'max_iter': 1000,
    'random_state': RANDOM_STATE,
    'penalty': 'l2'
}

RANDOM_FOREST_PARAMS = {
    'n_estimators': 50,
    'max_depth': 5,
    'random_state': RANDOM_STATE
}

XGBOOST_PARAMS = {
    'n_estimators': 50,
    'max_depth': 3,
    'learning_rate': 0.1,
    'random_state': RANDOM_STATE,
    'eval_metric': 'logloss'
}

# Walk-forward configuration (optional)
WALK_FORWARD_CONFIG = {
    'train_window': 104,  # weeks
    'test_window': 26,    # weeks
    'step_size': 26       # weeks
}