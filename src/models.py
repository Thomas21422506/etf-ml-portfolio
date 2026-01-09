"""
Machine learning model training and evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mutual_info_score
from scipy import stats
import logging

from .utils import TRAIN_TEST_SPLIT, PROJECT_PATH

logger = logging.getLogger(__name__)


def train_and_evaluate_models(features, targets, etf_name):
    """
    Train classification models for a given ETF using the global split ratio.
    """
    target_col = f'{etf_name}_target_next_week'
    y = targets[target_col]
    X = features
    
    split_idx = int(len(X) * TRAIN_TEST_SPLIT)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    n_up = y_train.sum()
    n_down = len(y_train) - n_up
    
    if n_up > n_down:
        class_weights = {1: 1.0, 0: n_up/n_down}
    else:
        class_weights = {1: n_down/n_up, 0: 1.0}
    
    print(f"    Class balance {etf_name} - Up: {n_up}/{len(y_train)} "
          f"({n_up/len(y_train):.1%}), Weights: {class_weights}")

    models = {
        'Logistic Regression': LogisticRegression(
            random_state=42, max_iter=1000, 
            class_weight=class_weights,
            C=0.1,
            penalty='l2'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=50, random_state=42, max_depth=5,
            class_weight=class_weights
        ),
        'XGBoost': XGBClassifier(
            random_state=42, n_estimators=50, max_depth=3, 
            scale_pos_weight=n_down/n_up if n_up > 0 else 1.0,
            learning_rate=0.1,
            eval_metric='logloss'
        )
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc_roc': auc_roc,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    best_model_name = max(results.keys(), 
                         key=lambda x: results[x]['auc_roc'] * 0.4 + 
                                       results[x]['accuracy'] * 0.3 + 
                                       results[x]['recall'] * 0.3)
    
    best_model = results[best_model_name]['model']

    return results, X_test, y_test, best_model_name, best_model


def temporal_stratified_evaluation(features, targets, best_model_names, n_splits=5):
    """
    Temporal cross-validation without artificial stratification.
    """
    print("\n" + "="*80)
    print("[TEMPORAL VALIDATION] Time Series Cross-Validation")
    print("="*80)
    print(f"Configuration: {n_splits} folds temporels")
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    all_results = []
    etf_list = ['SPY', 'QQQ', 'EEM', 'TLT', 'GLD']
    
    for etf in etf_list:
        print(f"\n[{etf}] Évaluation temporelle avec {best_model_names[etf]}...")
        
        target_col = f'{etf}_target_next_week'
        y = targets[target_col].values
        X = features.values
        
        fold_metrics = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            n_up = y_train.sum()
            n_down = len(y_train) - n_up
            up_proportion = y_train.mean()
            
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
                    C=0.1,
                    penalty='l2',
                    class_weight=class_weights
                )
            
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            
            try:
                auc_roc = roc_auc_score(y_test, y_pred_proba)
            except:
                auc_roc = 0.5
            
            fold_metrics.append({
                'ETF': etf,
                'Model': model_name,
                'Fold': fold_idx,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'ROC_AUC': auc_roc,
                'Train_Size': len(X_train),
                'Test_Size': len(X_test),
                'Train_Up_Proportion': up_proportion
            })
            
            print(f"  Fold {fold_idx}: Acc={accuracy:.3f}, Prec={precision:.3f}, "
                  f"Rec={recall:.3f}, AUC={auc_roc:.3f}, Train_Up={up_proportion:.1%}")
        
        fold_df = pd.DataFrame(fold_metrics)
        avg_accuracy = fold_df['Accuracy'].mean()
        std_accuracy = fold_df['Accuracy'].std()
        avg_auc = fold_df['ROC_AUC'].mean()
        
        print(f"  → Moyenne: Acc={avg_accuracy:.3f} (±{std_accuracy:.3f}), AUC={avg_auc:.3f}")
        
        all_results.extend(fold_metrics)
    
    results_df = pd.DataFrame(all_results)
    
    print("\n" + "="*80)
    print("[TEMPORAL VALIDATION] Résumé global")
    print("="*80)
    
    summary = results_df.groupby(['ETF', 'Model'])[['Accuracy', 'Precision', 'Recall', 'ROC_AUC']].agg(['mean', 'std'])
    print(summary.round(3))
    
    results_df.to_csv(f'{PROJECT_PATH}/temporal_validation_results.csv', index=False)
    print(f"\n[TEMPORAL VALIDATION] Résultats sauvegardés: temporal_validation_results.csv")
    
    return results_df


def display_ml_metrics(etf_performance):
    """Displays and saves all ML performance metrics."""
    print("\n[PART 2] ML PERFORMANCE BY MODEL AND ETF")
    print("=" * 80)

    ml_metrics = []
    for etf in ['SPY', 'QQQ', 'EEM', 'TLT', 'GLD']:
        for model_name, result in etf_performance[etf].items():
            ml_metrics.append({
                'ETF': etf,
                'Model': model_name,
                'Accuracy': result['accuracy'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'ROC-AUC': result['auc_roc']
            })

    ml_metrics_df = pd.DataFrame(ml_metrics)

    print("\nAccuracy by ETF and Model:")
    accuracy_pivot = ml_metrics_df.pivot(index='ETF', columns='Model', values='Accuracy')
    print(accuracy_pivot.round(4))

    print("\nPrecision by ETF and Model:")
    precision_pivot = ml_metrics_df.pivot(index='ETF', columns='Model', values='Precision')
    print(precision_pivot.round(4))

    print("\nRecall by ETF and Model:")
    recall_pivot = ml_metrics_df.pivot(index='ETF', columns='Model', values='Recall')
    print(recall_pivot.round(4))

    print("\nROC-AUC by ETF and Model:")
    roc_pivot = ml_metrics_df.pivot(index='ETF', columns='Model', values='ROC-AUC')
    print(roc_pivot.round(4))

    print("\nModel-wise averages:")
    model_avg = ml_metrics_df.groupby('Model')[['Accuracy', 'Precision', 'Recall', 'ROC-AUC']].mean()
    print(model_avg.round(4))

    ml_metrics_df.to_csv(f'{PROJECT_PATH}/ml_model_performance.csv', index=False)
    print("\n[PART 2] ML metrics saved to: ml_model_performance.csv")

    return ml_metrics_df


def visualize_ml_performance(ml_metrics_df):
    """Visualizes ML performance across models and ETFs."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    metrics = ['Accuracy', 'Precision', 'Recall', 'ROC-AUC']

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        pivot = ml_metrics_df.pivot(index='ETF', columns='Model', values=metric)
        pivot.plot(kind='bar', ax=ax, alpha=0.7, edgecolor='black')
        ax.set_title(f'{metric} by ETF and Model', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric, fontsize=10)
        ax.set_xlabel('ETF', fontsize=10)
        ax.legend(title='Model', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

        if metric in ['Accuracy', 'Recall']:
            ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5)

    plt.tight_layout()
    plt.savefig('outputs/figures/ml_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()