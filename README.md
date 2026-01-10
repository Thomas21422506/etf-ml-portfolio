# ETF ML Portfolio - Machine Learning for Weekly ETF Allocation

Systematic portfolio allocation using machine learning to predict weekly ETF returns, with rigorous temporal integrity, walk-forward validation, and interactive dashboard.

---

## Project Overview

This data science course project implements a machine learning-based weekly allocation strategy for 5 liquid ETFs (SPY, QQQ, EEM, TLT, GLD). The focus is on methodological rigor rather than extraordinary returns: preventing look-ahead bias, walk-forward validation, and transparent limitation reporting.

### Key Features

- Temporal Integrity: No data leakage - features[t] → target[t+1]
- 3 ML Models: Logistic Regression, Random Forest, XGBoost with class balancing
- Walk-Forward Validation: 15 rolling windows (104W train, 26W test)
- Strong Benchmarks: Equal Weight, Momentum, Markowitz (Min-Var & Max-Sharpe)
- Risk Management: Stop-loss, trailing stop, volatility scaling
- Interactive Dashboard: Streamlit-based visualization and exploration

### Research Question

*Can supervised ML classifiers predict the sign of next-week ETF returns better than chance? If so, can these predictions improve a weekly allocation strategy compared to strong benchmarks?*

**Answer**: Modest improvements (3-5% extra annually) in some regimes, but **not systematic**. Primary value: risk management and allocation discipline.

---

## Quick Start

### Prerequisites

- **Python 3.8+** (tested on 3.10+)
- **pip** package manager
- **10-20 minutes** for first-time setup and execution

## Installation

### Prerequisites

- **Python 3.10+** installed on your system
- **pip** package manager
- **10-20 minutes** for first-time setup and execution

### Setup Instructions

1. **Clone or download** the repository:
```bash
   git clone https://github.com/Thomas21422506/etf-ml-portfolio.git
   cd etf-ml-portfolio
```


2. **Install dependencies**:
```bash
   pip install -r requirements.txt
```

3. **Generate results**:
```bash
   python main.py or python3 main.py
```
   **Note**: Close pop-up chart windows (press Q or click X) to continue execution.

5. **Launch dashboard**:
```bash
   streamlit run dashboard.py
```
   The dashboard opens automatically at `http://localhost:8501`

### Quick Install (one command)

```bash
pip install -r requirements.txt && python main.py && streamlit run dashboard.py
```

# 4. Install dependencies
pip install pandas
pip install numpy
pip install scikit-learn
pip install xgboost
pip install matplotlib
pip install seaborn
pip install streamlit
pip install plotly
pip install scipy

# Or use requirements.txt:
pip install -r requirements.txt

# 5. Generate results
python main.py
# Note: Close any pop-up chart windows (press Q or click X) to continue execution

# 6. Launch dashboard
streamlit run dashboard.py
```

---

## Important Notes

### Closing Visualization Windows

When running `python main.py`, the pipeline generates several visualization charts that will automatically display in pop-up windows. **You must close each chart window** for the execution to continue:

- Charts include: model performance comparisons, portfolio evolution, walk-forward results, etc.
- **How to close**: Click the X button on the window or press `Q`
- The terminal will show a message indicating it's waiting for you to close the chart
- Once closed, the pipeline continues automatically to the next step

**Tip**: If you prefer to skip chart displays and only generate saved PNG files, you can modify the chart generation functions in the code to use `plt.savefig()` without `plt.show()`.

### First-Time Execution Time

- Complete execution of `main.py` takes **10-15 minutes**
- This includes:
  - Feature engineering (44 features for 509 weeks)
  - Training 15 ML models (3 models × 5 ETFs)
  - Backtesting 8 strategies
  - Walk-forward validation (15 rolling windows)
  - Generating and saving visualization charts

### Dashboard Launch

When launching the dashboard with `streamlit run dashboard.py`:
- **First time**: Streamlit may prompt for an email address - simply press `Enter` to skip
- Dashboard opens automatically at `http://localhost:8501`
- If port 8501 is busy, use: `streamlit run dashboard.py --server.port 8502`

---

## Project Structure
```
etf-ml-portfolio/
├── main.py                      # Main execution pipeline
├── dashboard.py                 # Interactive Streamlit dashboard
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── src/                         # Source code modules
│   ├── __init__.py
│   ├── data_loader.py           # Load and preprocess ETF data
│   ├── features.py              # Feature engineering (44 features)
│   ├── models.py                # ML model training and evaluation
│   ├── backtest.py              # Portfolio backtesting logic
│   ├── benchmarks.py            # Benchmark strategies (Markowitz, Momentum)
│   ├── evaluation.py            # Walk-forward validation
│   └── utils.py                 # Helper functions and constants
│
└── outputs/                     # Generated results (created after main.py)
    └── results/
        ├── etf_features.csv              # Engineered features
        ├── etf_targets.csv               # Binary targets (up/down)
        ├── etf_weekly_returns.csv        # Weekly returns per ETF
        ├── ml_model_performance.csv      # Model metrics (Accuracy, ROC-AUC, etc.)
        ├── final_performance_results.csv # Strategy comparison (Sharpe, Return, etc.)
        ├── walk_forward_results.csv      # Robustness analysis (15 windows)
        └── all_strategy_returns.csv      # Weekly returns per strategy
```

---

## Methodology

### 1. Data & Features

**Dataset**:
- **Source**: CEDIF database (HEC Lausanne)
- **Period**: 2016-2025 (~10 years)
- **Frequency**: Weekly (Friday close)
- **ETFs**: SPY, QQQ, EEM, TLT, GLD (5 liquid ETFs)
- **Observations**: 509 weeks (after feature engineering)

**Feature Engineering** (44 features total):

**Technical Features (35)** - Per ETF (7 × 5):
- Lagged returns: 1-week, 4-week
- Momentum: 4-week, 12-week cumulative
- Volatility: 8-week rolling std
- RSI (Relative Strength Index, 14 periods)
- MA ratio: Price / 20-week moving average
- Volume ratio: Volume / 20-week average volume

**Macro/Regime Features (9)**:
- Risk-on/off proxy: SPY momentum - TLT momentum
- Flight-to-safety: GLD momentum - SPY momentum
- Correlations: Rolling SPY-QQQ, SPY-EEM
- Market breadth: % ETFs with positive 4W return
- Dispersion: Cross-sectional std of returns

### 2. Machine Learning Models

**3 Models Tested (per ETF)**:

1. **Logistic Regression**
   - Linear baseline
   - L2 regularization (C=0.1)
   - Class weights for imbalance

2. **Random Forest**
   - 50 trees, max_depth=5
   - Bootstrap sampling
   - Class weights for imbalance

3. **XGBoost**
   - 50 estimators, max_depth=3
   - Learning rate=0.1
   - `scale_pos_weight` for imbalance

**Model Selection**: Best model per ETF by ROC-AUC on validation set.

**Target**: Binary classification - predict if next week's return > 0 (UP) or ≤ 0 (DOWN).

### 3. Allocation Strategy

**ML Strategy (Base)**:
1. Predict P(up) for each ETF
2. Rank by probability (highest first)
3. Select top 3 ETFs
4. Allocate equally: 33.33% each
5. Rebalance weekly

**ML + Risk Management**:
- Stop-loss: Exit position if loss > 3%
- Trailing stop: Reduce to 50% if drawdown > 10%
- Volatility scaling: Target 15% annual volatility

**Fallback rule**: If all P(up) < 0.50, still invest in top 3 (long-only constraint).

### 4. Benchmark Strategies

1. **Buy & Hold SPY**: 100% SPY, no rebalancing
2. **Equal Weight**: 20% each ETF, weekly rebalancing
3. **Momentum 4W**: Top 3 by 4-week return
4. **Markowitz Min-Variance**: 104W rolling optimization
5. **Markowitz Max-Sharpe**: 104W rolling optimization

### 5. Validation

**Train/Test Split**:
- 80/20 chronological split
- Train: 407 weeks (~8 years)
- Test: 102 weeks (~2 years)

**Walk-Forward Analysis**:
- 15 rolling windows
- Train: 104 weeks (2 years)
- Test: 26 weeks (6 months)
- Step: 26 weeks (non-overlapping)

**Metrics**:
- Sharpe Ratio, Annual Return, Volatility
- Max Drawdown, Win Rate
- ROC-AUC, Accuracy, Precision, Recall

---

## Key Results

### ML Model Performance (Test Set - 102 weeks)

| ETF | Best Model | Accuracy | ROC-AUC | Precision | Recall |
|-----|-----------|----------|---------|-----------|--------|
| **QQQ** | XGBoost | 0.618 | 0.642 | 0.655 | 0.623 |
| **SPY** | Random Forest | 0.578 | 0.587 | 0.590 | 0.612 |
| **EEM** | XGBoost | 0.569 | 0.581 | 0.574 | 0.589 |
| **TLT** | Logistic Reg | 0.539 | 0.548 | 0.543 | 0.556 |
| **GLD** | Random Forest | 0.529 | 0.536 | 0.531 | 0.547 |

**Key Findings:**
- QQQ is the most predictable ETF (62% accuracy, statistically significant p<0.05)
- Technology and equity ETFs (SPY, QQQ) outperform bonds and commodities (TLT, GLD)
- All models beat random chance (50%) but with varying degrees of significance

### Strategy Performance (Test Set - 102 weeks)

| Strategy | Sharpe Ratio | Annual Return | Volatility | Max Drawdown | Win Rate |
|----------|--------------|---------------|------------|--------------|----------|
| **ML Strategy** | **1.48** | **20.3%** | 13.7% | -12.4% | 58.8% |
| ML + Risk Mgmt | 1.42 | 18.9% | 13.3% | -11.2% | 57.8% |
| Momentum 4W | 1.65 | 21.8% | 13.2% | -10.8% | 59.8% |
| Markowitz Max-Sharpe | 1.52 | 19.7% | 13.0% | -11.5% | 56.9% |
| Equal Weight | 1.38 | 18.2% | 13.2% | -13.1% | 55.9% |
| Buy & Hold SPY | 1.35 | 18.7% | 13.8% | -14.2% | 54.9% |

**Key Findings:**
- ML Strategy achieves 1.48 Sharpe, beating SPY by ~1.6% annually
- Momentum 4W slightly outperforms (1.65 Sharpe) - simpler strategy, competitive results
- Risk management adds stability (lower max DD) but slightly reduces returns
- **Important**: These results exclude transaction costs

### Walk-Forward Validation (15 windows)

| Metric | Mean | Median | Std Dev | Min | Max |
|--------|------|--------|---------|-----|-----|
| **Sharpe Ratio** | 0.76 | 0.82 | 1.24 | -1.45 | 2.38 |
| **Annual Return** | 12.4% | 13.7% | 18.2% | -28.3% | 41.2% |
| **Win Rate** | 54.3% | 55.1% | 8.7% | 38.5% | 68.2% |

**Key Findings:**
- Performance is **regime-dependent**: strong in bull markets, weak in bear markets
- High variance across windows (Sharpe std=1.24) indicates lack of robustness
- Mean Sharpe of 0.76 is significantly lower than single test period (1.48)
- 11 out of 15 windows are profitable (73% consistency)

**Interpretation:** The single test period result (Sharpe 1.48) was partially lucky. True expected performance is closer to Sharpe 0.76, which is still positive but more modest.

---

## Dashboard Features

Launch the interactive dashboard:
```bash
streamlit run dashboard.py
```

The dashboard will automatically open in your default browser at `http://localhost:8501`

**First Launch**: When running Streamlit for the first time, you'll be prompted for an email address. This is optional—simply press `Enter` to skip and proceed to the dashboard.

### Dashboard Pages

**6 Interactive Sections**:

1. **Home**: Project overview, motivation, key metrics
2. **Data & Methodology**: Dataset description, feature engineering, temporal integrity explanation
3. **ML Models**: Performance metrics by ETF, model comparison radar charts, statistical significance tests
4. **Trading Strategies**: ML base strategy, risk management variants, benchmark descriptions
5. **Results & Backtesting**: Portfolio evolution charts, Sharpe ratio comparisons, walk-forward analysis
6. **Conclusions**: Key learnings, limitations, future work recommendations

### Interactive Elements

- **ETF Selection**: Choose specific ETFs for detailed performance analysis
- **Metric Comparison**: Switch between Accuracy, Precision, Recall, and ROC-AUC
- **Strategy Selection**: Compare multiple strategies on the same chart
- **Walk-Forward Distribution**: Explore robustness across different time periods

### Navigation

Use the sidebar on the left to navigate between sections. Each page builds on the previous one to tell the complete story of the methodology and results.

---

## Limitations & Caveats

### Critical Limitations

1. **Transaction Costs NOT Modeled**
   - Realistic costs: 5-10 bps per trade
   - ML Strategy: ~130 trades/year → **-3 to -5% annual drag**
   - After costs: ML likely **underperforms** Buy & Hold SPY
   - **Priority #1 for future work**

2. **Small Universe**
   - Only 5 ETFs (limited cross-sectional signals)
   - Cannot exploit sector rotations
   - Future: expand to 20-30 ETFs

3. **Technical Features Only**
   - No fundamentals (P/E, earnings)
   - No macro data (unemployment, inflation, GDP)
   - No alternative data (sentiment, news)

4. **Short Sample Period**
   - 6 years (2016-2025) only
   - Missing 2008 financial crisis
   - COVID-19 only major stress test

5. **No Hyperparameter Optimization**
   - Parameters fixed arbitrarily
   - GridSearchCV could improve 2-5%
   - Risk of overfitting if not careful

### Benchmark Caveats

**Markowitz strategies are OPTIMISTIC**:
- 104-week lookback (assumes stable correlations)
- Weekly rebalancing (ignores costs)
- In reality: costs would reduce Markowitz Sharpe by ~0.5-1.0

---

## Technologies Used

| Category | Libraries |
|----------|-----------|
| **Data Processing** | pandas, numpy |
| **Machine Learning** | scikit-learn, xgboost |
| **Optimization** | scipy |
| **Visualization** | matplotlib, seaborn, plotly |
| **Dashboard** | streamlit |
| **Statistics** | scipy.stats |

**Python Version**: 3.8+ (tested on 3.10, 3.11, 3.12)

---

## Academic Contribution

### What This Project Demonstrates

**Not**: Extraordinary ML returns (results are modest)

**Yes**: Rigorous experimental design for time-series ML in finance

**Key Methodological Contributions**:
1. Temporal integrity enforcement (no look-ahead bias)
2. Walk-forward validation (robustness across regimes)
3. Strong benchmark comparisons (Momentum, Markowitz)
4. Transparent limitation reporting (honest about failures)
5. Reproducible pipeline (single `main.py` execution)

**Core Message**: In financial ML, **avoiding mistakes is more important than finding alpha**. A mediocre strategy without look-ahead bias is infinitely more valuable than a "great" strategy that cheats.

---

## Future Work

### Short Term (1-2 months)

1. **Transaction Cost Modeling** (PRIORITY)
   - 5-10 bps per trade
   - Bid-ask spread, slippage
   - Impact on all strategies

2. **Hyperparameter Optimization**
   - GridSearchCV with TimeSeriesSplit
   - Nested cross-validation

3. **Feature Selection**
   - LASSO, RFE, feature importance
   - Remove low-value features

### Medium Term (6+ months)

1. **Expand Universe**
   - 20-30 ETFs (sectors, international)
   - More cross-sectional signals

2. **Richer Features**
   - Macro: VIX, yield curve, unemployment
   - Fundamentals: P/E, earnings growth
   - Alternative data: sentiment, news

3. **Advanced Models**
   - Ensemble methods (stacking, voting)
   - Deep learning (LSTM, Transformers)
   - Online learning (continuous retraining)

4. **Regime Detection**
   - Hidden Markov Models
   - Activate ML only in favorable regimes
   - Defensive allocation in bear markets

---

## Contact & Resources

**Author**: Thomas Remandet, 21422506  
**Email**: thomas.remandet@unil.ch  
**Course**: Advanced Programming 2025

**Resources**:
- GitHub: https://github.com/Thomas21422506/etf-ml-portfolio
- Dashboard: `streamlit run dashboard.py`
- Full Report: Available in repository

---

## Acknowledgments

- **Data Source**: CEDIF database (HEC Lausanne)
- **Inspiration**: Systematic trading research, academic papers on ML in finance
- **Tools**: Open-source Python ecosystem (pandas, scikit-learn, streamlit)
- **Guidance**: Advanced Programming course instructors

---
