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

### Installation (Windows)

1. **Download and extract** the project ZIP

2. **Open in VS Code**:
   - Right-click on folder → "Open with Code"
   - Or: `File > Open Folder`

3. **Open terminal** in VS Code:
   - Press `Ctrl + `` ` (backtick)
   - Or: `View > Terminal`

4. **Run installation script**:
```cmd
   install_and_run.bat
```
   
   This will:
   - Create virtual environment
   - Install all dependencies
   - Generate results (~10-15 min)
   - Launch dashboard automatically

5. **Open dashboard** in browser:
```
   http://localhost:8501
```

### Installation (Linux/Mac)

1. **Extract the project**:
```bash
   unzip projet_etf_ml.zip
   cd projet_etf_ml
```

2. **Run installation script**:
```bash
   chmod +x setup.sh
   ./setup.sh
```

3. **Launch dashboard**:
```bash
   streamlit run dashboard.py
```

### Manual Installation (All Platforms)

If automated scripts fail:
```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. Upgrade pip
python -m pip install --upgrade pip

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

# 6. Launch dashboard
streamlit run dashboard.py
```

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

**Author**: Thomas Remandet  
**Email**: thomas.remandet@unil.ch  
**Institution**: HEC Lausanne - Master in Finance  
**Course**: Advanced Programming 2025

**Resources**:
- GitHub: [Repository link]
- Dashboard: `streamlit run dashboard.py`
- Full Report: Available in repository

---

## License

This project is an academic work for educational purposes.

---

## Acknowledgments

- **Data Source**: CEDIF database (HEC Lausanne)
- **Inspiration**: Systematic trading research, academic papers on ML in finance
- **Tools**: Open-source Python ecosystem (pandas, scikit-learn, streamlit)
- **Guidance**: Advanced Programming course instructors

