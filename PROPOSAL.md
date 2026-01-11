# ETF ML Portfolio - Project Proposal

**Author**: Thomas Remandet (21422506)  
**Course**: Advanced Programming 2025  
**Institution**: HEC Lausanne

---

## Title

**Machine Learning for Weekly ETF Portfolio Allocation: A Rigorous Backtesting Framework with Walk-Forward Validation**

---

## Motivation

The idea for this project came from two sources. First, discussions with an independent wealth manager whose investment approach relied primarily on diversified ETF portfolios. He raised the question of whether simple predictive models could provide incremental value in a systematic allocation framework. Second, my own experience as an investor—I allocate part of my savings to broad market ETFs like MSCI World—made me curious about how machine learning might improve risk-adjusted returns in practice.

Traditional portfolio management relies on mean-variance optimization (Markowitz) or momentum heuristics. These approaches assume stable correlations and ignore non-linear relationships between market variables. Machine learning offers the potential to capture complex patterns and adapt to changing regimes.

However, financial ML is notoriously prone to look-ahead bias and overfitting. Studies showing impressive backtests often fail in live trading. This project prioritizes methodological rigor over extraordinary returns—the goal is a defensible experimental design that avoids common pitfalls.

**Research Question**: Can supervised ML classifiers predict the direction of next-week ETF returns better than random chance? If so, can these predictions generate superior risk-adjusted returns compared to strong benchmarks?

---

## Planned Approach & Technologies

### Data
- **Source**: CEDIF database (HEC Lausanne)
- **Period**: 2016-2025 (approximately 10 years, 509 weekly observations)
- **Assets**: 5 liquid ETFs (SPY, QQQ, EEM, TLT, GLD)
- **Frequency**: Weekly (Friday close prices)

### Feature Engineering (44 features)
**Technical Features** (35 - 7 per ETF):
- Lagged returns (1-week, 4-week)
- Momentum indicators (4-week, 12-week cumulative)
- Volatility (8-week rolling standard deviation)
- RSI (14-period Relative Strength Index)
- Moving average ratio (Price / MA_20W)
- Volume ratio (Volume / MA_20W)

**Macro/Regime Features** (9):
- Risk-on/off proxy (SPY momentum - TLT momentum)
- Flight-to-safety indicator (GLD momentum - SPY momentum)
- Cross-asset correlations (SPY-QQQ, SPY-EEM rolling)
- Market breadth (% ETFs with positive 4W return)
- Cross-sectional dispersion

**Critical Design Principle**: Features at time t use ONLY data up to time t. Target is return at t+1. This ensures zero look-ahead bias.

### Machine Learning Models
Three models tested per ETF (15 models total):

1. **Logistic Regression**
   - Linear baseline with L2 regularization (C=0.1)
   - Fast, interpretable, resistant to overfitting

2. **Random Forest**
   - 50 trees, max_depth=5
   - Captures non-linearities and feature interactions

3. **XGBoost**
   - 50 estimators, max_depth=3, learning_rate=0.1
   - State-of-the-art gradient boosting

**Target Variable**: Binary classification - predict if next week's return > 0 (UP) or ≤ 0 (DOWN)

**Model Selection**: Best model per ETF chosen by ROC-AUC on validation set

### Allocation Strategy
**ML Strategy**:
1. For each week, predict P(UP) for each of the 5 ETFs
2. Rank ETFs by predicted probability (highest first)
3. Select top 3 ETFs
4. Allocate equally: 33.33% each
5. Rebalance weekly

**Fallback Rule**: If all P(UP) < 50%, still invest in top 3 (long-only constraint)

### Benchmark Strategies
1. **Buy & Hold SPY**: 100% S&P 500, no rebalancing
2. **Equal Weight**: 20% per ETF, weekly rebalancing
3. **Momentum 4W**: Top 3 ETFs by 4-week return
4. **Markowitz Min-Variance**: Rolling 104-week covariance optimization
5. **Markowitz Max-Sharpe**: Rolling 104-week mean-variance optimization

### Validation Framework
**Train/Test Split**:
- 80/20 chronological split (no random shuffle)
- Train: 407 weeks (approximately 8 years)
- Test: 102 weeks (approximately 2 years)

**Walk-Forward Analysis** (robustness test):
- 15 rolling windows
- Train window: 104 weeks (2 years)
- Test window: 26 weeks (6 months)
- Step size: 26 weeks (non-overlapping)
- Models completely retrained at each window

**Performance Metrics**:
- Sharpe Ratio (primary metric for risk-adjusted returns)
- Annualized Return & Volatility
- Maximum Drawdown
- Win Rate
- ML Metrics: Accuracy, ROC-AUC, Precision, Recall

**Statistical Tests**:
- Binomial test: Is prediction accuracy significantly > 50%?
- Mutual information: Which features are most informative?

### Implementation Architecture
**Modular Pipeline** (7 Python modules in `src/`):
1. `data_loader.py`: Load and preprocess ETF price data
2. `features.py`: Engineer 44 technical and macro features
3. `models.py`: Train and evaluate 3 ML models per ETF
4. `backtest.py`: Implement allocation logic and compute returns
5. `benchmarks.py`: Calculate 5 benchmark strategy returns
6. `evaluation.py`: Walk-forward validation and statistical tests
7. `utils.py`: Constants and helper functions

**Interactive Dashboard**: Streamlit-based interface with 6 pages:
- Home: Project overview and key metrics
- Data & Methodology: Dataset and temporal integrity explanation
- ML Models: Performance by ETF, statistical significance
- Trading Strategies: Strategy descriptions and logic
- Results & Backtesting: Performance charts, walk-forward analysis
- Conclusions: Key learnings and limitations

---

## Technologies

| Category | Libraries | Purpose |
|----------|-----------|---------|
| **Data Processing** | pandas, numpy | Data manipulation, feature engineering |
| **Machine Learning** | scikit-learn, xgboost | Model training (Logistic, RF, XGBoost) |
| **Optimization** | scipy | Portfolio optimization (Markowitz) |
| **Visualization** | matplotlib, seaborn, plotly | Static and interactive charts |
| **Dashboard** | streamlit | Interactive web interface |
| **Statistics** | scipy.stats | Binomial tests, mutual information |
| **Version Control** | git, GitHub | Code versioning and collaboration |

**Python Version**: 3.10+ (tested on 3.10, 3.11, 3.12)

**Reproducibility**: Fixed random seeds (`random_state=42`) everywhere for deterministic results

---

## Expected Challenges & Mitigation

### Challenge 1: Look-Ahead Bias
**Risk**: Accidentally using future data in features (e.g., calculating momentum with t+1 data)

**Mitigation**:
- Strict temporal discipline: features[t] → target[t+1]
- Code review for all `.shift()` operations
- Unit tests to verify no future data leakage

### Challenge 2: Overfitting
**Risk**: Models memorize training data, fail on new data

**Mitigation**:
- Simple models with regularization (L2, max_depth limits)
- Walk-forward validation with 15 independent test periods
- No hyperparameter tuning on test set
- Class balancing to prevent majority-class memorization

### Challenge 3: Survivorship Bias
**Risk**: ETFs that survived 10 years may not represent future opportunities

**Mitigation**:
- Acknowledge limitation in documentation
- Use highly liquid, major ETFs (SPY, QQQ) likely to persist
- Future work: expand universe to 20-30 ETFs

### Challenge 4: Transaction Costs
**Risk**: Backtests ignore trading costs, inflating returns

**Mitigation**:
- Transparent reporting: "Results exclude transaction costs"
- Estimate impact: approximately 130 trades/year × 5-10 bps = -3-5% annual drag
- Acknowledge this as primary limitation in all documentation
- Future work: implement realistic cost model

---

## Success Criteria

### Minimum Viable Product

**Core Functionality**:
- Pipeline executes end-to-end without errors
- ML models train successfully for all 5 ETFs
- Allocation strategy generates valid portfolio returns
- Walk-forward validation completes with 15 windows

**Statistical Validity**:
- ML models beat random chance (accuracy > 50%) with p < 0.05
- At least 2 out of 5 ETFs show statistically significant predictions

**Performance**:
- ML strategy beats Buy & Hold SPY on Sharpe ratio (before costs)
- Walk-forward mean Sharpe > 0.5 (positive risk-adjusted returns)

**Documentation**:
- Clean, modular code with docstrings
- README with installation instructions and methodology
- GitHub repository with version control
- Interactive dashboard with 6 pages

**Transparency**:
- Limitations clearly documented (especially transaction costs)
- No cherry-picking of results
- Walk-forward distribution shown (good and bad periods)

---


### Enhancement 1: Transaction Cost Modeling
Implement realistic trading costs (5-10 basis points per trade) to assess true strategy viability.

### Enhancement 2: Hyperparameter Optimization
GridSearchCV with TimeSeriesSplit to optimize model parameters without overfitting.

### Enhancement 3: Feature Selection
LASSO, Recursive Feature Elimination, or mutual information ranking to identify most predictive features.

### Enhancement 4: Expanded Universe
Add 15-20 additional ETFs (sectors, international, bonds, commodities) for richer signal space.

### Enhancement 5: Advanced Models
LSTM for sequential patterns, ensemble methods (stacking, voting), or online learning for continuous adaptation.

---


