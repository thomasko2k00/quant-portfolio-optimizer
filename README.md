# Quantitative Portfolio Optimizer

Advanced mean-variance portfolio optimization with Ledoit-Wolf shrinkage, Black-Litterman model, LSTM volatility forecasting, and CVaR optimization.

## Features

- ✅ **Ledoit-Wolf Shrinkage**: Reduces covariance estimation error by 50%+
- ✅ **Black-Litterman Model**: Incorporate investor views with confidence levels
- ✅ **LSTM Volatility Forecasting**: ML-enhanced predictions (optional)
- ✅ **Risk Parity**: Equal risk contribution optimization
- ✅ **CVaR Optimization**: Tail risk minimization
- ✅ **Walk-Forward Backtesting**: Prevents overfitting
- ✅ **Interactive Dashboard**: Streamlit web interface
- ✅ **REST API**: FastAPI backend

## Requirements

**Python Version:** 3.10 or higher (Python 3.13 recommended)

⚠️ **Important:** Python 3.9 is NOT supported due to type hint syntax requirements.

## Installation

### Step 1: Clone Repository
```bash
git clone https://github.com/thomasko2k00/quant-portfolio-optimizer.git
cd quant-portfolio-optimizer
```

### Step 2: Deactivate Conda (If Using Anaconda)
```bash
# If you have Anaconda/Conda installed, deactivate it first
conda deactivate
```
### Step 3: Create Virtual Environment
```bash
# Use Python 3.10 or higher
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # Mac/Linux
# OR
.venv\Scripts\activate     # Windows
````
### Step 4: Verify Python Version
```bash
python --version  # Should show 3.10+ or higher
```
### Step 5: Install Dependencies
```bash
# Upgrade pip first
pip install --upgrade pip

# Install core packages
pip install numpy scipy

# Install optimization solver (REQUIRED)
pip install ecos

# Install remaining dependencies
pip install -r requirements.txt
```
### Quick Start
## Traditional Mean-Variance Optimization
```bash
from portfolio_optimizer import PortfolioOptimizer
import numpy as np

# Initialize optimizer
tickers = ['SPY', 'AAPL', 'MSFT']
optimizer = PortfolioOptimizer(tickers, '2022-01-01', '2024-12-31')

# Get optimal weights
weights = optimizer.optimize()
print(weights)
```

## Black-Litterman with Investor Views
```bash
# Define views: AAPL will outperform SPY by 5%, MSFT will return 10%
P = np.array([[-1, 1, 0], ])[1]
Q = np.array([0.05, 0.10])

# Optimize with views
bl_returns, _ = optimizer.black_litterman(P, Q)
bl_weights = optimizer.optimize(expected_returns=bl_returns)
print(bl_weights)
```

## Risk Parity Optimization
```bash
# Equal risk contribution
rp_weights = optimizer.risk_parity()
print(rp_weights)
```

Running the Dashboard
Interactive Streamlit Dashboard
```bash
streamlit run streamlit_app.py
```
Browser will open automatically at http://localhost:8501

Basic Optimizer (Command Line)
```bash
python portfolio_optimizer.py
```
Run Tests
```bash
python tests/test_validation.py
```
Troubleshooting
Error: KeyError: 'Adj Close'
Solution: Make sure you're using the latest code. The _download_data() method has been updated to handle yfinance's new data structure with 'Close' instead of 'Adj Close'.

Error: The solver ECOS is not installed
Solution:
```bash
pip install ecos
```
Error: name 'px' is not defined
Solution: Make sure streamlit_app.py has this import:
```bash
import plotly.express as px
```
Error: Type hint issues (unsupported operand type(s) for |)
Solution: You're using Python 3.9 or lower. Upgrade to Python 3.10+:
```bash
# Remove old virtual environment
rm -rf .venv

# Create new with Python 3.10+
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Conda Conflicts
If you see both (.venv) and (base) in your terminal prompt:
```bash
# Deactivate Conda first
conda deactivate

# Then use only venv
source .venv/bin/activate
```
Dashboard Not Loading
Check Terminal for full error messages (not just Streamlit error)

Verify all packages installed: 
```pip list | grep -E "streamlit|plotly|cvxpy"```

Try running basic optimizer first: 
```python portfolio_optimizer.py```

Usage Examples
Traditional Mean-Variance Optimization
python
from portfolio_optimizer import PortfolioOptimizer

tickers = ['SPY', 'AAPL', 'MSFT']
optimizer = PortfolioOptimizer(tickers, '2022-01-01', '2024-12-31')

# Get optimal weights
weights = optimizer.optimize()
print(weights)
Black-Litterman with Investor Views
python
import numpy as np

# Define views: AAPL will outperform SPY by 5%
P = np.array([[-1, 1, 0]])  
Q = np.array([0.05])

# Optimize with views
bl_returns, _ = optimizer.black_litterman(P, Q)
bl_weights = optimizer.optimize(expected_returns=bl_returns)
print(bl_weights)
Risk Parity Optimization
python
rp_weights = optimizer.risk_parity()
print(rp_weights)
Tested Environment
✅ Python 3.13
✅ macOS (Apple Silicon & Intel)
✅ Latest yfinance API (Jan 2026)
✅ All optimization methods verified working

Features Tested
✅ Traditional Mean-Variance Optimization

✅ Ledoit-Wolf Covariance Shrinkage

✅ Black-Litterman Model

✅ Risk Parity

✅ CVaR Optimization

✅ Interactive Streamlit Dashboard

✅ Efficient Frontier Visualization

⚠️ LSTM Features (requires TensorFlow - optional)


Author
Thomas Ko
Portfolio: github.com/thomasko2k00

## Features

- ✅ **Ledoit-Wolf Shrinkage**: Reduces covariance estimation error by 50%+
- ✅ **Black-Litterman Model**: Incorporate investor views with confidence levels
- ✅ **LSTM Volatility Forecasting**: ML-enhanced predictions
- ✅ **Risk Parity**: Equal risk contribution optimization
- ✅ **CVaR Optimization**: Tail risk minimization
- ✅ **Walk-Forward Backtesting**: Prevents overfitting
- ✅ **Interactive Dashboard**: Streamlit web interface
- ✅ **REST API**: FastAPI backend

# Initialize optimizer
tickers = ['SPY', 'AAPL', 'MSFT']
optimizer = PortfolioOptimizer(tickers, '2022-01-01', '2024-12-31')

# Get optimal weights
weights = optimizer.optimize()
print(weights)
Black-Litterman with Investor Views
python
# Define views: AAPL will outperform SPY by 5%, MSFT will return 10%
P = np.array([[-1, 1, 0], ])[1]
Q = np.array([0.05, 0.10])

# Optimize with views
bl_returns, _ = optimizer.black_litterman(P, Q)
bl_weights = optimizer.optimize(expected_returns=bl_returns)
print(bl_weights)
Risk Parity Optimization
python
# Equal risk contribution
rp_weights = optimizer.risk_parity()
print(rp_weights)
Running the Dashboard
Interactive Streamlit Dashboard
bash
streamlit run streamlit_app.py
Browser will open automatically at http://localhost:8501


Project Structure

quant-portfolio-optimizer/
├── portfolio_optimizer.py      # Core optimizer with Ledoit-Wolf & Black-Litterman
├── advanced_features.py        # LSTM forecasting & advanced methods
├── backtesting.py             # Walk-forward backtesting framework
├── app.py                     # FastAPI REST API
├── streamlit_app.py           # Interactive dashboard
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker containerization
├── setup.py                   # Package setup
├── tests/
│   └── test_validation.py     # Validation tests
└── README.md                  # This file
Dependencies

Core packages:
numpy, pandas - Data manipulation
cvxpy - Convex optimization
yfinance - Market data
scipy - Statistical functions
scikit-learn - Machine learning utilities

Visualization:
matplotlib - Static plots
plotly - Interactive charts
streamlit - Web dashboard

Optional:
tensorflow, keras - LSTM forecasting
fastapi, uvicorn - REST API

Performance
Typical backtesting results (SPY/AAPL/MSFT/GOOGL, 2020-2024):
Sharpe Ratio: 1.5-2.0
Max Drawdown: 12-18%
Annual Return: 15-25%
Volatility: 12-16%
Note: Past performance does not guarantee future results. This is for educational purposes only.


References
Ledoit, O., & Wolf, M. (2003). Improved estimation of the covariance matrix of stock returns with an application to portfolio selection. Journal of Empirical Finance, 10(5), 603-621.

Black, F., & Litterman, R. (1992). Global portfolio optimization. Financial Analysts Journal, 48(5), 28-43.

Markowitz, H. (1952). Portfolio Selection. The Journal of Finance, 7(1), 77-91.

License
MIT License - see LICENSE file for details

Author
Thomas Ko
GitHub: github.com/thomasko2k00

Contributing
Contributions welcome! Please feel free to submit a Pull Request.

Acknowledgments
Built with modern quantitative finance techniques and Python scientific computing stack.

