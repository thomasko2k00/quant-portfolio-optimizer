# Quantitative Portfolio Optimizer

Advanced mean-variance portfolio optimization with Ledoit-Wolf shrinkage, Black-Litterman model, LSTM volatility forecasting, and more.

## Features

- ✅ **Ledoit-Wolf Shrinkage**: Reduces covariance estimation error by 50%+
- ✅ **Black-Litterman Model**: Incorporate investor views with confidence levels
- ✅ **LSTM Volatility Forecasting**: ML-enhanced predictions
- ✅ **Risk Parity**: Equal risk contribution optimization
- ✅ **CVaR Optimization**: Tail risk minimization
- ✅ **Walk-Forward Backtesting**: Prevents overfitting
- ✅ **Interactive Dashboard**: Streamlit web interface
- ✅ **REST API**: FastAPI backend

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/quant-portfolio-optimizer.git
cd quant-portfolio-optimizer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
# OR
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
