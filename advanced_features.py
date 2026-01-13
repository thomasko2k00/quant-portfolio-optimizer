"""
Advanced Portfolio Features (Prompt 3)
- LSTM Volatility Forecasting
- Risk Parity Optimization
- CVaR (Conditional Value at Risk)
- FastAPI Backend

Evidence:
- LSTM improves dynamic volatility forecasting vs GARCH
- Risk parity provides better diversification
- CVaR captures tail risk better than variance
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from portfolio_optimizer import PortfolioOptimizer
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')


class AdvancedPortfolioOptimizer(PortfolioOptimizer):
    """
    Extended optimizer with ML and advanced risk measures
    """

    def __init__(self, tickers, start_date, end_date):
        """
        Initialize portfolio optimizer with market data
        """
        # Handle both list and comma-separated string
        if isinstance(tickers, str):
            self.tickers = [t.strip() for t in tickers.split(',') if t.strip()]
        else:
            self.tickers = [t for t in tickers if t]

        if not self.tickers:
            raise ValueError("No valid tickers provided")

        self.start_date = start_date
        self.end_date = end_date

        # Download data
        try:
            raw_data = yf.download(self.tickers, start=start_date, end=end_date, progress=False)

            # Handle both single and multiple tickers
            if len(self.tickers) == 1:
                self.data = pd.DataFrame(raw_data['Adj Close'])
                self.data.columns = self.tickers
            else:
                if isinstance(raw_data.columns, pd.MultiIndex):
                    self.data = raw_data['Adj Close']
                else:
                    self.data = raw_data

            # Validate data
            if self.data.empty:
                raise ValueError(f"No data downloaded for {self.tickers}")

            # Drop NaN values
            self.data = self.data.dropna()

            if self.data.empty:
                raise ValueError("No valid data after cleaning")

            # Calculate returns
            self.returns = self.data.pct_change().dropna()

        except Exception as e:
            raise ValueError(f"Error downloading data: {str(e)}")

    def train_lstm_volatility(self, lookback=60, epochs=50):
        """
        Train LSTM for volatility forecasting

        Args:
            lookback: Number of days to look back
            epochs: Training epochs

        Returns:
            Trained Keras model
        """
        try:
            from tensorflow import keras
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
        except ImportError:
            print("⚠️  TensorFlow not installed. Install with: pip install tensorflow")
            return None

        # Calculate rolling volatility
        volatility = self.returns.rolling(window=20).std() * np.sqrt(252)
        volatility = volatility.dropna()

        X, y = [], []
        vol_values = volatility.values

        for i in range(lookback, len(vol_values)):
            X.append(vol_values[i - lookback:i])
            y.append(vol_values[i])

        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], volatility.shape[1]))

        # Build LSTM model
        model = Sequential([
            LSTM(50, activation='relu', return_sequences=True,
                 input_shape=(lookback, volatility.shape[1])),
            Dropout(0.2),
            LSTM(50, activation='relu'),
            Dropout(0.2),
            Dense(volatility.shape[1])
        ])

        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=epochs, batch_size=32, verbose=0,
                  validation_split=0.2)

        self.lstm_model = model
        return model

    def forecast_volatility(self, lookback=60):
        """Forecast next period volatility using LSTM"""
        if self.lstm_model is None:
            print("Training LSTM model...")
            self.train_lstm_volatility(lookback)

        if self.lstm_model is None:
            # Fallback to historical volatility
            return self.returns.std().values * np.sqrt(252)

        volatility = self.returns.rolling(window=20).std() * np.sqrt(252)
        volatility = volatility.dropna()
        recent_vol = volatility.values[-lookback:]
        recent_vol = recent_vol.reshape((1, lookback, volatility.shape[1]))

        forecast = self.lstm_model.predict(recent_vol, verbose=0)
        return forecast[0]

    def lstm_black_litterman(self):
        """
        Use LSTM volatility forecasts as Black-Litterman views
        Lower forecasted volatility → positive expected return view
        """
        try:
            forecast_vol = self.forecast_volatility()
        except:
            print("⚠️  LSTM forecasting failed, using historical volatility")
            forecast_vol = self.returns.std().values * np.sqrt(252)

        historical_vol = self.returns.std() * np.sqrt(252)

        # Generate views based on volatility forecast
        P = np.eye(len(self.tickers))
        Q = np.zeros(len(self.tickers))

        for i, (forecast, historical) in enumerate(zip(forecast_vol, historical_vol)):
            if forecast < historical:
                Q[i] = 0.05  # Expect positive return if volatility decreasing
            else:
                Q[i] = -0.03  # Expect negative return if volatility increasing

        bl_returns, bl_cov = self.black_litterman(P, Q)
        return bl_returns, bl_cov

    def risk_parity(self):
        """
        Risk parity portfolio optimization
        Equal risk contribution from each asset

        Returns:
            Optimal weights with equal risk budgets
        """
        n_assets = len(self.tickers)

        # Initial guess: equal weights
        w0 = np.ones(n_assets) / n_assets

        # Objective: minimize difference in risk contributions
        def risk_budget_objective(weights):
            portfolio_vol = np.sqrt(weights @ self.cov_matrix @ weights)
            marginal_contrib = self.cov_matrix @ weights
            risk_contrib = weights * marginal_contrib / portfolio_vol
            target_risk = portfolio_vol / n_assets
            return np.sum((risk_contrib - target_risk) ** 2)

        from scipy.optimize import minimize

        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Fully invested
        ]
        bounds = tuple((0, 1) for _ in range(n_assets))  # Long-only

        result = minimize(
            risk_budget_objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        return pd.Series(result.x, index=self.tickers)

    def cvar_optimization(self, alpha=0.05, n_scenarios=1000):
        """
        CVaR (Conditional Value at Risk) optimization
        Minimizes expected loss in worst alpha% of cases

        Args:
            alpha: Confidence level (default 5% = 95% CVaR)
            n_scenarios: Number of return scenarios

        Returns:
            Optimal weights minimizing CVaR
        """
        n_assets = len(self.tickers)

        # Generate scenarios from historical returns
        scenarios = self.returns.sample(n_scenarios, replace=True).values

        w = cp.Variable(n_assets)
        z = cp.Variable(n_scenarios)
        v = cp.Variable()

        # CVaR formulation
        portfolio_returns = scenarios @ w
        objective = cp.Minimize(v + (1 / (alpha * n_scenarios)) * cp.sum(z))

        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            z >= 0,
            z >= -portfolio_returns - v
        ]

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS)

        if w.value is None:
            print("⚠️  CVaR optimization failed, using traditional MV")
            return self.optimize()

        return pd.Series(w.value, index=self.tickers)


# FastAPI Backend
try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from typing import List, Optional

    app = FastAPI(
        title="Portfolio Optimizer API",
        description="Advanced portfolio optimization with ML",
        version="1.0.0"
    )


    class OptimizationRequest(BaseModel):
        tickers: List[str]
        start_date: str
        end_date: str
        method: str = 'traditional'
        risk_aversion: float = 2.5
        views_P: Optional[List[List[float]]] = None
        views_Q: Optional[List[float]]] = None


    class OptimizationResponse(BaseModel):
        weights: dict
        expected_return: float
        volatility: float
        sharpe_ratio: float


    @app.post("/optimize", response_model=OptimizationResponse)
    async def optimize_portfolio(request: OptimizationRequest):
        """
        Optimize portfolio weights

        Methods: traditional, black_litterman, risk_parity, cvar, lstm_bl
        """
        try:
            optimizer = AdvancedPortfolioOptimizer(
                request.tickers,
                request.start_date,
                request.end_date,
                request.risk_aversion
            )

            if request.method == 'risk_parity':
                weights = optimizer.risk_parity()
            elif request.method == 'cvar':
                weights = optimizer.cvar_optimization()
            elif request.method == 'lstm_bl':
                bl_returns, _ = optimizer.lstm_black_litterman()
                weights = optimizer.optimize(expected_returns=bl_returns)
            elif request.method == 'black_litterman':
                if request.views_P and request.views_Q:
                    P = np.array(request.views_P)
                    Q = np.array(request.views_Q)
                    bl_returns, _ = optimizer.black_litterman(P, Q)
                    weights = optimizer.optimize(expected_returns=bl_returns)
                else:
                    raise HTTPException(400, "Black-Litterman requires views_P and views_Q")
            else:
                weights = optimizer.optimize()

            expected_return = float(optimizer.mean_returns @ weights.values)
            volatility = float(np.sqrt(weights.values @ optimizer.cov_matrix @ weights.values))
            sharpe_ratio = expected_return / volatility if volatility > 0 else 0

            return OptimizationResponse(
                weights=weights.to_dict(),
                expected_return=expected_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio
            )
        except Exception as e:
            raise HTTPException(500, f"Optimization failed: {str(e)}")


    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {"status": "healthy", "version": "1.0.0"}

except ImportError:
    print("⚠️  FastAPI not installed. API functionality unavailable.")
    print("   Install with: pip install fastapi uvicorn")

# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("ADVANCED PORTFOLIO OPTIMIZATION - EXAMPLE")
    print("=" * 70)

    tickers = ['SPY', 'AAPL', 'TSLA', 'MSFT']
    optimizer = AdvancedPortfolioOptimizer(tickers, '2021-01-01', '2024-12-31')

    print(f"\n📊 Assets: {', '.join(tickers)}")

    # Traditional
    print("\n1. Traditional Mean-Variance:")
    trad_weights = optimizer.optimize()
    print(trad_weights)

    # Risk Parity
    print("\n2. Risk Parity:")
    rp_weights = optimizer.risk_parity()
    print(rp_weights)

    # CVaR
    print("\n3. CVaR Optimization:")
    cvar_weights = optimizer.cvar_optimization()
    print(cvar_weights)

    print("\n✅ Advanced features demonstration complete!")
