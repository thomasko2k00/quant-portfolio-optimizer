"""
FastAPI Application for Portfolio Optimizer
REST API endpoints for portfolio optimization
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from advanced_features import AdvancedPortfolioOptimizer
from backtesting import PortfolioBacktester

app = FastAPI(
    title="Quantitative Portfolio Optimizer API",
    description="Advanced portfolio optimization with Ledoit-Wolf, Black-Litterman, LSTM, and more",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class OptimizationRequest(BaseModel):
    tickers: List[str]
    start_date: str
    end_date: str
    method: str = 'traditional'
    risk_aversion: float = 2.5
    views_P: Optional[List[List[float]]] = None
    views_Q: Optional[List[float]] = None


class OptimizationResponse(BaseModel):
    weights: dict
    expected_return: float
    volatility: float
    sharpe_ratio: float
    method_used: str


class BacktestRequest(BaseModel):
    tickers: List[str]
    start_date: str
    end_date: str
    train_window: int = 252
    test_window: int = 21
    transaction_cost: float = 0.0005
    method: str = 'traditional'


class BacktestResponse(BaseModel):
    sharpe_ratio: float
    max_drawdown: float
    cumulative_return: float
    avg_turnover: float


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Portfolio Optimizer API",
        "version": "1.0.0",
        "endpoints": {
            "optimize": "/optimize",
            "backtest": "/backtest",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.post("/optimize", response_model=OptimizationResponse)
async def optimize_portfolio(request: OptimizationRequest):
    """
    Optimize portfolio weights using various methods

    Methods available:
    - traditional: Mean-variance optimization
    - black_litterman: With investor views
    - risk_parity: Equal risk contribution
    - cvar: Conditional Value at Risk
    - lstm_bl: LSTM volatility forecasts as views
    """
    try:
        optimizer = AdvancedPortfolioOptimizer(
            request.tickers,
            request.start_date,
            request.end_date,
            request.risk_aversion
        )

        # Select optimization method
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
        else:  # traditional
            weights = optimizer.optimize()

        # Calculate metrics
        expected_return = float(optimizer.mean_returns @ weights.values)
        volatility = float(np.sqrt(weights.values @ optimizer.cov_matrix @ weights.values))
        sharpe_ratio = expected_return / volatility if volatility > 0 else 0

        return OptimizationResponse(
            weights=weights.to_dict(),
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            method_used=request.method
        )

    except Exception as e:
        raise HTTPException(500, f"Optimization failed: {str(e)}")


@app.post("/backtest", response_model=BacktestResponse)
async def backtest_portfolio(request: BacktestRequest):
    """
    Backtest portfolio strategy with walk-forward optimization
    """
    try:
        from portfolio_optimizer import PortfolioOptimizer

        optimizer = PortfolioOptimizer(request.tickers)
        backtester = PortfolioBacktester(
            optimizer,
            train_window=request.train_window,
            test_window=request.test_window,
            transaction_cost=request.transaction_cost
        )

        results = backtester.backtest(
            start_date=request.start_date,
            end_date=request.end_date,
            method=request.method
        )

        return BacktestResponse(
            sharpe_ratio=results['sharpe_ratio'],
            max_drawdown=results['max_drawdown'],
            cumulative_return=results['cumulative_return'],
            avg_turnover=results['avg_turnover']
        )

    except Exception as e:
        raise HTTPException(500, f"Backtesting failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "service": "portfolio-optimizer"
    }


if __name__ == "__main__":
    import uvicorn

    print("🚀 Starting Portfolio Optimizer API...")
    print("📖 API Documentation: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
