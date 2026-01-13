"""
Validation Tests (Prompt 4)
Compare shrinkage vs unshrunk, BL vs traditional
"""

import pytest
import numpy as np
import sys

sys.path.insert(0, '..')

from portfolio_optimizer import PortfolioOptimizer
from advanced_features import AdvancedPortfolioOptimizer
from backtesting import PortfolioBacktester


def test_optimizer_initialization():
    """Test that optimizer initializes correctly"""
    tickers = ['SPY', 'AAPL', 'TSLA']
    optimizer = PortfolioOptimizer(tickers, '2021-01-01', '2023-12-31')

    assert optimizer.tickers == tickers
    assert optimizer.prices is not None
    assert optimizer.returns is not None
    assert optimizer.cov_matrix is not None
    print("✅ Optimizer initialization test passed")


def test_traditional_optimization():
    """Test traditional mean-variance optimization"""
    tickers = ['SPY', 'AAPL']
    optimizer = PortfolioOptimizer(tickers, '2022-01-01', '2023-12-31')

    weights = optimizer.optimize()

    assert len(weights) == len(tickers)
    assert abs(weights.sum() - 1.0) < 0.01  # Weights sum to 1
    assert all(weights >= -0.01)  # Long-only (small tolerance)
    print(f"✅ Traditional optimization test passed: {weights.to_dict()}")


def test_black_litterman():
    """Test Black-Litterman optimization"""
    tickers = ['SPY', 'AAPL', 'TSLA']
    optimizer = PortfolioOptimizer(tickers, '2021-01-01', '2023-12-31')

    P = np.array([[-1, 1, 0], [0, 0, 1]])
    Q = np.array([0.05, 0.10])

    bl_returns, bl_cov = optimizer.black_litterman(P, Q)

    assert len(bl_returns) == len(tickers)
    assert bl_cov.shape == (len(tickers), len(tickers))
    print("✅ Black-Litterman test passed")


def test_risk_parity():
    """Test risk parity optimization"""
    tickers = ['SPY', 'QQQ', 'AGG']
    optimizer = AdvancedPortfolioOptimizer(tickers, '2022-01-01', '2023-12-31')

    rp_weights = optimizer.risk_parity()

    assert len(rp_weights) == len(tickers)
    assert abs(rp_weights.sum() - 1.0) < 0.01
    print(f"✅ Risk parity test passed: {rp_weights.to_dict()}")


def test_cvar_optimization():
    """Test CVaR optimization"""
    tickers = ['SPY', 'AAPL']
    optimizer = AdvancedPortfolioOptimizer(tickers, '2022-01-01', '2023-12-31')

    cvar_weights = optimizer.cvar_optimization()

    assert len(cvar_weights) == len(tickers)
    assert abs(cvar_weights.sum() - 1.0) < 0.01
    print(f"✅ CVaR optimization test passed: {cvar_weights.to_dict()}")


def test_backtesting():
    """Test backtesting module"""
    tickers = ['SPY', 'AAPL']
    optimizer = PortfolioOptimizer(tickers)
    backtester = PortfolioBacktester(optimizer, train_window=252, test_window=21)

    results = backtester.backtest('2022-01-01', '2023-12-31')

    assert 'sharpe_ratio' in results
    assert 'max_drawdown' in results
    assert 'cumulative_return' in results
    assert results['max_drawdown'] <= 0  # Drawdown is negative

    print(f"✅ Backtesting test passed:")
    print(f"   Sharpe: {results['sharpe_ratio']:.2f}")
    print(f"   Max DD: {results['max_drawdown'] * 100:.2f}%")
    print(f"   Return: {results['cumulative_return'] * 100:.2f}%")


def test_efficient_frontier():
    """Test efficient frontier generation"""
    tickers = ['SPY', 'AAPL']
    optimizer = PortfolioOptimizer(tickers, '2022-01-01', '2023-12-31')

    frontier = optimizer.efficient_frontier(n_points=20)

    assert len(frontier) > 0
    assert 'Return' in frontier.columns
    assert 'Volatility' in frontier.columns
    assert 'Weights' in frontier.columns
    print(f"✅ Efficient frontier test passed: {len(frontier)} points generated")


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING VALIDATION TESTS")
    print("=" * 70)

    test_optimizer_initialization()
    test_traditional_optimization()
    test_black_litterman()
    test_risk_parity()
    test_cvar_optimization()
    test_efficient_frontier()
    test_backtesting()

    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)
