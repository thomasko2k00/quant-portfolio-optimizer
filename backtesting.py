"""
Portfolio Backtesting Module (Prompt 2)
Walk-Forward Optimization with Rolling Windows

Evidence:
- Walk-forward optimization prevents overfitting vs static backtests
- Transaction costs significantly impact real-world performance
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from portfolio_optimizer import PortfolioOptimizer
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class PortfolioBacktester:
    """
    Backtest portfolio strategies with walk-forward optimization
    """

    def __init__(self, optimizer, train_window=252, test_window=21,
                 rebalance_freq='monthly', transaction_cost=0.0005):
        """
        Initialize backtester

        Args:
            optimizer: PortfolioOptimizer instance
            train_window: Training window size in days (default 252 = 1 year)
            test_window: Test window size in days (default 21 = 1 month)
            rebalance_freq: Rebalancing frequency
            transaction_cost: Transaction cost in basis points (default 5bps)
        """
        self.optimizer = optimizer
        self.train_window = train_window
        self.test_window = test_window
        self.rebalance_freq = rebalance_freq
        self.transaction_cost = transaction_cost

    def backtest(self, start_date='2015-01-01', end_date='2025-01-01',
                 method='traditional', P=None, Q=None):
        """
        Walk-forward backtesting with rolling windows

        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            method: 'traditional' or 'black_litterman'
            P: View matrix for Black-Litterman
            Q: View returns for Black-Litterman

        Returns:
            Dictionary with performance metrics
        """
        # Download fresh data for backtesting period
        temp_opt = PortfolioOptimizer(
            self.optimizer.tickers,
            start_date=start_date,
            end_date=end_date
        )
        prices = temp_opt.prices
        returns = temp_opt.returns

        portfolio_values = [100000]
        portfolio_weights_history = []
        rebalance_dates = []
        turnover_history = []

        prev_weights = None

        # Walk-forward optimization
        for i in range(self.train_window, len(returns) - self.test_window, self.test_window):
            # Training window
            train_start = returns.index[i - self.train_window]
            train_end = returns.index[i - 1]

            # Create optimizer for this training window
            window_opt = PortfolioOptimizer(
                self.optimizer.tickers,
                start_date=train_start.strftime('%Y-%m-%d'),
                end_date=train_end.strftime('%Y-%m-%d')
            )

            # Optimize
            if method == 'black_litterman' and P is not None and Q is not None:
                bl_returns, _ = window_opt.black_litterman(P, Q)
                weights = window_opt.optimize(expected_returns=bl_returns).values
            else:
                weights = window_opt.optimize().values

            # Calculate turnover
            if prev_weights is not None:
                turnover = np.sum(np.abs(weights - prev_weights))
                cost = turnover * self.transaction_cost
                turnover_history.append(turnover)
            else:
                cost = 0

            # Test window - apply weights
            test_returns = returns.iloc[i:i + self.test_window]

            for j, daily_return in enumerate(test_returns.values):
                portfolio_return = np.sum(weights * daily_return)
                if j == 0:  # Apply transaction cost only on rebalance
                    portfolio_return -= cost
                portfolio_values.append(portfolio_values[-1] * (1 + portfolio_return))

            portfolio_weights_history.append(weights)
            rebalance_dates.append(test_returns.index[0])
            prev_weights = weights

        # Calculate performance metrics
        portfolio_returns = pd.Series(portfolio_values).pct_change().dropna()

        sharpe_ratio = (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(252)
        max_drawdown = self._calculate_max_drawdown(portfolio_values)
        avg_turnover = np.mean(turnover_history) if turnover_history else 0
        cumulative_return = (portfolio_values[-1] / portfolio_values[0]) - 1

        results = {
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'cumulative_return': cumulative_return,
            'avg_turnover': avg_turnover,
            'portfolio_values': portfolio_values,
            'rebalance_dates': rebalance_dates,
            'weights_history': portfolio_weights_history
        }

        return results

    def _calculate_max_drawdown(self, portfolio_values):
        """Calculate maximum drawdown"""
        cumulative = pd.Series(portfolio_values)
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def plot_backtest_results(self, results, benchmark_ticker='SPY'):
        """
        Create interactive Plotly visualizations

        Args:
            results: Backtest results dictionary
            benchmark_ticker: Benchmark ticker for comparison

        Returns:
            Tuple of (equity_curve_file, drawdown_file)
        """
        # Equity curve
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            y=results['portfolio_values'],
            mode='lines',
            name='Strategy',
            line=dict(color='blue', width=2)
        ))

        fig.update_layout(
            title='Portfolio Backtest - Cumulative Returns',
            xaxis_title='Trading Days',
            yaxis_title='Portfolio Value ($)',
            template='plotly_white',
            hovermode='x unified',
            height=600
        )

        fig.write_html('backtest_results.html')

        # Drawdown chart
        portfolio_series = pd.Series(results['portfolio_values'])
        running_max = portfolio_series.expanding().max()
        drawdown = (portfolio_series - running_max) / running_max * 100

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            y=drawdown,
            mode='lines',
            fill='tozeroy',
            name='Drawdown',
            line=dict(color='red')
        ))

        fig2.update_layout(
            title='Portfolio Drawdown Over Time',
            xaxis_title='Trading Days',
            yaxis_title='Drawdown (%)',
            template='plotly_white',
            height=500
        )

        fig2.write_html('drawdown.html')

        return 'backtest_results.html', 'drawdown.html'

    def generate_summary_table(self, results):
        """Generate performance metrics summary table"""
        summary = pd.DataFrame({
            'Metric': [
                'Sharpe Ratio',
                'Max Drawdown',
                'Cumulative Return',
                'Avg Monthly Turnover'
            ],
            'Value': [
                f"{results['sharpe_ratio']:.2f}",
                f"{results['max_drawdown'] * 100:.2f}%",
                f"{results['cumulative_return'] * 100:.2f}%",
                f"{results['avg_turnover']:.2%}"
            ]
        })
        return summary


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("PORTFOLIO BACKTESTING - EXAMPLE")
    print("=" * 70)

    tickers = ['SPY', 'AAPL', 'TSLA']
    optimizer = PortfolioOptimizer(tickers)
    backtester = PortfolioBacktester(
        optimizer,
        train_window=252,
        test_window=21,
        transaction_cost=0.0005
    )

    print(f"\n📊 Backtesting {', '.join(tickers)}")
    print(f"📅 Period: 2020-01-01 to 2024-12-31")
    print(f"🔄 Rebalancing: Monthly (252-day train, 21-day test)")
    print(f"💰 Transaction cost: 5 bps")

    print("\n⏳ Running walk-forward backtest...")
    results = backtester.backtest(
        start_date='2020-01-01',
        end_date='2024-12-31',
        method='traditional'
    )

    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)
    summary = backtester.generate_summary_table(results)
    print(summary.to_string(index=False))

    print("\n📊 Generating visualizations...")
    equity_file, dd_file = backtester.plot_backtest_results(results)
    print(f"✅ Equity curve saved: {equity_file}")
    print(f"✅ Drawdown chart saved: {dd_file}")

    print("\n✅ Backtesting completed!")
