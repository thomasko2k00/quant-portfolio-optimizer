"""
Portfolio Optimizer - Core Engine (Prompt 1)
Mean-Variance Optimization with Ledoit-Wolf Shrinkage and Black-Litterman

Evidence:
- Ledoit & Wolf (2003): Shrinkage reduces estimation error by 50%+
- Black & Litterman (1992): Bayesian approach stabilizes extreme optimizer inputs
"""

import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf
import yfinance as yf
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class PortfolioOptimizer:
    """
    Mean-Variance Portfolio Optimizer with Ledoit-Wolf Shrinkage and Black-Litterman

    Key Features:
    - Ledoit-Wolf shrinkage for covariance estimation (reduces error by 50%+)
    - Black-Litterman model for incorporating investor views
    - Efficient frontier generation
    - Risk-adjusted utility maximization
    """

    def __init__(self, tickers, start_date=None, end_date=None, risk_aversion=2.5):
        """
        Initialize portfolio optimizer

        Args:
            tickers: List of asset tickers
            start_date: Start date for historical data
            end_date: End date for historical data
            risk_aversion: Risk aversion parameter (lambda)
        """
        self.tickers = tickers
        self.risk_aversion = risk_aversion
        self.start_date = start_date or (datetime.now() - timedelta(days=252 * 3)).strftime('%Y-%m-%d')
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')

        self.prices = self._download_data()
        self.returns = self.prices.pct_change().dropna()
        self.cov_matrix = self._ledoit_wolf_cov()
        self.mean_returns = self.returns.mean() * 252

    def _download_data(self):
        """Download historical data from Yahoo Finance"""
        import pandas as pd

        print(f"Downloading data for {self.tickers}...")
        data = yf.download(self.tickers, start=self.start_date, end=self.end_date, progress=False)

        # Extract Close prices (which are already adjusted in newer yfinance)
        if isinstance(data.columns, pd.MultiIndex):
            # MultiIndex case (multiple tickers or yfinance structure)
            if 'Close' in data.columns.get_level_values(0):
                prices = data['Close']
            elif 'Adj Close' in data.columns.get_level_values(0):
                prices = data['Adj Close']
            else:
                raise ValueError("Cannot find 'Close' or 'Adj Close' in data")
        else:
            # Simple case (single ticker, older yfinance)
            if 'Close' in data.columns:
                prices = data[['Close']].copy()
                prices.columns = self.tickers
            elif 'Adj Close' in data.columns:
                prices = data[['Adj Close']].copy()
                prices.columns = self.tickers
            else:
                prices = data

        # Validate data
        if prices.empty:
            raise ValueError(f"No data downloaded for {self.tickers}. Check ticker symbols and date range.")

        # Remove NaN values
        prices = prices.dropna()

        if prices.empty:
            raise ValueError("All data contains NaN values. Check ticker symbols.")

        print(f"✅ Successfully downloaded {len(prices)} rows for {list(prices.columns)}")
        return prices

    def _ledoit_wolf_cov(self):
        """Apply Ledoit-Wolf shrinkage to covariance matrix"""
        lw = LedoitWolf()
        lw.fit(self.returns)
        return lw.covariance_ * 252  # Annualize

    def black_litterman(self, P, Q, omega=None, tau=0.025, market_caps=None):
        """
        Black-Litterman model for combining market equilibrium with investor views

        Args:
            P: View matrix (K x N) where K = number of views, N = number of assets
            Q: View returns (K x 1) - expected returns for each view
            omega: Confidence matrix (K x K) - uncertainty in views
            tau: Scalar representing uncertainty in prior
            market_caps: Market capitalizations (optional, for equilibrium)

        Returns:
            BL expected returns (N x 1), BL covariance matrix
        """
        n_assets = len(self.tickers)

        if market_caps is None:
            market_weights = np.ones(n_assets) / n_assets
        else:
            market_weights = np.array(market_caps) / np.sum(market_caps)

        pi = self.risk_aversion * self.cov_matrix @ market_weights

        if omega is None:
            omega = np.diag(np.diag(P @ (tau * self.cov_matrix) @ P.T))

        tau_sigma = tau * self.cov_matrix
        M_inv = np.linalg.inv(np.linalg.inv(tau_sigma) + P.T @ np.linalg.inv(omega) @ P)
        bl_returns = M_inv @ (np.linalg.inv(tau_sigma) @ pi + P.T @ np.linalg.inv(omega) @ Q)

        return bl_returns, M_inv + self.cov_matrix

    def optimize(self, expected_returns=None, target_return=None, constraints=None):
        """
        Optimize portfolio weights to maximize utility

        Args:
            expected_returns: Custom expected returns (if None, uses historical mean)
            target_return: Target return constraint (optional)
            constraints: Additional constraints dict

        Returns:
            Optimal weights as pandas Series
        """
        n_assets = len(self.tickers)

        if expected_returns is None:
            expected_returns = self.mean_returns.values

        w = cp.Variable(n_assets)
        portfolio_return = expected_returns @ w
        portfolio_variance = cp.quad_form(w, self.cov_matrix)
        utility = portfolio_return - (self.risk_aversion / 2) * portfolio_variance

        cons = [cp.sum(w) == 1, w >= 0]

        if target_return is not None:
            cons.append(portfolio_return >= target_return)

        if constraints:
            if 'max_weight' in constraints:
                cons.append(w <= constraints['max_weight'])
            if 'min_weight' in constraints:
                cons.append(w >= constraints['min_weight'])

        problem = cp.Problem(cp.Maximize(utility), cons)
        problem.solve(solver=cp.ECOS)

        if w.value is None:
            raise ValueError("Optimization failed")

        return pd.Series(w.value, index=self.tickers)

    def efficient_frontier(self, n_points=50, expected_returns=None):
        """
        Generate efficient frontier

        Args:
            n_points: Number of portfolios to compute
            expected_returns: Custom expected returns

        Returns:
            DataFrame with returns, volatilities, and weights
        """
        if expected_returns is None:
            expected_returns = self.mean_returns.values

        n_assets = len(self.tickers)
        w = cp.Variable(n_assets)

        problem = cp.Problem(
            cp.Minimize(cp.quad_form(w, self.cov_matrix)),
            [cp.sum(w) == 1, w >= 0]
        )
        problem.solve(solver=cp.ECOS)
        min_return = expected_returns @ w.value
        max_return = np.max(expected_returns)

        target_returns = np.linspace(min_return, max_return * 0.95, n_points)
        frontier_weights = []
        frontier_vols = []
        frontier_rets = []

        for target in target_returns:
            try:
                w = cp.Variable(n_assets)
                objective = cp.Minimize(cp.quad_form(w, self.cov_matrix))
                constraints = [cp.sum(w) == 1, w >= 0, expected_returns @ w >= target]
                problem = cp.Problem(objective, constraints)
                problem.solve(solver=cp.ECOS)

                if w.value is not None:
                    weights = w.value
                    ret = expected_returns @ weights
                    vol = np.sqrt(weights @ self.cov_matrix @ weights)
                    frontier_weights.append(weights)
                    frontier_rets.append(ret)
                    frontier_vols.append(vol)
            except:
                continue

        return pd.DataFrame({
            'Return': frontier_rets,
            'Volatility': frontier_vols,
            'Weights': [dict(zip(self.tickers, w)) for w in frontier_weights]
        })

    def plot_efficient_frontier(self, frontier_df=None, optimal_weights=None):
        """Plot efficient frontier with optimal portfolio"""
        if frontier_df is None:
            frontier_df = self.efficient_frontier()

        plt.figure(figsize=(12, 7))
        plt.plot(frontier_df['Volatility'], frontier_df['Return'],
                 'b-', linewidth=2, label='Efficient Frontier')

        asset_vols = np.sqrt(np.diag(self.cov_matrix))
        plt.scatter(asset_vols, self.mean_returns,
                    marker='o', s=100, c='red', label='Individual Assets')

        for i, ticker in enumerate(self.tickers):
            plt.annotate(ticker, (asset_vols[i], self.mean_returns[i]))

        if optimal_weights is not None:
            opt_ret = self.mean_returns @ optimal_weights
            opt_vol = np.sqrt(optimal_weights @ self.cov_matrix @ optimal_weights)
            plt.scatter(opt_vol, opt_ret, marker='*', s=500,
                        c='gold', edgecolors='black', label='Optimal Portfolio')

        plt.xlabel('Volatility (Annual)', fontsize=12)
        plt.ylabel('Expected Return (Annual)', fontsize=12)
        plt.title('Efficient Frontier - Mean-Variance Optimization',
                  fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('efficient_frontier.png', dpi=300, bbox_inches='tight')
        plt.close()
        return 'efficient_frontier.png'


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("PORTFOLIO OPTIMIZER - EXAMPLE")
    print("=" * 70)

    tickers = ['SPY', 'AAPL', 'TSLA']
    optimizer = PortfolioOptimizer(tickers, start_date='2021-01-01',
                                   end_date='2024-12-31')

    print(f"\n📊 Assets: {', '.join(tickers)}")
    print(f"\nAnnualized Mean Returns:")
    print(optimizer.mean_returns)

    print("\n" + "=" * 70)
    print("TRADITIONAL MEAN-VARIANCE OPTIMIZATION")
    print("=" * 70)
    traditional_weights = optimizer.optimize()
    print("\nOptimal Weights:")
    for ticker, weight in traditional_weights.items():
        print(f"  {ticker}: {weight:.2%}")

    print("\n" + "=" * 70)
    print("BLACK-LITTERMAN WITH INVESTOR VIEWS")
    print("=" * 70)
    P = np.array([[-1, 1, 0], [0, 0, 1]])
    Q = np.array([0.05, 0.10])
    print("\nViews:")
    print("  View 1: AAPL outperforms SPY by 5%")
    print("  View 2: TSLA expected return of 10%")

    bl_returns, _ = optimizer.black_litterman(P, Q)
    bl_weights = optimizer.optimize(expected_returns=bl_returns)
    print("\nBL Optimal Weights:")
    for ticker, weight in bl_weights.items():
        print(f"  {ticker}: {weight:.2%}")

    print("\n" + "=" * 70)
    print("GENERATING EFFICIENT FRONTIER")
    print("=" * 70)
    frontier = optimizer.efficient_frontier()
    optimizer.plot_efficient_frontier(frontier, bl_weights.values)
    print(f"\n✅ Generated {len(frontier)} efficient portfolios")
    print("✅ Plot saved: efficient_frontier.png")
