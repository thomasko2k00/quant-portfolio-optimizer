"""
Streamlit Dashboard for Portfolio Optimizer
Interactive web interface for portfolio optimization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from portfolio_optimizer import PortfolioOptimizer as AdvancedPortfolioOptimizer
from backtesting import PortfolioBacktester
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Portfolio Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🚀 Quantitative Portfolio Optimizer</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced Mean-Variance Optimization with ML & Black-Litterman</p>',
            unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Asset selection
st.sidebar.subheader("1. Asset Selection")
preset = st.sidebar.selectbox(
    "Preset Portfolios",
    ["Custom", "Tech Giants", "Diversified", "Index Funds"]
)

if preset == "Tech Giants":
    default_tickers = "AAPL,MSFT,GOOGL,NVDA,TSLA"
elif preset == "Diversified":
    default_tickers = "SPY,QQQ,IWM,EFA,AGG,GLD"
elif preset == "Index Funds":
    default_tickers = "SPY,QQQ,DIA,IWM,EFA"
else:
    default_tickers = "SPY,AAPL,TSLA,MSFT"

tickers_input = st.sidebar.text_input("Tickers (comma-separated)", default_tickers)
tickers = [t.strip().upper() for t in tickers_input.split(',')]

# Date range
st.sidebar.subheader("2. Date Range")
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "Start Date",
        datetime.now() - timedelta(days=3 * 365)
    )
with col2:
    end_date = st.date_input(
        "End Date",
        datetime.now()
    )

# Optimization method
st.sidebar.subheader("3. Optimization Method")
method = st.sidebar.selectbox(
    "Method",
    ["Traditional MV", "Black-Litterman", "Risk Parity", "CVaR", "LSTM-BL"]
)

risk_aversion = st.sidebar.slider(
    "Risk Aversion (λ)",
    0.5, 5.0, 2.5, 0.1,
    help="Higher values = more conservative"
)

# Black-Litterman views (if selected)
views_P = None
views_Q = None
if method == "Black-Litterman":
    st.sidebar.subheader("4. Investor Views")
    st.sidebar.markdown("*Example: AAPL will outperform SPY by 5%*")

    use_default_views = st.sidebar.checkbox("Use example views", value=True)
    if use_default_views and len(tickers) >= 2:
        views_P = np.array([[-1, 1] + [0] * (len(tickers) - 2)])
        views_Q = np.array([0.05])

# Run optimization button
run_optimization = st.sidebar.button("🚀 Run Optimization", type="primary")

# Main content
if run_optimization:
    try:
        with st.spinner("Downloading data and optimizing..."):
            # Create optimizer
            optimizer = AdvancedPortfolioOptimizer(
                tickers,
                str(start_date),
                str(end_date),
                risk_aversion
            )

            # Optimize based on method
            if method == "Risk Parity":
                weights = optimizer.risk_parity()
                method_desc = "Risk Parity (Equal Risk Contribution)"
            elif method == "CVaR":
                weights = optimizer.cvar_optimization()
                method_desc = "CVaR (Conditional Value at Risk)"
            elif method == "LSTM-BL":
                bl_returns, _ = optimizer.lstm_black_litterman()
                weights = optimizer.optimize(expected_returns=bl_returns)
                method_desc = "LSTM-Enhanced Black-Litterman"
            elif method == "Black-Litterman" and views_P is not None:
                bl_returns, _ = optimizer.black_litterman(views_P, views_Q)
                weights = optimizer.optimize(expected_returns=bl_returns)
                method_desc = "Black-Litterman with Views"
            else:
                weights = optimizer.optimize()
                method_desc = "Traditional Mean-Variance"

            # Calculate metrics
            portfolio_return = optimizer.mean_returns @ weights.values
            portfolio_vol = np.sqrt(weights.values @ optimizer.cov_matrix @ weights.values)
            sharpe_ratio = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0

        # Display results
        st.success("✅ Optimization Complete!")

        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Expected Return", f"{portfolio_return:.2%}")
        with col2:
            st.metric("Volatility", f"{portfolio_vol:.2%}")
        with col3:
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
        with col4:
            st.metric("Assets", len(tickers))

        st.markdown("---")

        # Two columns for visualization
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.subheader("📊 Optimal Portfolio Weights")
            st.caption(f"Method: {method_desc}")

            # Pie chart
            fig_pie = go.Figure(data=[go.Pie(
                labels=weights.index,
                values=weights.values,
                hole=0.4,
                textinfo='label+percent',
                marker=dict(colors=px.colors.qualitative.Set3)
            )])
            fig_pie.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig_pie, use_container_width=True)

            # Weights table
            weights_df = pd.DataFrame({
                'Asset': weights.index,
                'Weight': weights.values,
                'Weight %': [f"{w:.2%}" for w in weights.values]
            }).sort_values('Weight', ascending=False)
            st.dataframe(weights_df, hide_index=True, use_container_width=True)

        with col_right:
            st.subheader("📈 Efficient Frontier")
            st.caption("Optimal portfolio marked with ⭐")

            # Generate efficient frontier
            frontier = optimizer.efficient_frontier(n_points=30)

            fig_frontier = go.Figure()

            # Efficient frontier line
            fig_frontier.add_trace(go.Scatter(
                x=frontier['Volatility'],
                y=frontier['Return'],
                mode='lines',
                name='Efficient Frontier',
                line=dict(color='blue', width=3)
            ))

            # Individual assets
            asset_vols = np.sqrt(np.diag(optimizer.cov_matrix))
            fig_frontier.add_trace(go.Scatter(
                x=asset_vols,
                y=optimizer.mean_returns,
                mode='markers+text',
                name='Individual Assets',
                text=optimizer.mean_returns.index,
                textposition='top center',
                marker=dict(size=12, color='red')
            ))

            # Optimal portfolio
            fig_frontier.add_trace(go.Scatter(
                x=[portfolio_vol],
                y=[portfolio_return],
                mode='markers',
                name='Optimal Portfolio',
                marker=dict(size=20, color='gold', symbol='star',
                            line=dict(color='black', width=2))
            ))

            fig_frontier.update_layout(
                xaxis_title='Volatility (Annual)',
                yaxis_title='Expected Return (Annual)',
                height=400,
                hovermode='closest',
                showlegend=True
            )
            st.plotly_chart(fig_frontier, use_container_width=True)

        # Additional analysis
        st.markdown("---")
        st.subheader("📉 Historical Performance Analysis")

        # Plot historical prices
        fig_prices = go.Figure()
        for ticker in tickers:
            normalized = optimizer.prices[ticker] / optimizer.prices[ticker].iloc[0] * 100
            fig_prices.add_trace(go.Scatter(
                x=optimizer.prices.index,
                y=normalized,
                mode='lines',
                name=ticker
            ))

        fig_prices.update_layout(
            title="Normalized Price Performance (Base = 100)",
            xaxis_title="Date",
            yaxis_title="Normalized Price",
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig_prices, use_container_width=True)

        # Correlation matrix
        st.subheader("🔗 Asset Correlation Matrix")
        corr_matrix = optimizer.returns.corr()

        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        fig_corr.update_layout(height=400)
        st.plotly_chart(fig_corr, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.error("Please check your ticker symbols and date range.")

else:
    # Welcome message
    st.info("👈 Configure your portfolio in the sidebar and click '🚀 Run Optimization' to start!")

    st.markdown("""
    ### 🎯 Features

    - **Ledoit-Wolf Shrinkage**: Reduces covariance estimation error by 50%+
    - **Black-Litterman**: Incorporate investor views with confidence levels
    - **Risk Parity**: Equal risk contribution from each asset
    - **CVaR Optimization**: Minimize tail risk
    - **LSTM Forecasting**: ML-enhanced volatility predictions
    - **Interactive Visualizations**: Efficient frontier, correlations, and more

    ### 📚 How to Use

    1. Select assets or use a preset portfolio
    2. Choose date range for historical data
    3. Select optimization method
    4. Adjust risk aversion (higher = more conservative)
    5. Click 'Run Optimization'

    ### 📊 Optimization Methods

    - **Traditional MV**: Classic mean-variance optimization
    - **Black-Litterman**: Combine market equilibrium with your views
    - **Risk Parity**: Equal risk contribution (not return)
    - **CVaR**: Minimize losses in worst 5% scenarios
    - **LSTM-BL**: Use ML volatility forecasts as views
    """)

# Footer
st.markdown("---")
st.markdown("*Built with Ledoit-Wolf shrinkage, Black-Litterman model, and CVXPY optimization*")

# Import plotly.express for colors
import plotly.express as px
