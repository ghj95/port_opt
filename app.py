#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import cvxpy as cp

# Load asset data
st.title("📊 Portfolio Optimization & Risk Analysis")

st.sidebar.header("Select Assets & Parameters")
assets = st.sidebar.multiselect("Select Stocks:", ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NFLX"], ["AAPL", "MSFT", "GOOGL"])
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2019-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))

if st.sidebar.button("Fetch Data"):
    st.sidebar.success("Fetching stock data...")

    # Fetch stock data
    data = yf.download(assets, start=start_date, end=end_date)['Close']
    st.write("### Stock Price Data")
    st.line_chart(data)

    # Compute returns
    returns = data.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    # Portfolio optimization
    n_assets = len(assets)
    w = cp.Variable(n_assets)
    expected_return = mean_returns.values @ w
    portfolio_risk = cp.quad_form(w, cov_matrix.values)
    objective = cp.Maximize(expected_return - 0.5 * portfolio_risk)
    constraints = [cp.sum(w) == 1, w >= 0]
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Display optimal weights
    optimal_weights = pd.Series(w.value, index=assets)
    st.write("### Optimal Portfolio Weights")
    st.bar_chart(optimal_weights)

    # Compute risk metrics
    portfolio_returns = returns @ optimal_weights
    VaR_95 = np.percentile(portfolio_returns, 5)
    CVaR_95 = portfolio_returns[portfolio_returns <= VaR_95].mean()

    st.write(f"📉 **Value at Risk (VaR 95%):** {VaR_95:.4f}")
    st.write(f"⚠️ **Conditional Value at Risk (CVaR 95%):** {CVaR_95:.4f}")

    # Plot Efficient Frontier
    n_portfolios = 1000
    results = np.zeros((3, n_portfolios))
    for i in range(n_portfolios):
        weights = np.random.dirichlet(np.ones(n_assets))
        port_return = np.dot(mean_returns, weights)
        port_volatility = np.sqrt(weights @ cov_matrix @ weights)
        results[0, i] = port_return
        results[1, i] = port_volatility
        results[2, i] = port_return / port_volatility

    # Extract optimal portfolio
    sharpe_max_idx = results[2].argmax()
    sharpe_max_return = results[0, sharpe_max_idx]
    sharpe_max_volatility = results[1, sharpe_max_idx]

    fig, ax = plt.subplots()
    ax.scatter(results[1, :], results[0, :], c=results[2, :], cmap='viridis', alpha=0.3)
    ax.scatter(sharpe_max_volatility, sharpe_max_return, color='red', marker='*', s=200, label="Max Sharpe Ratio")
    ax.set_xlabel("Risk (Volatility)")
    ax.set_ylabel("Expected Return")
    ax.set_title("Efficient Frontier")
    ax.legend()
    st.pyplot(fig)


# In[ ]:




