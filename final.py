# import relevant libraries

from datetime import date
import yfinance as yf
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# load asset data
# streamlit code included for web app interface

# designing menus for user input

st.title("Porfolio Optimizer")

st.sidebar.header("Select assets and parameters")
assets = st.sidebar.multiselect("Select stocks:", ["RY", "TD", "BMO", "BNS", "CM"], ["RY", "TD", "BMO", "BNS", "CM"])
start_date = st.sidebar.date_input("Start date", "2019-01-01")
end_date = st.sidebar.date_input("End date", date.today())

if st.sidebar.button("Fetch data"):
    st.sidebar.success("Data uploaded")

    # fetch stock data
    data = yf.download(assets, start=start_date, end=end_date)['Close']
    st.write("### Stock price")
    st.line_chart(data)

    # compute returns
    returns = data.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    # optimize portfolio
    n_assets = len(assets)
    w = cp.Variable(n_assets)
    expected_return = mean_returns.values @ w
    portfolio_risk = cp.quad_form(w, cov_matrix.values)
    objective = cp.Maximize(expected_return - 0.5 * portfolio_risk)
    constraints = [cp.sum(w) == 1, w >= 0]
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # display optimal weights
    optimal_weights = pd.Series(w.value, index=assets)
    st.write("### Optimal portfolio weights")
    st.bar_chart(optimal_weights)

    # simulation parameters
    n_sim = 10000
    time_horizon = 1

    # simulate and compute random porfolio
    sim_returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_sim)
    port_sim_returns = sim_returns @ optimal_weights

    # compute mc var and cvar
    VaR_95_mc = np.percentile(port_sim_returns, 5)
    CVaR_95_mc = port_sim_returns[port_sim_returns <= VaR_95_mc].mean()

    print(f"Monte Carlo VaR 95%: {VaR_95_mc:.4f}")
    print(f"Monte Carlo VaR 95%: {CVaR_95_mc:.4f}")

    # create histogram
    fig = go.Figure()
    hist_counts, bin_edges = np.histogram(port_sim_returns, bins=50, density=True)
    y_max = max(hist_counts)

    # add histogram of simulated returns
    fig.add_trace(go.Histogram(x=port_sim_returns, nbinsx=50, marker_color="#84ccfb", opacity=1, name="Simulated Returns", histnorm="probability density"))

    # add VaR line
    fig.add_trace(go.Scatter(x=[VaR_95_mc, VaR_95_mc], y=[0, y_max], mode="lines", line=dict(color="red", dash="dash"), name=f"VaR 95%: {VaR_95_mc:.4f}"))
    fig.add_trace(go.Scatter(x=[CVaR_95_mc, CVaR_95_mc], y=[0, y_max], mode="lines", line=dict(color="orange", dash="dash"), name=f"CVaR 95%: {CVaR_95_mc:.4f}"))
    
    # layout
    st.write("### Monte Carlo simulated portfolio returns")
    fig.update_layout(xaxis_title="Simulated Return", yaxis_title="Probability Density", template="plotly_white")
    
    st.plotly_chart(fig, use_container_width=True)