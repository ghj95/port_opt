# import relevant libraries

from datetime import date, timedelta
import yfinance as yf
import streamlit as st
import streamlit_shadcn_ui as ui
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from interpretations import optimization_strategies_info
from interpretations import appinfo

# load asset data
# streamlit code included for web app interface

# designing menus for user input

st.set_page_config(page_title="Stock Portfolio Optimizer", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

st.markdown("## 📊 Stock Portfolio Optimizer App")
linkedin_url = "https://www.linkedin.com/in/gabriel-hardy-joseph/"
st.markdown(
    f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;">`Created by : Gabriel Hardy-Joseph`</a>',
    unsafe_allow_html=True,
)

appinfo()

with st.expander("View Optimization Methodology"):
       optimization_strategies_info()

st.sidebar.header("Select assets and parameters")
assets = st.sidebar.multiselect("Select stocks:", ["AMZN", "GOOG", "MSFT", "AAPL"], ["AMZN", "GOOG", "MSFT"])
start_date = st.sidebar.date_input("Start date", date.today() - timedelta(days=365))
end_date = st.sidebar.date_input("End date", date.today())

if st.sidebar.button("Fetch data"):

    # fetch stock data
    data = yf.download(assets, start=start_date, end=end_date)["Close"]             #stocks

    sp500 = yf.download("^GSPC", start=start_date, end=end_date)["Close"]           #index
    sp500.rename(columns={"^GSPC": "S&P 500"}, inplace=True)

    risk_free_rate = yf.download("^TNX", period="1d")["Close"].iloc[-1]          #10y US Treasury bonds
    risk_free_rate = risk_free_rate["^TNX"]

    # compute returns
    returns = data.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    # optimize portfolio
    n_assets = len(assets)      # count nbr assets
    w = cp.Variable(n_assets)           # define vector of optimization variables
    expected_return = mean_returns.values @ w           # compute expected return : {Portfolio Return} = {Asset Returns}^T * {Portfolio Weights}

    portfolio_risk = cp.quad_form(w, cov_matrix.values)         # compute portoflio risk (variance)
    objective = cp.Maximize(expected_return - 0.5 * portfolio_risk)         # define objective function : max({Expected Return} - 0.5 * {Portfolio Risk})
    constraints = [cp.sum(w) == 1, w >= 0]          # constraints : fully invested portfolio + no short selling
    problem = cp.Problem(objective, constraints)            # define optimization problem
    problem.solve()         # solve

    # display optimal weights
    st.divider()
    optimal_weights = pd.Series(w.value, index=assets)
    optimal_weights_df = pd.DataFrame({"Asset": assets, "Weight (%)": (optimal_weights * 100).round(2)})
    st.write("### Summary")
    st.write("##### Optimal Portfolio Weights")
    
    # portfolio performance
    normalized_prices = data / data.iloc[0] * 100           # normalize to start at 100
    portfolio_performance = (normalized_prices * optimal_weights).sum(axis=1)
    
    # index performance
    normalized_index = sp500 / sp500.iloc[0] * 100          # normalize to start at 100
    normalized_index["Portfolio"] = portfolio_performance.reindex(normalized_index.index, method="ffill")

    left_table, right_pie = st.columns(2)
    with left_table:
        ui.table(optimal_weights_df)
    with right_pie:
        clean_weights = optimal_weights_df[optimal_weights_df["Weight (%)"] != 0]
        fig1 = px.pie(clean_weights, names="Asset", values="Weight (%)", hole=0.3, color_discrete_sequence=px.colors.sequential.GnBu)
        fig1.update_layout(width=200, height=200, showlegend=True, margin=dict(t=0, b=40, l=0, r=0))
        st.plotly_chart(fig1, use_container_width=True)
    
    # performance metrics
    
    # portoflio performance
    V_i = portfolio_performance.iloc[0]         # intiail value
    V_f = portfolio_performance.iloc[-1]            # final value
    T = (portfolio_performance.index[-1] - portfolio_performance.index[0]).days / 365           

    # annualized return
    portfolio_mean = ((V_f / V_i) ** (1 / T) - 1) * 100  

    # portfolio std
    portfolio_std = np.sqrt(optimal_weights.T @ cov_matrix @ optimal_weights) * np.sqrt(252) * 100

    # portfolio sharpe
    portfolio_sharpe = (portfolio_mean - risk_free_rate) / portfolio_std

    # index perfofmance
    index_returns = sp500.pct_change().dropna()
    index_mean = index_returns.mean() * 252 * 100  # Convert to percentage
    index_std = index_returns.std() * np.sqrt(252) * 100
    index_sharpe = (index_mean - risk_free_rate) / index_std

    # display
    st.write("#### Optimal Portfolio Performance")
    left_col, right_col = st.columns(2)
    with left_col:
        st.markdown(f"**Portfolio Annualized Return:** {portfolio_mean:.2f}%")
        st.markdown(f"**Portfolio Volatility:** {portfolio_std:.2f}%")
        st.markdown(f"**Portfolio Sharpe Ratio:** {portfolio_sharpe:.2f}")

    with right_col:
        st.markdown(f"**S&P 500 Annualized Return:** {index_mean.iloc[0]:.2f}%")
        st.markdown(f"**S&P 500 Volatility:** {index_std.iloc[0]:.2f}%")
        st.markdown(f"**S&P 500 Sharpe Ratio:** {index_sharpe.iloc[0]:.2f}")
    
    st.divider()
    st.write("### Portfolio Returns")
    fig2 = px.line(normalized_index, x=normalized_index.index, y=normalized_index.columns, color_discrete_sequence=px.colors.sequential.Greys)
    fig2.update_layout(width=1000, height=400, legend=dict(orientation="h", entrywidth=70, yanchor="bottom", xanchor="left", y=-0.3, x=0.35), legend_title=None, xaxis_title="Date", yaxis_title="Normalized Prices")
    for trace in fig2.data:
        if trace.name == "Portfolio":
            trace.line.width = 3
            trace.line.color = "#80ac8c"
        else:
            trace.opacity = 0.5
            trace.line.color = "#b5b5b5"
    
    st.plotly_chart(fig2, use_container_width=True)

    # simulation parameters
    n_sim = 10000
    time_horizon = 1

    # simulate and compute random porfolio
    sim_returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_sim)
    port_sim_returns = sim_returns @ optimal_weights

    # compute mc var and cvar
    VaR_95_mc = np.percentile(port_sim_returns, 5)
    CVaR_95_mc = port_sim_returns[port_sim_returns <= VaR_95_mc].mean()

    # create histogram
    fig3 = go.Figure()
    hist_counts, bin_edges = np.histogram(port_sim_returns, bins=50, density=True)
    y_max = max(hist_counts)

    # add histogram of simulated returns
    fig3.add_trace(go.Histogram(x=port_sim_returns, nbinsx=50, marker_color="#80ac8c", opacity=0.5, name="Simulated Returns", histnorm="probability density"))

    # add VaR line
    fig3.add_trace(go.Scatter(x=[VaR_95_mc, VaR_95_mc], y=[0, y_max], mode="lines", line=dict(color="black", dash="dash"), name=f"VaR 95%: {VaR_95_mc:.4f}"))
    fig3.add_trace(go.Scatter(x=[CVaR_95_mc, CVaR_95_mc], y=[0, y_max], mode="lines", line=dict(color="grey", dash="dash"), name=f"CVaR 95%: {CVaR_95_mc:.4f}"))
    
    # layout
    st.divider()

    st.write("### Monte Carlo Simulated Portfolio Returns")
    fig3.update_layout(xaxis_title="Simulated Return", yaxis_title="Probability Density", template="plotly_white")
    
    st.plotly_chart(fig3, use_container_width=True)