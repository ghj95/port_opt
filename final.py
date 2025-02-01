# import relevant libraries

from datetime import date
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

st.set_page_config(page_title="Portfolio Optimizer")

st.markdown("## Portfolio Optimizer App")
col1, col2 = st.columns([0.14, 0.86], gap="small")
col1.write("`Created by:`")
linkedin_url = "https://www.linkedin.com/in/gabriel-hardy-joseph/"
col2.markdown(
    f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="15" height="15" style="vertical-align: middle; margin-right: 10px;">`Gabriel Hardy-Joseph`</a>',
    unsafe_allow_html=True,
)

appinfo()

with st.expander("View Optimization Methodology"):
       optimization_strategies_info()

st.sidebar.header("Select assets and parameters")
assets = st.sidebar.multiselect("Select stocks:", ["XOM", "PG", "JNJ", "V", "ABBV"], ["XOM", "PG", "JNJ"])
start_date = st.sidebar.date_input("Start date", "2019-01-01")
end_date = st.sidebar.date_input("End date", date.today())

if st.sidebar.button("Fetch data"):
    st.sidebar.success("Data uploaded")

    # fetch stock data
    data = yf.download(assets, start=start_date, end=end_date)["Close"]             #stocks

    sp500 = yf.download("^GSPC", start=start_date, end=end_date)["Close"]           #index
    sp500.rename(columns={"^GSPC": "S&P 500"}, inplace=True)

    risk_free_rate = yf.download("^TNX")["Close"]          #10y US Treasury bonds
    
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
    normalized_index = sp500 / sp500.iloc[0]["S&P 500"] * 100          # normalize to start at 100
    normalized_index["Portfolio"] = portfolio_performance.reindex(normalized_index.index, method="ffill")

    left_table, right_pie = st.columns(2)
    with left_table:
        ui.table(optimal_weights_df)
    with right_pie:
        clean_weights = optimal_weights_df[optimal_weights_df["Weight (%)"] != 0]
        fig1 = px.pie(clean_weights, names="Asset", values="Weight (%)", hole=0.3, color_discrete_sequence=px.colors.sequential.Greens)
        fig1.update_layout(width=200, height=200, showlegend=True, margin=dict(t=0, b=40, l=0, r=0))
        st.plotly_chart(fig1, use_container_width=True)
    
    # # performance metrics
    # portfolio_mean = round(((portfolio_performance.iloc[0] / portfolio_performance.iloc[-1]) ** (1 / (portfolio_performance.index[-1] - portfolio_performance.index[0]).days / 365) - 1) * 100)
    # portfolio_std = np.sqrt(optimal_weights.T @ cov_matrix @ optimal_weights) * np.sqrt(252)
    # portfolio_sharpe = portfolio_mean / portfolio_risk
    
    # index_returns = sp500.pct_change().dropna()
    # index_mean = index_returns.mean() * 252
    # index_std = index_returns.std() * np.sqrt(252)
    # index_sharpe = index_mean / index_std

    # st.write("#### Optimal Portfolio Performance")
    # left_col, right_col = st.columns(2)
    # left_col.markdown(f"Portfolio Annualized Returns : {portfolio_mean}%")
    # left_col.markdown(f"Portfolio Volatility : {portfolio_std}%")
    # left_col.markdown(f"Portfolio Sharpe Ratio : {portfolio_sharpe}")

    # right_col, right_col = st.columns(2)
    # right_col.markdown(f"Index Returns : {index_mean}%")
    # right_col.markdown(f"Index Volatility : {index_std}%")
    # right_col.markdown(f"Index Sharpe Ratio : {index_sharpe}")
    
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

    print(f"Monte Carlo VaR 95%: {VaR_95_mc:.4f}")
    print(f"Monte Carlo VaR 95%: {CVaR_95_mc:.4f}")

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
