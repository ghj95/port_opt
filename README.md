### Overview
This project is a Streamlit-based web application that performs portfolio optimization and risk analysis using Mean-Variance Optimization (MVO). It allows users to select assets, fetch financial data, compute optimal portfolio weights, and compare the performance of their portfolio against a benchmark index, such as the S&P 500.

### Features
* Fetch real-time financial data from Yahoo Finance (yfinance).
* Perform Mean-Variance Optimization (MVO) using cvxpy to determine optimal asset allocations.
* Compute key portfolio metrics:
* Expected Return
* Portfolio Standard Deviation (Risk)
* Sharpe Ratio (Risk-Adjusted Return)
* Compare portfolio performance against the S&P 500.
* Interactive visualizations using Plotly for portfolio vs. index performance.
* Dynamic user input for asset selection and date ranges.

### How It Works
* Select assets and date range in the Streamlit sidebar.
* Fetch historical stock data from Yahoo Finance.
* Compute optimal portfolio allocation using convex optimization.
* Display portfolio performance metrics such as expected return, risk, and Sharpe ratio.
* Compare the portfolio against the S&P 500 through visualizations.
