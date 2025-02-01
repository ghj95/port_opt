import streamlit as st

def appinfo():
    st.markdown("This interactive app construct an optimal portfolio by applying Modern Portfolio Theory (MPT). It fetches real-time financial data, computes expected returns, and optimizes asset allocation to maximize returns while minimizing risk. It also evaluates portfolio risk by calculating Value at Risk (VaR) and Conditional Value at Risk (CVaR).")
    st.write("⬅️ Try it out !")

def optimization_strategies_info():
    # --- Introduction ---
    st.write(
        "This application uses **Mean-Variance Optimization (MVO)**, "
        "based on **Modern Portfolio Theory (MPT)**, to construct an optimal portfolio "
        "by balancing expected return and risk."
    )

    # --- Mean-Variance Optimization ---
    st.markdown("#### Mean-Variance Optimization")

    st.write("The objective of portfolio optimization is to maximize the risk-adjusted return:")
    st.latex(r"\max_{w} \left( E(R_p) - \lambda \cdot \sigma_p^2 \right)")

    st.write("where:")
    st.markdown(
        r"""
        - $E(R_p)$ is the **expected portfolio return**.
        - $\sigma_p^2$ is the **portfolio variance (risk)**.
        - $\lambda$ is the **risk-aversion parameter**.
        - $w$ represents the **portfolio weights**.
        """
    )

    # --- Portfolio Expected Return ---
    st.markdown("#### Portfolio Expected Return")

    st.write("The portfolio's expected return is computed as:")
    st.latex(r"E(R_p) = w^T \cdot E(R)")

    st.write("where:")
    st.markdown(
        r"""
        - $w$ is the **vector of portfolio weights**.
        - $E(R)$ is the **vector of expected asset returns**.
        """
    )

    # --- Portfolio Risk (Variance) ---
    st.markdown("#### Portfolio Risk (Variance)")

    st.write("Portfolio risk is determined using the covariance matrix:")
    st.latex(r"\sigma_p^2 = w^T \Sigma w")

    st.write("where:")
    st.markdown(
        r"""
        - $\Sigma$ is the **covariance matrix of asset returns**.
        - $w$ is the **vector of portfolio weights**.
        """
    )

    # --- Constraints ---
    st.markdown("#### Optimization Constraints")

    st.write("The portfolio optimization problem is subject to the following constraints:")
    st.latex(r"\sum w_i = 1, \quad w_i \geq 0 \quad \forall i")

    st.write("where:")
    st.markdown(
        r"""
        - The first constraint ensures the portfolio is **fully invested**: $\sum w_i = 1$.
        - The second constraint prevents **short selling**: $w_i \geq 0$.
        """
    )

    # --- Portfolio Standard Deviation ---
    st.markdown("#### Portfolio Standard Deviation")

    st.write("The standard deviation (volatility) of the portfolio is given by:")
    st.latex(r"\sigma_p = \sqrt{ w^T \Sigma w }")

    st.write("where:")
    st.markdown(
        r"""
        - $\sigma_p$ represents **portfolio volatility**.
        - $w$ is the **vector of asset weights**.
        - $\Sigma$ is the **covariance matrix of asset returns**.
        """
    )

    # --- Sharpe Ratio ---
    st.markdown("#### Sharpe Ratio")

    st.write("The **Sharpe Ratio** evaluates the risk-adjusted return of the portfolio:")
    st.latex(r"\text{Sharpe Ratio} = \frac{E(R_p) - R_f}{\sigma_p}")

    st.write("where:")
    st.markdown(
        r"""
        - $R_f$ is the **risk-free rate**.
        - Higher values indicate **better risk-adjusted performance**.
        """
    )

    # --- Conclusion ---
    st.write(
        "By solving this **convex optimization problem**, the optimal portfolio allocation is determined, "
        "allowing for an optimal balance between return and risk."
    )

    st.markdown("**Benchmark**: S&P 500 (NYSE : ^GSPC)")
    st.markdown("**Risk free rate**: 10Y US Treasury Bond Yields (NYSE : ^TNX)")