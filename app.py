import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Wynn Resorts Valuation",
    page_icon="🎰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Custom Bloomberg-Style CSS
# ---------------------------
st.markdown(
    """
    <style>
    body {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .block-container {
        padding: 1.5rem 2rem;
    }
    h1, h2, h3 {
        color: #FECB2F;
        font-weight: 700;
    }
    .stMetric {
        background-color: #1E1E1E;
        padding: 10px;
        border-radius: 10px;
    }
    .stDataFrame {
        background-color: #1E1E1E;
    }
    .css-1d391kg {  /* Sidebar background */
        background-color: #111111 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# App Title
# ---------------------------
st.title("🎰 Wynn Resorts (WYNN) Valuation Dashboard")
st.caption("Created by: **Navjot Dhah** | Inspired by Bloomberg Terminal")

# ---------------------------
# Sidebar Navigation
# ---------------------------
menu = st.sidebar.radio("Navigation", ["Overview", "Financial Statements", "Valuation", "UAE Project", "Comparables"])

ticker = "WYNN"
stock = yf.Ticker(ticker)

# ---------------------------
# OVERVIEW
# ---------------------------
if menu == "Overview":
    st.header("Company Overview")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Ticker", ticker)
    with col2:
        st.metric("Sector", stock.info.get("sector", "N/A"))
    with col3:
        st.metric("Industry", stock.info.get("industry", "N/A"))

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Market Cap", f"${stock.info.get('marketCap', 0):,.0f}")
    with col2:
        st.metric("Shares Outstanding", f"{stock.info.get('sharesOutstanding', 0):,}")

    st.subheader("Stock Price History")
    hist = stock.history(period="5y")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"], mode="lines", name="Close Price", line=dict(color="#FECB2F")))
    fig.update_layout(template="plotly_dark", title="Wynn Resorts (WYNN) Stock Price - 5Y",
                      xaxis_title="Date", yaxis_title="Price (USD)")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# FINANCIAL STATEMENTS
# ---------------------------
elif menu == "Financial Statements":
    st.header("Financial Statements")

    fin_choice = st.selectbox("Choose Statement:", ["Balance Sheet", "Income Statement", "Cash Flow"])

    if fin_choice == "Balance Sheet":
        fin = stock.balance_sheet
    elif fin_choice == "Income Statement":
        fin = stock.financials
    else:
        fin = stock.cashflow

    fin = fin.T
    numeric_cols = fin.select_dtypes(include=[np.number]).columns
    st.dataframe(fin.style.format({col: "{:,.0f}" for col in numeric_cols}))

# ---------------------------
# VALUATION
# ---------------------------
elif menu == "Valuation":
    st.header("Intrinsic Valuation (DCF Approximation)")

    growth_rate = st.slider("Revenue Growth Rate (next 5 years)", 0.01, 0.15, 0.06)
    discount_rate = st.slider("Discount Rate (WACC)", 0.05, 0.15, 0.09)
    terminal_growth = st.slider("Terminal Growth Rate", 0.01, 0.05, 0.025)

    cf = stock.cashflow
    try:
        fcf = cf.loc["Total Cash From Operating Activities"] - cf.loc["Capital Expenditures"]
        last_fcf = fcf.iloc[0]
    except:
        last_fcf = 500_000_000

    years = np.arange(1, 6)
    projected_fcfs = [last_fcf * (1 + growth_rate) ** yr for yr in years]
    discounted_fcfs = [fcf / (1 + discount_rate) ** yr for yr, fcf in zip(years, projected_fcfs)]

    terminal_value = projected_fcfs[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)
    terminal_value_discounted = terminal_value / (1 + discount_rate) ** 5

    intrinsic_value = sum(discounted_fcfs) + terminal_value_discounted
    shares_outstanding = stock.info.get("sharesOutstanding", 112_000_000)
    intrinsic_price = intrinsic_value / shares_outstanding

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Intrinsic Value (Total)", f"${intrinsic_value:,.0f}")
    with col2:
        st.metric("Intrinsic Value per Share", f"${intrinsic_price:,.2f}")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=[f"Year {yr}" for yr in years], y=discounted_fcfs, name="Discounted FCF", marker_color="#00FF00"))
    fig.add_trace(go.Bar(x=["Terminal Value"], y=[terminal_value_discounted], name="Terminal Value", marker_color="#FECB2F"))
    fig.update_layout(template="plotly_dark", title="DCF Components", xaxis_title="", yaxis_title="USD")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# UAE PROJECT
# ---------------------------
elif menu == "UAE Project":
    st.header("UAE Casino Project Valuation")

    st.markdown("Wynn Resorts is developing a **UAE casino project**, a multi-billion-dollar venture. Let’s model its impact.")

    uae_investment = st.number_input("Estimated Investment (USD)", value=3500000000, step=100000000)
    uae_expected_return = st.slider("Expected ROIC on UAE Project", 0.05, 0.20, 0.12)
    uae_years = st.slider("Years until Project Maturity", 3, 10, 5)
    discount_rate = 0.09

    uae_cashflow = uae_investment * uae_expected_return
    uae_pv = uae_cashflow / ((1 + discount_rate) ** uae_years)

    cf = stock.cashflow
    try:
        fcf = cf.loc["Total Cash From Operating Activities"] - cf.loc["Capital Expenditures"]
        last_fcf = fcf.iloc[0]
    except:
        last_fcf = 500_000_000

    intrinsic_value = last_fcf * 10  # Simplified baseline
    shares_outstanding = stock.info.get("sharesOutstanding", 112_000_000)

    new_intrinsic_value = intrinsic_value + uae_pv
    new_intrinsic_price = new_intrinsic_value / shares_outstanding

    col1, col2 = st.columns(2)
    with col1:
        st.metric("PV of UAE Project", f"${uae_pv:,.0f}")
    with col2:
        st.metric("Revised Intrinsic Price", f"${new_intrinsic_price:,.2f}")

# ---------------------------
# COMPARABLES
# ---------------------------
elif menu == "Comparables":
    st.header("Peer Comparison")

    peers = ["MGM", "LVS", "MLCO"]
    peer_data = {}
    for peer in peers:
        try:
            p = yf.Ticker(peer)
            peer_data[peer] = {
                "Market Cap": p.info.get("marketCap", np.nan),
                "P/E Ratio": p.info.get("trailingPE", np.nan),
                "EV/EBITDA": p.info.get("enterpriseToEbitda", np.nan)
            }
        except:
            continue

    peer_df = pd.DataFrame(peer_data).T
    numeric_cols_peer = peer_df.select_dtypes(include=[np.number]).columns
    st.dataframe(peer_df.style.format({col: "{:,.2f}" for col in numeric_cols_peer}))

