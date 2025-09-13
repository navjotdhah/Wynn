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

st.title("🎰 Wynn Resorts (WYNN) Valuation Dashboard")
st.markdown("Created by: **Navjot Dhah**")

# ---------------------------
# Load Data
# ---------------------------
ticker = "WYNN"
stock = yf.Ticker(ticker)

st.sidebar.header("Company Info")
st.sidebar.write(f"**Ticker:** {ticker}")
st.sidebar.write(f"**Sector:** {stock.info.get('sector', 'N/A')}")
st.sidebar.write(f"**Industry:** {stock.info.get('industry', 'N/A')}")
st.sidebar.write(f"**Market Cap:** {stock.info.get('marketCap', 'N/A'):,}")

# ---------------------------
# Historical Price Chart
# ---------------------------
st.subheader("Stock Price History")
hist = stock.history(period="5y")

fig = go.Figure()
fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"], mode="lines", name="Close Price"))
fig.update_layout(title="Wynn Resorts (WYNN) Stock Price - 5Y",
                  xaxis_title="Date", yaxis_title="Price (USD)")
st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Financial Statements
# ---------------------------
st.subheader("Financial Statements")

fin_choice = st.selectbox("Choose Financial Statement:", ["Balance Sheet", "Income Statement", "Cash Flow"])

if fin_choice == "Balance Sheet":
    fin = stock.balance_sheet
elif fin_choice == "Income Statement":
    fin = stock.financials
else:
    fin = stock.cashflow

# Transpose for readability
fin = fin.T

# Fix: format only numeric columns
numeric_cols = fin.select_dtypes(include=[np.number]).columns
st.dataframe(fin.style.format({col: "{:,.0f}" for col in numeric_cols}))

# ---------------------------
# Intrinsic Value (DCF Approximation)
# ---------------------------
st.subheader("Intrinsic Valuation (DCF Approximation)")

# User inputs
growth_rate = st.slider("Revenue Growth Rate (next 5 years)", 0.01, 0.15, 0.06)
discount_rate = st.slider("Discount Rate (WACC)", 0.05, 0.15, 0.09)
terminal_growth = st.slider("Terminal Growth Rate", 0.01, 0.05, 0.025)

# Pull free cash flow (FCF) data
cf = stock.cashflow
try:
    fcf = cf.loc["Total Cash From Operating Activities"] - cf.loc["Capital Expenditures"]
    last_fcf = fcf.iloc[0]
except:
    last_fcf = 500_000_000  # fallback assumption

years = np.arange(1, 6)
projected_fcfs = [last_fcf * (1 + growth_rate) ** yr for yr in years]
discounted_fcfs = [fcf / (1 + discount_rate) ** yr for yr, fcf in zip(years, projected_fcfs)]

# Terminal value
terminal_value = projected_fcfs[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)
terminal_value_discounted = terminal_value / (1 + discount_rate) ** 5

intrinsic_value = sum(discounted_fcfs) + terminal_value_discounted

shares_outstanding = stock.info.get("sharesOutstanding", 112_000_000)
intrinsic_price = intrinsic_value / shares_outstanding

st.metric("Intrinsic Value per Share", f"${intrinsic_price:,.2f}")

# ---------------------------
# UAE Casino Project Valuation
# ---------------------------
st.subheader("Upcoming UAE Casino Project Impact")

st.markdown("""
Wynn Resorts is developing a **UAE casino project**, estimated to be a multibillion-dollar venture.  
Here, we model its contribution to valuation under different scenarios.
""")

uae_investment = st.number_input("Estimated Investment (USD)", value=3500000000, step=100000000)
uae_expected_return = st.slider("Expected ROIC on UAE Project", 0.05, 0.20, 0.12)
uae_years = st.slider("Years until Project Maturity", 3, 10, 5)

uae_cashflow = uae_investment * uae_expected_return
uae_pv = uae_cashflow / ((1 + discount_rate) ** uae_years)

new_intrinsic_value = intrinsic_value + uae_pv
new_intrinsic_price = new_intrinsic_value / shares_outstanding

st.metric("Revised Intrinsic Value per Share (with UAE Project)", f"${new_intrinsic_price:,.2f}")

# ---------------------------
# Comparables
# ---------------------------
st.subheader("Peer Comparison")

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

# ---------------------------
# Conclusion
# ---------------------------
st.subheader("Conclusion")
current_price = stock.history(period="1d")["Close"].iloc[-1]
if new_intrinsic_price > current_price:
    st.success(f"Wynn Resorts (WYNN) appears **undervalued**. Intrinsic Price = ${new_intrinsic_price:,.2f}, Current Price = ${current_price:,.2f}")
else:
    st.error(f"Wynn Resorts (WYNN) appears **overvalued**. Intrinsic Price = ${new_intrinsic_price:,.2f}, Current Price = ${current_price:,.2f}")
