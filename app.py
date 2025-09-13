# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import date

# ---------------------------
# Page & CSS (Bloomberg-like)
# ---------------------------
st.set_page_config(page_title="Wynn Resorts — Valuation (WYNN)",
                   page_icon="🎰", layout="wide")

st.markdown(
    """
    <style>
        /* Background + text */
        .main { background-color: #0e1117; color: #e6e6e6; }
        .block-container { padding-top: 1rem; padding-bottom: 1rem; }
        h1, h2, h3 { color: #f5c518; font-weight: 700; }
        /* Card-like look for metric containers */
        .metric-card {
            background: #111316;
            padding: 12px;
            border-radius: 8px;
            border: 1px solid #222426;
        }
        /* Sidebar background */
        .css-1d391kg { background-color: #0b0c0e !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Title
# ---------------------------
st.title("🎰 Wynn Resorts (WYNN) — Interactive Valuation Dashboard")
st.caption("Built by Navjot Dhah — Bloomberg-style theme with scenario analysis for Wynn Al Marjan (UAE)")

# ---------------------------
# Sidebar - global inputs
# ---------------------------
st.sidebar.header("Model settings & assumptions")

ticker = st.sidebar.text_input("Ticker", value="WYNN").upper()
use_live = st.sidebar.checkbox("Pull live data from yfinance", value=True)
wacc = st.sidebar.number_input("Discount rate / WACC (%)", min_value=0.0, max_value=50.0, value=9.0, step=0.1) / 100.0
terminal_growth = st.sidebar.number_input("Terminal growth (%)", min_value=-2.0, max_value=6.0, value=2.5, step=0.1) / 100.0
projection_years = st.sidebar.selectbox("Explicit projection years", options=[5, 7, 10], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("UAE Project - global")
project_open_year = st.sidebar.number_input("Project Open Year", value=date.today().year + 2, step=1)
project_horizon = st.sidebar.number_input("Project analysis horizon (years)", min_value=5, max_value=15, value=10, step=1)

# ---------------------------
# Load data (yfinance)
# ---------------------------
@st.cache_data(ttl=600)
def fetch_ticker(ticker_sym):
    t = yf.Ticker(ticker_sym)
    # information and statements (may be empty)
    info = t.info if hasattr(t, "info") else {}
    fin = t.financials if hasattr(t, "financials") else pd.DataFrame()
    bs = t.balance_sheet if hasattr(t, "balance_sheet") else pd.DataFrame()
    cf = t.cashflow if hasattr(t, "cashflow") else pd.DataFrame()
    hist = t.history(period="5y") if hasattr(t, "history") else pd.DataFrame()
    return info, fin, bs, cf, hist

if use_live:
    info, fin, bs, cf, hist = fetch_ticker(ticker)
else:
    info, fin, bs, cf, hist = {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# Safe helpers
def safe_number(x, fallback=np.nan):
    try:
        return float(x)
    except Exception:
        return fallback

market_price = safe_number(info.get("currentPrice") or (hist["Close"].iloc[-1] if not hist.empty else np.nan))
market_cap = safe_number(info.get("marketCap"))
shares_outstanding = safe_number(info.get("sharesOutstanding") or info.get("floatShares") or 0) or np.nan
total_debt = safe_number(info.get("totalDebt") or 0)
total_cash = safe_number(info.get("totalCash") or info.get("cash") or 0)

# Header metrics
col1, col2, col3, col4 = st.columns(4)
col1.markdown("<div class='metric-card'><b>Ticker</b><br>{}</div>".format(ticker), unsafe_allow_html=True)
col2.markdown("<div class='metric-card'><b>Market Price</b><br>{}</div>".format(f"${market_price:,.2f}" if not np.isnan(market_price) else "N/A"), unsafe_allow_html=True)
col3.markdown("<div class='metric-card'><b>Market Cap</b><br>{}</div>".format(f"${market_cap:,.0f}" if not np.isnan(market_cap) else "N/A"), unsafe_allow_html=True)
col4.markdown("<div class='metric-card'><b>Shares Outstanding</b><br>{}</div>".format(f"{int(shares_outstanding):,}" if not np.isnan(shares_outstanding) else "N/A"), unsafe_allow_html=True)

# ---------------------------
# Price chart (5y)
# ---------------------------
st.subheader("Price history")
if not hist.empty:
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=hist.index, y=hist["Close"], mode="lines", name="Close", line=dict(color="#f5c518")))
    fig_price.update_layout(template="plotly_dark", height=350, margin=dict(t=30,l=10,r=10,b=10))
    st.plotly_chart(fig_price, use_container_width=True)
else:
    st.info("Historical price data not available (yfinance).")

# ---------------------------
# Financial statements display (safe formatting)
# ---------------------------
st.subheader("Financial Statements (yfinance source)")

def show_statement(df, title):
    st.markdown(f"**{title}**")
    if df is None or df.empty:
        st.info(f"{title} not available via yfinance.")
        return
    # transpose for readability
    df_t = df.T
    # numeric-only formatting
    numeric_cols = df_t.select_dtypes(include=[np.number]).columns
    format_map = {col: "{:,.0f}" for col in numeric_cols}
    st.dataframe(df_t.style.format(format_map), use_container_width=True)

with st.expander("Balance sheet"):
    show_statement(bs, "Balance Sheet")

with st.expander("Income statement"):
    show_statement(fin, "Income Statement")

with st.expander("Cash flow statement"):
    show_statement(cf, "Cash Flow Statement")

# ---------------------------
# DCF model (base)
# ---------------------------
st.subheader("Base DCF Valuation (5-year explicit + terminal)")

# derive/estimate historical FCF if possible
last_fcf = None
if not cf.empty:
    try:
        # yfinance cashflow columns often: 'Total Cash From Operating Activities' (positive), 'Capital Expenditures' (negative)
        cf_t = cf.copy()
        # best-effort lookups
        op_keys = [k for k in cf_t.index if "Operating" in str(k) or "operat" in str(k).lower()]
        capex_keys = [k for k in cf_t.index if "Capital Expend" in str(k) or "capitalexpend" in str(k).lower()]
        # fallback to commonly-named rows
        op = None
        capex = None
        for key in cf_t.index:
            low = str(key).lower()
            if ("operat" in low and ("cash" in low or "activities" in low)) and op is None:
                op = cf_t.loc[key].iloc[0]
            if ("capital" in low and "expend" in low) and capex is None:
                capex = cf_t.loc[key].iloc[0]
        if op is not None:
            # capex might be negative in the table; normal FCF = OCF + CAPEX
            capex_val = capex if capex is not None else 0.0
            last_fcf = float(op) + float(capex_val)
    except Exception:
        last_fcf = None

# Fallback default if not found
if last_fcf is None or np.isnan(last_fcf):
    last_fcf = st.number_input("Manual: Most recent Free Cash Flow (USD)", value=500_000_000.0, step=1e6, format="%.0f")

st.markdown(f"**Using latest FCF =** ${last_fcf:,.0f}")

# interactive DCF assumptions
ass_g = st.number_input("Explicit FCF growth rate (annual %) for projection period", value=5.0, step=0.1) / 100.0
ass_tg = terminal_growth
ass_wacc = wacc
ass_years = projection_years

# project & discount
proj_nominal = [last_fcf * (1 + ass_g) ** i for i in range(1, ass_years + 1)]
proj_pv = [val / ((1 + ass_wacc) ** i) for i, val in enumerate(proj_nominal, start=1)]
terminal_nominal = proj_nominal[-1] * (1 + ass_tg) / (ass_wacc - ass_tg) if ass_wacc > ass_tg else np.nan
terminal_pv = terminal_nominal / ((1 + ass_wacc) ** ass_years) if not np.isnan(terminal_nominal) else np.nan
ev_base = sum(proj_pv) + (terminal_pv if not np.isnan(terminal_pv) else 0)
equity_base = ev_base - (total_debt or 0) + (total_cash or 0)
price_base = equity_base / shares_outstanding if (shares_outstanding and equity_base and not np.isnan(equity_base)) else np.nan

col1, col2, col3 = st.columns(3)
col1.metric("EV (DCF total)", f"${ev_base:,.0f}")
col2.metric("Equity value (net debt adj)", f"${equity_base:,.0f}")
col3.metric("Intrinsic price (base)", f"${price_base:,.2f}" if not np.isnan(price_base) else "N/A")

# stacked bars: PV contributions + terminal
fig = go.Figure()
fig.add_trace(go.Bar(x=[f"Year {i}" for i in range(1, ass_years + 1)], y=proj_pv, name="Discounted FCF", marker_color="#00CC96"))
fig.add_trace(go.Bar(x=["Terminal"], y=[terminal_pv], name="Discounted Terminal", marker_color="#f5c518"))
fig.update_layout(template="plotly_dark", barmode="stack", title="DCF PV Contributions", height=360, margin=dict(t=30))
st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# UAE Project: scenario analysis
# ---------------------------
st.subheader("Wynn Al Marjan (UAE) — Project Scenario Analysis")

st.markdown("""
We model three scenarios (Bear / Base / Bull). For each scenario you can adjust:
- Project CAPEX (total investment)
- Stabilized annual incremental FCF (on stabilization)
- Ramp years (how many years to reach stabilization)
We compute the **NPV (discounted incremental FCF)** and then the implied uplift to intrinsic price (per share).
""")

# Default scenario parameters (we'll let the user adjust)
colA, colB, colC = st.columns(3)
with colA:
    st.markdown("**Bear scenario**")
    bear_capex = st.number_input("Bear CAPEX (USD)", value=2_500_000_000, step=100_000_000, key="bear_capex")
    bear_stab_fcf = st.number_input("Bear stabilized incremental FCF (USD)", value=250_000_000, step=10_000_000, key="bear_fcf")
    bear_ramp = st.number_input("Bear ramp years", min_value=0, max_value=10, value=3, key="bear_ramp")
with colB:
    st.markdown("**Base scenario**")
    base_capex = st.number_input("Base CAPEX (USD)", value=3_900_000_000, step=100_000_000, key="base_capex")
    base_stab_fcf = st.number_input("Base stabilized incremental FCF (USD)", value=600_000_000, step=10_000_000, key="base_fcf")
    base_ramp = st.number_input("Base ramp years", min_value=0, max_value=10, value=2, key="base_ramp")
with colC:
    st.markdown("**Bull scenario**")
    bull_capex = st.number_input("Bull CAPEX (USD)", value=5_000_000_000, step=100_000_000, key="bull_capex")
    bull_stab_fcf = st.number_input("Bull stabilized incremental FCF (USD)", value=1_000_000_000, step=10_000_000, key="bull_fcf")
    bull_ramp = st.number_input("Bull ramp years", min_value=0, max_value=10, value=2, key="bull_ramp")

# helper to compute project NPV (incremental FCF schedule)
def project_npv(capex, stab_fcf, ramp_years, open_year, horizon, discount_rate):
    """
    - capex: upfront CAPEX (paid at t=0 for simplicity)
    - stab_fcf: stabilized annual incremental FCF (from stabilization onward)
    - ramp_years: number of years to ramp after open (open year has partial year)
    - open_year: calendar year project opens
    - horizon: years to model (from now)
    - discount_rate: decimal
    returns: npv_of_incremental_cashflows (discounted), schedule_df
    """
    today = date.today().year
    schedule = []
    npv = 0.0
    for t in range(1, horizon + 1):
        cal_year = today + t
        # determine incremental FCF for that year
        if cal_year < open_year:
            inc = 0.0
        elif cal_year == open_year:
            # assume partial year (25% - adjustable choice)
            inc = stab_fcf * 0.25
        elif open_year < cal_year <= open_year + ramp_years:
            # linear ramp from 25% -> 100% across ramp_years
            years_since_open = cal_year - open_year
            # ramp progress fraction (from 0 to 1)
            frac = 0.25 + (years_since_open / (ramp_years + 0.0)) * (1.0 - 0.25)
            inc = stab_fcf * frac
        else:
            inc = stab_fcf
        disc = inc / ((1 + discount_rate) ** t)
        schedule.append({"Year": cal_year, "Incremental FCF": inc, "Discounted PV": disc})
        npv += disc
    # subtract capex (assume paid today at t=0)
    net_npv = npv - capex
    schedule_df = pd.DataFrame(schedule)
    return net_npv, npv, schedule_df

# compute for each scenario
bear_netnpv, bear_grossnpv, bear_sched = project_npv(bear_capex, bear_stab_fcf, bear_ramp, project_open_year, project_horizon, ass_wacc)
base_netnpv, base_grossnpv, base_sched = project_npv(base_capex, base_stab_fcf, base_ramp, project_open_year, project_horizon, ass_wacc)
bull_netnpv, bull_grossnpv, bull_sched = project_npv(bull_capex, bull_stab_fcf, bull_ramp, project_open_year, project_horizon, ass_wacc)

# present scenario table
scenario_df = pd.DataFrame({
    "Scenario": ["Bear", "Base", "Bull"],
    "CapEx (USD)": [bear_capex, base_capex, bull_capex],
    "Stabilized annual FCF (USD)": [bear_stab_fcf, base_stab_fcf, bull_stab_fcf],
    "Ramp years": [bear_ramp, base_ramp, bull_ramp],
    "Discounted Gross FCF (PV)": [bear_grossnpv, base_grossnpv, bull_grossnpv],
    "Net NPV (PV - CapEx)": [bear_netnpv, base_netnpv, bull_netnpv]
})

# formatting numeric columns only
num_cols = scenario_df.select_dtypes(include=[np.number]).columns
fmt = {c: "{:,.0f}" for c in num_cols}
st.markdown("**Scenario results (NPV)**")
st.dataframe(scenario_df.style.format(fmt), use_container_width=True)

# compute implied price per share after adding project NPV to base DCF EV
def implied_price_with_project(ev_base, project_netpv, total_debt, total_cash, shares):
    ev_combined = ev_base + project_netpv
    equity = ev_combined - (total_debt or 0) + (total_cash or 0)
    price = equity / shares if (shares and not np.isnan(equity)) else np.nan
    return price, ev_combined, equity

price_bear, ev_bear, eq_bear = implied_price_with_project(ev_base, bear_netnpv, total_debt, total_cash, shares_outstanding)
price_base_scenario, ev_base_scenario, eq_base_scenario = implied_price_with_project(ev_base, base_netnpv, total_debt, total_cash, shares_outstanding)
price_bull, ev_bull, eq_bull = implied_price_with_project(ev_base, bull_netnpv, total_debt, total_cash, shares_outstanding)

# summary display
st.markdown("**Implied price per share under scenarios**")
summary_df = pd.DataFrame({
    "Scenario": ["Bear", "Base", "Bull", "Market"],
    "Implied Price ($)": [price_bear, price_base_scenario, price_bull, market_price]
})
summary_df["Implied Price ($)"] = summary_df["Implied Price ($)"].map(lambda x: f"${x:,.2f}" if (not pd.isna(x)) else "N/A")
st.table(summary_df)

# plot scenario comparison
fig_scen = go.Figure()
fig_scen.add_trace(go.Bar(x=["Bear", "Base", "Bull"], y=[price_bear or 0, price_base_scenario or 0, price_bull or 0],
                          marker_color=["#FF4C4C", "#F5C518", "#00CC96"], name="Intrinsic price with project"))
fig_scen.add_trace(go.Scatter(x=["Bear", "Base", "Bull"], y=[market_price, market_price, market_price],
                             mode="lines+markers", name="Market Price", line=dict(color="#8888FF"), marker=dict(size=8)))
fig_scen.update_layout(template="plotly_dark", title="Implied share price by scenario vs market", height=420)
st.plotly_chart(fig_scen, use_container_width=True)

# allow user to inspect schedule for any scenario
st.markdown("Expand scenario schedule to inspect year-by-year incremental FCF (discounted).")
with st.expander("Bear schedule"):
    if not bear_sched.empty:
        st.dataframe(bear_sched.assign(**{"Incremental FCF": bear_sched["Incremental FCF"].map(lambda x: f"${x:,.0f}"),
                                          "Discounted PV": bear_sched["Discounted PV"].map(lambda x: f"${x:,.0f}")}))
with st.expander("Base schedule"):
    if not base_sched.empty:
        st.dataframe(base_sched.assign(**{"Incremental FCF": base_sched["Incremental FCF"].map(lambda x: f"${x:,.0f}"),
                                          "Discounted PV": base_sched["Discounted PV"].map(lambda x: f"${x:,.0f}")}))
with st.expander("Bull schedule"):
    if not bull_sched.empty:
        st.dataframe(bull_sched.assign(**{"Incremental FCF": bull_sched["Incremental FCF"].map(lambda x: f"${x:,.0f}"),
                                          "Discounted PV": bull_sched["Discounted PV"].map(lambda x: f"${x:,.0f}") }))

# ---------------------------
# Sensitivity: heatmap of implied price (WACC vs terminal growth)
# ---------------------------
st.subheader("Sensitivity: Intrinsic price (WACC vs Terminal growth)")

# small grid for sensitivity
wacc_grid = np.arange(max(ass_wacc - 0.02, 0.04), ass_wacc + 0.021, 0.005)  # +/- 2% around chosen WACC
tg_grid = np.arange(max(ass_tg - 0.01, -0.01), ass_tg + 0.011, 0.0025)  # +/-1% around tg

sens_matrix = np.zeros((len(tg_grid), len(wacc_grid)))
for i, tg in enumerate(tg_grid):
    for j, wa in enumerate(wacc_grid):
        # recalc terminal value safely
        try:
            proj_nom = [last_fcf * (1 + ass_g) ** k for k in range(1, ass_years + 1)]
            proj_pv_local = [val / ((1 + wa) ** k) for k, val in enumerate(proj_nom, start=1)]
            term_nom = proj_nom[-1] * (1 + tg) / (wa - tg) if wa > tg else np.nan
            term_pv_local = term_nom / ((1 + wa) ** ass_years) if not np.isnan(term_nom) else np.nan
            ev_local = sum(proj_pv_local) + (term_pv_local if not np.isnan(term_pv_local) else 0)
            equity_local = ev_local - (total_debt or 0) + (total_cash or 0)
            price_local = equity_local / shares_outstanding if (shares_outstanding and not np.isnan(equity_local)) else np.nan
            sens_matrix[i, j] = price_local if not np.isnan(price_local) else 0
        except Exception:
            sens_matrix[i, j] = 0

# plot heatmap
sens_df = pd.DataFrame(sens_matrix, index=[f"{tg*100:.2f}%" for tg in tg_grid], columns=[f"{wa*100:.2f}%" for wa in wacc_grid])
fig_heat = go.Figure(data=go.Heatmap(
    z=sens_df.values,
    x=sens_df.columns,
    y=sens_df.index,
    colorscale="Viridis",
    colorbar=dict(title="Implied $")
))
fig_heat.update_layout(template="plotly_dark", title="Sensitivity: Implied Price ($) — Terminal growth vs WACC", height=450)
st.plotly_chart(fig_heat, use_container_width=True)

# ---------------------------
# Comparables (simple)
# ---------------------------
st.subheader("Peer quick-comps")
peers = st.text_input("Peer tickers (comma separated)", value="MGM,LVS,MLCO")
peer_list = [p.strip().upper() for p in peers.split(",") if p.strip()]

if peer_list:
    peer_rows = []
    for p in peer_list:
        try:
            pinfo, pfin, pbs, pcf, _ = fetch_ticker(p)
            peer_rows.append({
                "Ticker": p,
                "Market Cap": safe_number(pinfo.get("marketCap")),
                "EV/EBITDA": safe_number(pinfo.get("enterpriseToEbitda")),
                "Trailing P/E": safe_number(pinfo.get("trailingPE"))
            })
        except Exception:
            peer_rows.append({"Ticker": p, "Market Cap": np.nan, "EV/EBITDA": np.nan, "Trailing P/E": np.nan})
    peer_table = pd.DataFrame(peer_rows).set_index("Ticker")
    # format numeric columns only
    numcols = peer_table.select_dtypes(include=[np.number]).columns
    fmt_map = {c: "{:,.2f}" for c in numcols}
    st.dataframe(peer_table.style.format(fmt_map), use_container_width=True)

# ---------------------------
# Footer / Caveats
# ---------------------------
st.markdown("---")
st.write("**Notes & limitations:** This tool is for exploratory analysis and education — not investment advice. Financial statement scraping via yfinance is best-effort; always verify with official filings (SEC / company investor releases). Project NPV here is simplified (no taxes, working capital, financing structure or JV agreements).")
