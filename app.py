# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date
import matplotlib.pyplot as plt

st.set_page_config(page_title="Wynn Resorts Valuation (WYNN)", page_icon="🎰", layout="wide")

# ----- Helper functions -----
def ticker_fetch(ticker):
    t = yf.Ticker(ticker)
    info = t.info
    # financial statements (pandas DataFrames)
    # Note: yfinance column/index orientation can vary; we'll try to normalize
    try:
        fin = t.financials.T  # income statement
    except Exception:
        fin = pd.DataFrame()
    try:
        bs = t.balance_sheet.T
    except Exception:
        bs = pd.DataFrame()
    try:
        cf = t.cashflow.T
    except Exception:
        cf = pd.DataFrame()
    return info, fin, bs, cf

def simple_dcf(fcf, g, d, tg, years=5):
    """
    fcf: most recent year FCF (float)
    g: growth rate (decimal) for explicit years
    d: discount rate (decimal)
    tg: terminal growth (decimal)
    returns: intrinsic enterprise value (float), list of projected discounted cash flows (for plotting)
    """
    pv = 0.0
    discounted_flow_list = []
    for i in range(1, years+1):
        projected = fcf * (1 + g)**i
        discounted = projected / (1 + d)**i
        discounted_flow_list.append(discounted)
        pv += discounted

    terminal_nominal = (fcf * (1 + g)**years) * (1 + tg) / (d - tg)
    discounted_terminal = terminal_nominal / (1 + d)**years
    ev = pv + discounted_terminal
    return ev, discounted_flow_list, discounted_terminal

def enterprise_value_from_info(info):
    # safe getters for market cap, total debt, cash
    market_cap = info.get("marketCap") or info.get("market_cap") or np.nan
    # yfinance may include totalDebt and totalCash
    debt = info.get("totalDebt", 0) or info.get("longTermDebt", 0) or 0
    cash = info.get("totalCash", 0) or info.get("cash", 0) or 0
    # Enterprise value approx:
    if market_cap and debt is not None and cash is not None:
        ev = market_cap + (debt or 0) - (cash or 0)
    else:
        ev = np.nan
    return {"market_cap": market_cap, "debt": debt, "cash": cash, "ev": ev}

def format_currency(x):
    if pd.isna(x): return "N/A"
    return f"${x:,.0f}"

# ----- Sidebar Inputs -----
st.sidebar.header("Model inputs & data source")
ticker = st.sidebar.text_input("Ticker", value="WYNN").upper()
use_live = st.sidebar.checkbox("Pull live data (yfinance)", value=True)
manual_override = st.sidebar.checkbox("Manual override of key inputs (useful for audit)", value=False)

# Valuation assumptions default
st.sidebar.subheader("DCF assumptions")
fcf_input = st.sidebar.number_input("Most recent Free Cash Flow (USD)", value=1_300_000_000.0, step=1e6, format="%.0f")
growth_rate = st.sidebar.number_input("Explicit growth rate (annual %) (next 5 yrs)", value=5.0, step=0.1)
discount_rate = st.sidebar.number_input("Discount Rate (WACC %) (decimal input shown as %)", value=10.0, step=0.1)
terminal_growth = st.sidebar.number_input("Terminal growth rate (%)", value=2.0, step=0.1)
shares_out_override = st.sidebar.number_input("Shares outstanding (if manual override)", value=112_078_263, format="%.0f")

st.sidebar.subheader("UAE project (Wynn Al Marjan) scenario")
st.sidebar.write("Model incremental impact of the Wynn Al Marjan Island project.")
project_capex = st.sidebar.number_input("Project capex (USD)", value=3_900_000_000.0, step=1e6, format="%.0f")
project_start_year = st.sidebar.number_input("Project opens (year)", value=2027, step=1)
project_incremental_fcf = st.sidebar.number_input("Estimated stabilized incremental FCF per year (USD)", value=600_000_000.0, step=1e6, format="%.0f")
project_stabilization_years = st.sidebar.number_input("Years to stabilization (after open)", value=2, step=1)

# ----- Data fetching -----
st.header(f"{ticker} — Company valuation dashboard")
if use_live:
    with st.spinner("Fetching data from yfinance..."):
        info, fin, bs, cf = ticker_fetch(ticker)
        ev_components = enterprise_value_from_info(info)
        # try to pull shares outstanding and current price
        market_price = info.get("currentPrice") or info.get("previousClose") or np.nan
        shares_out = info.get("sharesOutstanding") or info.get("floatShares") or np.nan
else:
    info, fin, bs, cf = {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    ev_components = {"market_cap": np.nan, "debt": np.nan, "cash": np.nan, "ev": np.nan}
    market_price = np.nan
    shares_out = np.nan

# If manual override, prefer input shares & fcf from the sidebar
if manual_override:
    fcf = float(fcf_input)
    shares = float(shares_out_override)
else:
    # try to get a proxy for FCF from cashflow statement:
    # yfinance cashflow DataFrame commonly has 'Total Cash From Operating Activities' and 'Capital Expenditures'
    fcf = None
    try:
        # take last (most recent) year row
        if not cf.empty:
            cf_latest = cf.iloc[0]  # yfinance often gives most recent as first row
            operating = cf_latest.get("Total Cash From Operating Activities") or cf_latest.get("Total cash from operating activities") or cf_latest.get("Operating Cash Flow") or np.nan
            capex = cf_latest.get("Capital Expenditures") or cf_latest.get("Capital Expenditure") or 0
            if pd.notna(operating):
                fcf = float(operating) + float(capex)  # note capex is typically negative, so OCF + CAPEX approximates FCF
    except Exception:
        fcf = None

    # fallback to sidebar input if yfinance didn't return FCF
    if fcf is None or pd.isna(fcf):
        fcf = float(fcf_input)

    shares = shares_out if (shares_out and not pd.isna(shares_out)) else float(shares_out_override)

# Display top-line facts
col1, col2, col3, col4 = st.columns(4)
col1.metric("Market Price (approx)", format_currency(market_price) if market_price else "N/A")
col2.metric("Market Cap", format_currency(ev_components.get("market_cap")))
col3.metric("Enterprise Value", format_currency(ev_components.get("ev")))
col4.metric("Shares Outstanding", f"{int(shares):,}" if shares else "N/A")

# ----- Financial statements display -----
st.subheader("Financial statements (pulled from yfinance)")
st.write("### Income statement (most recent years)")
if not fin.empty:
    st.dataframe(fin.fillna("").astype(object).style.format("{:,.0f}"))
else:
    st.info("Income statement not available from yfinance for this ticker in this session.")

st.write("### Balance sheet (most recent years)")
if not bs.empty:
    st.dataframe(bs.fillna("").astype(object).style.format("{:,.0f}"))
else:
    st.info("Balance sheet not available from yfinance for this ticker in this session.")

st.write("### Cash flow statement (most recent years)")
if not cf.empty:
    st.dataframe(cf.fillna("").astype(object).style.format("{:,.0f}"))
else:
    st.info("Cash flow statement not available from yfinance for this ticker in this session.")

# ----- DCF base case -----
st.subheader("Discounted Cash Flow (DCF) — base case")
g = float(growth_rate) / 100.0
d = float(discount_rate) / 100.0
tg = float(terminal_growth) / 100.0

base_ev, discounted_list, discounted_terminal = simple_dcf(fcf=float(fcf), g=g, d=d, tg=tg, years=5)
st.write(f"Using base FCF = {format_currency(fcf)}, explicit growth = {g*100:.2f}%, discount = {d*100:.2f}%, terminal growth = {tg*100:.2f}%")
st.write(f"Estimated enterprise value (EV, base DCF): **{format_currency(base_ev)}**")
equity_value = base_ev - (ev_components.get("debt") or 0) + (ev_components.get("cash") or 0)
price_per_share = equity_value / shares if shares and equity_value and not pd.isna(equity_value) else np.nan
st.write(f"Implied equity value (EV - net debt): **{format_currency(equity_value)}**")
st.write(f"Implied price per share: **{format_currency(price_per_share)}**")

# show contributions plot
fig, ax = plt.subplots(figsize=(8,3))
years = list(range(1,6))
proj_fcfs = [fcf*(1+g)**i for i in years]
ax.plot(years, proj_fcfs, marker="o", label="Projected nominal FCF")
ax.bar(years, discounted_list, alpha=0.6, label="Discounted FCF (PV)")
ax.set_xlabel("Year")
ax.set_ylabel("USD")
ax.set_title("Projected FCF (nominal) and Discounted PV contributions")
ax.legend()
st.pyplot(fig)

# ----- UAE Project scenario -----
st.subheader("UAE: Wynn Al Marjan project scenario impact")
st.write("Model: we treat the project as an additional investment (capex) that produces incremental stabilized FCF after opening & ramping.")

# Build incremental FCF schedule from project assumptions
current_year = date.today().year
open_year = int(project_start_year)
ramp_years = int(project_stabilization_years)
inc_fcf_schedule = []
discounted_inc = 0.0
for year in range(1, 11):  # project horizon 10 years (user can change)
    cal_year = current_year + year
    if cal_year < open_year:
        inc = 0.0
    elif cal_year == open_year:
        inc = project_incremental_fcf * 0.25  # assume partial year ramp (25%) — adjustable later
    elif open_year < cal_year <= open_year + ramp_years:
        # linear ramp to stabilization
        ramp_progress = (cal_year - open_year) / (ramp_years + 0.0)
        inc = project_incremental_fcf * max(0.25, ramp_progress)  # minimum 25% in open year
    else:
        inc = project_incremental_fcf
    inc_fcf_schedule.append((cal_year, inc))
    # discount to present using d
    years_from_now = cal_year - current_year
    discounted_inc += inc / ((1 + d) ** years_from_now)

# Net present value of project FCF (not subtracting project capex)
npv_project_cashflows = discounted_inc
npv_project_net = npv_project_cashflows - project_capex  # simple NPV subtracting capex now
st.write(f"NPV of incremental (discounted) FCF over next 10 years: **{format_currency(npv_project_cashflows)}**")
st.write(f"Less project capex = {format_currency(project_capex)} -> Net NPV = **{format_currency(npv_project_net)}**")
st.write("If Net NPV > 0, the project adds value to enterprise value (ignoring funding costs, dilution, taxes).")

# Show schedule
proj_df = pd.DataFrame(inc_fcf_schedule, columns=["Year", "Estimated incremental FCF"])
proj_df["Discounted PV"] = [v / ((1 + d) ** (int(year)-current_year)) for year, v in inc_fcf_schedule]
proj_df["Estimated incremental FCF"] = proj_df["Estimated incremental FCF"].map(lambda x: f"${x:,.0f}")
proj_df["Discounted PV"] = proj_df["Discounted PV"].map(lambda x: f"${x:,.0f}")
st.table(proj_df)

# Add project value to DCF EV and recompute implied price
ev_with_project = base_ev + npv_project_cashflows - project_capex  # (this is a simplified approach)
equity_with_project = ev_with_project - (ev_components.get("debt") or 0) + (ev_components.get("cash") or 0)
price_with_project = equity_with_project / shares if shares and equity_with_project and not pd.isna(equity_with_project) else np.nan

st.write("### Valuation summary (simplified)")
st.write(f"Base EV (DCF): {format_currency(base_ev)}")
st.write(f"Project NPV (discounted incremental FCF): {format_currency(npv_project_cashflows)}")
st.write(f"Combined EV (base + project NPV - capex): {format_currency(ev_with_project)}")
st.write(f"Implied price per share (with project): **{format_currency(price_with_project)}**")
st.metric("Price uplift vs base implied price", value=f"{((price_with_project/price_per_share-1)*100 if price_per_share and price_with_project else 0):.1f}%")

# ----- Comparable companies (simple multiples) -----
st.subheader("Comparable analysis (simple EV/EBITDA approach)")
peer_tickers = ["MGM", "LVS", "CZR"]
peers = {}
for p in peer_tickers:
    info_p, fin_p, bs_p, cf_p = ticker_fetch(p)
    peers[p] = enterprise_value_from_info(info_p)
    # try to use trailing EBITDA from Yahoo via info field 'ebitda'
    peers[p]["ebitda"] = info_p.get("ebitda") or np.nan

# create peers table
peers_df = pd.DataFrame(peers).T
peers_df["EV"] = peers_df["ev"].map(lambda x: format_currency(x) if x and not pd.isna(x) else "N/A")
peers_df["Debt"] = peers_df["debt"].map(lambda x: format_currency(x) if x and not pd.isna(x) else "N/A")
peers_df["Cash"] = peers_df["cash"].map(lambda x: format_currency(x) if x and not pd.isna(x) else "N/A")
peers_df["EBITDA"] = peers_df["ebitda"].map(lambda x: format_currency(x) if x and not pd.isna(x) else "N/A")
st.table(peers_df[["EV","Debt","Cash","EBITDA"]])

# compute median EV/EBITDA of peers (where possible)
ev_ebitda_ratios = []
for p,v in peers.items():
    ev = v.get("ev")
    ebitda = v.get("ebitda")
    if ev and ebitda and ebitda>0:
        ev_ebitda_ratios.append(ev / ebitda)
median_ev_ebitda = np.median(ev_ebitda_ratios) if ev_ebitda_ratios else np.nan
st.write(f"Median peer EV/EBITDA: **{median_ev_ebitda:.2f}**" if not pd.isna(median_ev_ebitda) else "EV/EBITDA median not available")

# Implied EV using peer median (use WYNN ebitda if available)
wynn_ebitda = info.get("ebitda") if info else None
if wynn_ebitda and not pd.isna(median_ev_ebitda):
    implied_ev_from_peers = median_ev_ebitda * wynn_ebitda
    implied_equity = implied_ev_from_peers - (ev_components.get("debt") or 0) + (ev_components.get("cash") or 0)
    implied_price = implied_equity / shares if shares else np.nan
    st.write(f"Implied price per share from peer median EV/EBITDA: **{format_currency(implied_price)}**")
else:
    st.info("WYNN EBITDA or peer median missing. Cannot compute implied price from peers.")

# ----- Limitations & next steps -----
st.subheader("Notes, limitations & next steps")
st.markdown("""
- This model is a simplified professional-style analysis for educational and exploratory use only.
- Data pulled via `yfinance` — verify with official filings (SEC / company investor site) for any professional use.  
- Project modelling here is a simplified NPV of incremental FCF — in practice you'd model taxes, depreciation, financing (debt/equity), incremental working capital, and schedule of capex.  
- Next steps: add sensitivity tables, WACC calc from capital structure, tax assumptions, and run Monte Carlo scenario analysis.
""")

# Footer: cite live sources
st.write("**Sources / references:** yfinance (market & financials), Wynn investor releases & 10-K filings; press reporting on Wynn Al Marjan (UAE).")
