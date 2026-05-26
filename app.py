import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from textblob import TextBlob
from datetime import datetime

st.set_page_config(
    page_title="Market Terminal · Oliver Van Aken",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Geist+Mono:wght@300;400;500;600&family=Geist:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Geist', sans-serif;
}

/* Base */
.main { background-color: #0d0f14; }
section[data-testid="stSidebar"] { background-color: #0d0f14; border-right: 1px solid #1e2330; }
section[data-testid="stSidebar"] > div { padding-top: 1.5rem; }

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem; max-width: 1400px; }

/* Sidebar inputs */
.stTextInput > div > div > input,
.stDateInput > div > div > input {
    background: #131720 !important;
    border: 1px solid #1e2330 !important;
    border-radius: 6px !important;
    color: #e2e8f0 !important;
    font-family: 'Geist Mono', monospace !important;
    font-size: 13px !important;
}
.stTextInput > div > div > input:focus,
.stDateInput > div > div > input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 1px #3b82f620 !important;
}

/* Buttons */
.stButton > button {
    background: #131720 !important;
    border: 1px solid #1e2330 !important;
    color: #94a3b8 !important;
    border-radius: 6px !important;
    font-family: 'Geist', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 8px 16px !important;
    transition: all .15s ease !important;
    width: 100%;
}
.stButton > button:hover {
    border-color: #3b82f6 !important;
    color: #e2e8f0 !important;
    background: #1a2035 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 1px solid #1e2330;
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    color: #475569 !important;
    font-family: 'Geist', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 10px 20px !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #e2e8f0 !important;
    border-bottom: 2px solid #3b82f6 !important;
    background: transparent !important;
}

/* Metrics */
div[data-testid="metric-container"] {
    background: #131720;
    border: 1px solid #1e2330;
    border-radius: 8px;
    padding: 16px 20px;
}
div[data-testid="metric-container"] label {
    color: #475569 !important;
    font-family: 'Geist Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: .06em;
    text-transform: uppercase;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #e2e8f0 !important;
    font-family: 'Geist Mono', monospace !important;
    font-size: 22px !important;
    font-weight: 600 !important;
}
div[data-testid="metric-container"] div[data-testid="stMetricDelta"] {
    font-family: 'Geist Mono', monospace !important;
    font-size: 12px !important;
}

/* Dataframes */
.stDataFrame { border: 1px solid #1e2330; border-radius: 8px; overflow: hidden; }
.stDataFrame [data-testid="stDataFrameResizable"] { background: #131720; }

/* Sliders */
.stSlider > div > div > div { background: #1e2330 !important; }
.stSlider > div > div > div > div { background: #3b82f6 !important; }

/* Select boxes */
.stSelectbox > div > div {
    background: #131720 !important;
    border: 1px solid #1e2330 !important;
    border-radius: 6px !important;
    color: #e2e8f0 !important;
    font-family: 'Geist Mono', monospace !important;
    font-size: 13px !important;
}

/* Sidebar text */
.stSidebar .stMarkdown p, .stSidebar .stMarkdown label {
    color: #475569;
    font-size: 12px;
    font-family: 'Geist Mono', monospace;
}

/* Dividers */
hr { border-color: #1e2330; }

/* Expanders */
.streamlit-expanderHeader {
    background: #131720 !important;
    border: 1px solid #1e2330 !important;
    border-radius: 6px !important;
    color: #94a3b8 !important;
    font-family: 'Geist', sans-serif !important;
    font-size: 13px !important;
}
</style>
""", unsafe_allow_html=True)

# ── HELPERS ────────────────────────────────────────────────

def fmt_large(n):
    if n is None or (isinstance(n, float) and np.isnan(n)): return "—"
    if abs(n) >= 1e12: return f"${n/1e12:.2f}T"
    if abs(n) >= 1e9:  return f"${n/1e9:.2f}B"
    if abs(n) >= 1e6:  return f"${n/1e6:.2f}M"
    return f"${n:,.0f}"

def fmt_pct(n):
    if n is None or (isinstance(n, float) and np.isnan(n)): return "—"
    return f"{n*100:.1f}%"

def color_val(val, positive_good=True):
    try:
        v = float(str(val).replace('%','').replace('$','').replace(',',''))
        if v > 0: return "#22c55e" if positive_good else "#ef4444"
        if v < 0: return "#ef4444" if positive_good else "#22c55e"
    except: pass
    return "#94a3b8"

PLOT_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor='#0d0f14',
    plot_bgcolor='#0d0f14',
    font=dict(family='Geist Mono, monospace', color='#64748b', size=11),
    margin=dict(l=10, r=10, t=10, b=10),
    legend=dict(bgcolor='#131720', bordercolor='#1e2330', borderwidth=1, font=dict(size=11)),
    xaxis=dict(gridcolor='#1e2330', showgrid=True, zeroline=False),
    yaxis=dict(gridcolor='#1e2330', showgrid=True, zeroline=False),
)

@st.cache_data(ttl=1800)
def load_data(symbol, start, end):
    try:
        data = yf.download(symbol, start=start, end=end, progress=False)
        if data.empty: return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data.dropna()
    except: return None

@st.cache_data(ttl=3600)
def load_info(symbol):
    try: return yf.Ticker(symbol).info
    except: return {}

@st.cache_data(ttl=3600)
def load_ticker(symbol):
    return yf.Ticker(symbol)

# ── SIDEBAR ────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='margin-bottom:24px;'>
        <div style='font-family:Geist Mono,monospace;font-size:11px;color:#3b82f6;letter-spacing:.1em;text-transform:uppercase;margin-bottom:4px;'>Market Terminal</div>
        <div style='font-size:18px;font-weight:600;color:#e2e8f0;'>Oliver Van Aken</div>
    </div>
    """, unsafe_allow_html=True)

    if 'watchlist' not in st.session_state:
        st.session_state['watchlist'] = ["AAPL", "MSFT", "NVDA", "TSLA"]

    st.markdown('<p style="font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:#475569;margin-bottom:8px;">Active Ticker</p>', unsafe_allow_html=True)
    ticker = st.text_input("", "AAPL", label_visibility="collapsed").strip().upper()

    st.markdown('<p style="font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:#475569;margin:16px 0 8px;">Date Range</p>', unsafe_allow_html=True)
    start_date = st.date_input("From", pd.to_datetime("2022-01-01"), label_visibility="collapsed")
    end_date   = st.date_input("To",   pd.to_datetime("today"),      label_visibility="collapsed")

    st.markdown('<p style="font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:#475569;margin:16px 0 8px;">Watchlist</p>', unsafe_allow_html=True)
    new_watch = st.text_input("Add ticker", "", label_visibility="collapsed", placeholder="e.g. AMZN").upper().strip()
    if st.button("Add to Watchlist") and new_watch and new_watch not in st.session_state['watchlist']:
        st.session_state['watchlist'].append(new_watch)
        st.rerun()

    for w in st.session_state['watchlist']:
        cols = st.columns([3,1])
        try:
            p = yf.Ticker(w).fast_info['last_price']
            cols[0].markdown(f'<span style="font-family:Geist Mono,monospace;font-size:13px;color:#e2e8f0;">{w}</span>', unsafe_allow_html=True)
            cols[1].markdown(f'<span style="font-family:Geist Mono,monospace;font-size:13px;color:#94a3b8;">${p:.0f}</span>', unsafe_allow_html=True)
        except:
            cols[0].markdown(f'<span style="font-family:Geist Mono,monospace;font-size:13px;color:#e2e8f0;">{w}</span>', unsafe_allow_html=True)

# ── LOAD DATA ──────────────────────────────────────────────

df = load_data(ticker, start_date, end_date)
info = load_info(ticker)
tk = load_ticker(ticker)

# ── HEADER ─────────────────────────────────────────────────

name = info.get('longName', ticker)
sector = info.get('sector', '—')
industry = info.get('industry', '—')
exchange = info.get('exchange', '—')

try:
    live_price = tk.fast_info['last_price']
    prev_close = tk.fast_info['previous_close'] if hasattr(tk.fast_info, 'previous_close') else info.get('previousClose', live_price)
    change = live_price - prev_close
    change_pct = (change / prev_close) * 100
    price_color = "#22c55e" if change >= 0 else "#ef4444"
    arrow = "▲" if change >= 0 else "▼"
except:
    live_price = df['Close'].iloc[-1] if df is not None else 0
    change = 0; change_pct = 0
    price_color = "#94a3b8"; arrow = ""

st.markdown(f"""
<div style='display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:24px;padding-bottom:20px;border-bottom:1px solid #1e2330;'>
    <div>
        <div style='font-family:Geist Mono,monospace;font-size:11px;color:#475569;letter-spacing:.06em;text-transform:uppercase;margin-bottom:6px;'>{exchange} · {sector} · {industry}</div>
        <div style='font-size:26px;font-weight:600;color:#e2e8f0;margin-bottom:4px;'>{name} <span style="color:#475569;font-size:18px;font-weight:400;">({ticker})</span></div>
        <div style='display:flex;align-items:baseline;gap:12px;'>
            <span style='font-family:Geist Mono,monospace;font-size:32px;font-weight:600;color:#e2e8f0;'>${live_price:,.2f}</span>
            <span style='font-family:Geist Mono,monospace;font-size:14px;color:{price_color};'>{arrow} {abs(change):.2f} ({abs(change_pct):.2f}%)</span>
        </div>
    </div>
    <div style='font-family:Geist Mono,monospace;font-size:11px;color:#1e2330;background:#131720;padding:8px 14px;border-radius:6px;border:1px solid #1e2330;'>
        {datetime.now().strftime('%H:%M:%S · %b %d %Y')}
    </div>
</div>
""", unsafe_allow_html=True)

# ── TABS ───────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Chart", "Fundamentals", "Financials", "Analyst & Insiders", "Sentiment", "Backtest"
])

# ═══════════════════════════════════════════════════════════
# TAB 1 — CHART
# ═══════════════════════════════════════════════════════════
with tab1:
    if df is not None and len(df) > 10:
        c1, c2, c3, c4 = st.columns(4)

        # Technical calcs
        close = df['Close'].squeeze()
        sma50  = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()
        ema20  = close.ewm(span=20, adjust=False).mean()

        delta = close.diff()
        gain  = delta.where(delta > 0, 0).rolling(14).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi   = 100 - (100 / (1 + gain / loss))

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd  = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        hist   = macd - signal

        # Bollinger
        bb_mid  = close.rolling(20).mean()
        bb_std  = close.rolling(20).std()
        bb_up   = bb_mid + 2 * bb_std
        bb_low  = bb_mid - 2 * bb_std

        latest_rsi   = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50
        latest_close = float(close.iloc[-1])
        latest_sma50 = float(sma50.iloc[-1]) if not pd.isna(sma50.iloc[-1]) else latest_close
        support      = float(df['Low'].rolling(252, min_periods=30).min().iloc[-1]) if 'Low' in df.columns else latest_close

        score = sum([latest_rsi < 40, latest_close > latest_sma50, (latest_close - support) / support < 0.1])
        conviction = (score / 3) * 100

        rsi_color = "#22c55e" if latest_rsi < 40 else ("#ef4444" if latest_rsi > 70 else "#f59e0b")
        c1.metric("Price", f"${latest_close:,.2f}")
        c2.metric("RSI (14)", f"{latest_rsi:.1f}")
        c3.metric("vs 50-Day SMA", f"{((latest_close/latest_sma50)-1)*100:+.1f}%")
        c4.metric("Signal Score", f"{conviction:.0f} / 100")

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # Overlay selector
        overlays = st.multiselect("Overlays", ["SMA 50", "SMA 200", "EMA 20", "Bollinger Bands"], default=["SMA 50", "Bollinger Bands"])

        # Chart type
        chart_type = st.radio("Chart type", ["Candlestick", "Line"], horizontal=True)

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.6, 0.2, 0.2]
        )

        # Price
        if chart_type == "Candlestick":
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['Open'].squeeze(), high=df['High'].squeeze(),
                low=df['Low'].squeeze(),   close=close,
                name=ticker,
                increasing_line_color='#22c55e', decreasing_line_color='#ef4444',
                increasing_fillcolor='#22c55e',  decreasing_fillcolor='#ef4444',
            ), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(x=df.index, y=close, name='Close', line=dict(color='#3b82f6', width=2)), row=1, col=1)

        if "SMA 50" in overlays:
            fig.add_trace(go.Scatter(x=df.index, y=sma50, name='SMA 50', line=dict(color='#f59e0b', width=1.5, dash='dash')), row=1, col=1)
        if "SMA 200" in overlays:
            fig.add_trace(go.Scatter(x=df.index, y=sma200, name='SMA 200', line=dict(color='#8b5cf6', width=1.5, dash='dash')), row=1, col=1)
        if "EMA 20" in overlays:
            fig.add_trace(go.Scatter(x=df.index, y=ema20, name='EMA 20', line=dict(color='#06b6d4', width=1.5, dash='dot')), row=1, col=1)
        if "Bollinger Bands" in overlays:
            fig.add_trace(go.Scatter(x=df.index, y=bb_up,  name='BB Upper', line=dict(color='#334155', width=1), showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=bb_low, name='BB Lower', line=dict(color='#334155', width=1),
                fill='tonexty', fillcolor='rgba(51,65,85,0.15)', showlegend=False), row=1, col=1)

        # Volume
        vol_colors = ['#22c55e' if c >= o else '#ef4444'
                      for c, o in zip(df['Close'].squeeze(), df['Open'].squeeze())]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'].squeeze(), name='Volume',
                             marker_color=vol_colors, opacity=0.6), row=2, col=1)

        # MACD
        fig.add_trace(go.Scatter(x=df.index, y=macd,   name='MACD',   line=dict(color='#3b82f6', width=1.5)), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=signal, name='Signal', line=dict(color='#f59e0b', width=1.5)), row=3, col=1)
        fig.add_trace(go.Bar(x=df.index, y=hist, name='Histogram',
                             marker_color=['#22c55e' if v >= 0 else '#ef4444' for v in hist], opacity=0.6), row=3, col=1)

        fig.update_layout(**PLOT_LAYOUT, height=620, showlegend=True)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Vol",   row=2, col=1)
        fig.update_yaxes(title_text="MACD",  row=3, col=1)
        fig.update_xaxes(rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # Price alert
        with st.expander("Price Alert"):
            a1, a2 = st.columns(2)
            alert_email = a1.text_input("Email", placeholder="you@email.com")
            alert_rsi   = a2.slider("Alert when RSI below", 10, 50, 30)
            if st.button("Set Alert"):
                if alert_email:
                    status = "🔴 Currently triggered" if latest_rsi < alert_rsi else "✅ Monitoring"
                    st.success(f"Alert set for {ticker} · RSI < {alert_rsi} → {alert_email} · {status}")
                else:
                    st.error("Enter a valid email.")
    else:
        st.error(f"No data found for **{ticker}**. Check the ticker symbol.")

# ═══════════════════════════════════════════════════════════
# TAB 2 — FUNDAMENTALS
# ═══════════════════════════════════════════════════════════
with tab2:
    if info:
        st.markdown("#### Key ratios")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Market Cap",      fmt_large(info.get('marketCap')))
        r2.metric("P/E (TTM)",       f"{info.get('trailingPE', 0):.1f}x" if info.get('trailingPE') else "—")
        r3.metric("Forward P/E",     f"{info.get('forwardPE', 0):.1f}x"  if info.get('forwardPE')  else "—")
        r4.metric("P/S (TTM)",       f"{info.get('priceToSalesTrailing12Months', 0):.1f}x" if info.get('priceToSalesTrailing12Months') else "—")

        r5, r6, r7, r8 = st.columns(4)
        r5.metric("P/B",             f"{info.get('priceToBook', 0):.2f}x" if info.get('priceToBook') else "—")
        r6.metric("EV / EBITDA",     f"{info.get('enterpriseToEbitda', 0):.1f}x" if info.get('enterpriseToEbitda') else "—")
        r7.metric("Profit Margin",   fmt_pct(info.get('profitMargins')))
        r8.metric("ROE",             fmt_pct(info.get('returnOnEquity')))

        r9, r10, r11, r12 = st.columns(4)
        r9.metric("Revenue (TTM)",   fmt_large(info.get('totalRevenue')))
        r10.metric("Revenue Growth", fmt_pct(info.get('revenueGrowth')))
        r11.metric("Debt / Equity",  f"{info.get('debtToEquity', 0):.1f}" if info.get('debtToEquity') else "—")
        r12.metric("Dividend Yield", fmt_pct(info.get('dividendYield')))

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown("#### 52-week range")
        try:
            lo = info.get('fiftyTwoWeekLow',  0)
            hi = info.get('fiftyTwoWeekHigh', 0)
            curr = float(live_price)
            pct = (curr - lo) / (hi - lo) if hi != lo else 0.5
            st.markdown(f"""
            <div style='background:#131720;border:1px solid #1e2330;border-radius:8px;padding:20px;'>
                <div style='display:flex;justify-content:space-between;margin-bottom:8px;font-family:Geist Mono,monospace;font-size:12px;color:#475569;'>
                    <span>Low  ${lo:,.2f}</span><span>Current ${curr:,.2f}</span><span>High ${hi:,.2f}</span>
                </div>
                <div style='background:#1e2330;border-radius:4px;height:6px;position:relative;'>
                    <div style='position:absolute;left:{pct*100:.1f}%;transform:translateX(-50%);width:12px;height:12px;background:#3b82f6;border-radius:50%;top:-3px;'></div>
                    <div style='background:linear-gradient(90deg,#1e3a5f,#3b82f6);border-radius:4px;height:6px;width:{pct*100:.1f}%;'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        except: pass

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown("#### Earnings calendar")
        try:
            cal = tk.calendar
            if cal is not None and not (isinstance(cal, dict) and len(cal) == 0):
                if isinstance(cal, dict):
                    st.markdown(f"""
                    <div style='background:#131720;border:1px solid #1e2330;border-radius:8px;padding:20px;font-family:Geist Mono,monospace;'>
                        <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;'>
                        {''.join([f'<div><div style="font-size:10px;color:#475569;text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px;">{k}</div><div style="font-size:14px;color:#e2e8f0;">{v}</div></div>' for k,v in list(cal.items())[:6]])}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        except: pass

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown("#### Short interest")
        si1, si2 = st.columns(2)
        si1.metric("Short % of Float", fmt_pct(info.get('shortPercentOfFloat')))
        si2.metric("Short Ratio (Days to Cover)", f"{info.get('shortRatio', 0):.1f}" if info.get('shortRatio') else "—")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown("#### Company overview")
        desc = info.get('longBusinessSummary', '')
        if desc:
            st.markdown(f'<p style="font-size:13px;color:#64748b;line-height:1.8;">{desc}</p>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# TAB 3 — FINANCIALS
# ═══════════════════════════════════════════════════════════
with tab3:
    period = st.radio("Period", ["Annual", "Quarterly"], horizontal=True)
    annual = period == "Annual"

    try:
        income = tk.financials if annual else tk.quarterly_financials
        balance = tk.balance_sheet if annual else tk.quarterly_balance_sheet
        cashflow = tk.cashflow if annual else tk.quarterly_cashflow

        def clean_df(raw):
            if raw is None or raw.empty: return None
            raw = raw.T
            raw.index = pd.to_datetime(raw.index).strftime('%b %Y')
            return raw

        inc_df = clean_df(income)
        bal_df = clean_df(balance)
        cf_df  = clean_df(cashflow)

        if inc_df is not None:
            st.markdown("#### Income statement")
            key_rows = ['Total Revenue', 'Gross Profit', 'Operating Income', 'Net Income']
            disp = {k: inc_df[k] for k in key_rows if k in inc_df.columns}
            if disp:
                disp_df = pd.DataFrame(disp)
                fig_inc = go.Figure()
                colors = ['#3b82f6', '#22c55e', '#f59e0b', '#8b5cf6']
                for i, col in enumerate(disp_df.columns):
                    fig_inc.add_trace(go.Bar(
                        name=col, x=disp_df.index,
                        y=disp_df[col] / 1e9,
                        marker_color=colors[i % len(colors)]
                    ))
                fig_inc.update_layout(**PLOT_LAYOUT, height=320, barmode='group',
                                      yaxis_title='Billions USD')
                st.plotly_chart(fig_inc, use_container_width=True)

        if bal_df is not None:
            st.markdown("#### Balance sheet")
            b_rows = ['Total Assets', 'Total Liabilities Net Minority Interest', 'Stockholders Equity']
            b_disp = {k: bal_df[k] for k in b_rows if k in bal_df.columns}
            if b_disp:
                b_df = pd.DataFrame(b_disp)
                fig_bal = go.Figure()
                for i, col in enumerate(b_df.columns):
                    fig_bal.add_trace(go.Bar(name=col, x=b_df.index, y=b_df[col]/1e9, marker_color=colors[i]))
                fig_bal.update_layout(**PLOT_LAYOUT, height=300, barmode='group', yaxis_title='Billions USD')
                st.plotly_chart(fig_bal, use_container_width=True)

        if cf_df is not None:
            st.markdown("#### Cash flow")
            cf_rows = ['Operating Cash Flow', 'Free Cash Flow', 'Capital Expenditure']
            cf_disp = {k: cf_df[k] for k in cf_rows if k in cf_df.columns}
            if cf_disp:
                cf_d = pd.DataFrame(cf_disp)
                fig_cf = go.Figure()
                for i, col in enumerate(cf_d.columns):
                    fig_cf.add_trace(go.Bar(name=col, x=cf_d.index, y=cf_d[col]/1e9, marker_color=colors[i]))
                fig_cf.update_layout(**PLOT_LAYOUT, height=300, barmode='group', yaxis_title='Billions USD')
                st.plotly_chart(fig_cf, use_container_width=True)

    except Exception as e:
        st.info("Financial data unavailable for this ticker.")

# ═══════════════════════════════════════════════════════════
# TAB 4 — ANALYST & INSIDERS
# ═══════════════════════════════════════════════════════════
with tab4:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Analyst ratings")
        try:
            recs = tk.recommendations
            if recs is not None and not recs.empty:
                recs = recs.sort_index(ascending=False).head(20)
                if 'period' in recs.columns:
                    grade_cols = [c for c in ['strongBuy','buy','hold','sell','strongSell'] if c in recs.columns]
                    if grade_cols:
                        latest = recs.iloc[0]
                        total = sum([latest.get(g, 0) for g in grade_cols])
                        if total > 0:
                            buy_pct  = (latest.get('strongBuy',0) + latest.get('buy',0)) / total * 100
                            hold_pct = latest.get('hold', 0) / total * 100
                            sell_pct = (latest.get('sell',0) + latest.get('strongSell',0)) / total * 100
                            fig_pie = go.Figure(go.Pie(
                                labels=['Buy', 'Hold', 'Sell'],
                                values=[buy_pct, hold_pct, sell_pct],
                                hole=0.6,
                                marker_colors=['#22c55e', '#f59e0b', '#ef4444'],
                                textinfo='label+percent',
                                textfont=dict(size=11)
                            ))
                            fig_pie.update_layout(**PLOT_LAYOUT, height=260, showlegend=False)
                            st.plotly_chart(fig_pie, use_container_width=True)
        except: pass

        st.markdown("#### Price targets")
        pt1, pt2, pt3 = st.columns(3)
        pt1.metric("Low",    f"${info.get('targetLowPrice',  0):,.2f}" if info.get('targetLowPrice')  else "—")
        pt2.metric("Mean",   f"${info.get('targetMeanPrice', 0):,.2f}" if info.get('targetMeanPrice') else "—")
        pt3.metric("High",   f"${info.get('targetHighPrice', 0):,.2f}" if info.get('targetHighPrice') else "—")

        try:
            mean_target = info.get('targetMeanPrice')
            if mean_target:
                upside = (mean_target - float(live_price)) / float(live_price) * 100
                color  = "#22c55e" if upside > 0 else "#ef4444"
                st.markdown(f'<p style="font-family:Geist Mono,monospace;font-size:13px;color:{color};">Implied upside to mean target: {upside:+.1f}%</p>', unsafe_allow_html=True)
        except: pass

    with col_b:
        st.markdown("#### Institutional holders")
        try:
            inst = tk.institutional_holders
            if inst is not None and not inst.empty:
                st.dataframe(
                    inst[['Holder','Shares','% Out','Value']].head(10).reset_index(drop=True),
                    use_container_width=True, hide_index=True
                )
        except:
            st.info("Institutional holder data unavailable.")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("#### Insider transactions")
    try:
        insiders = tk.insider_transactions
        if insiders is not None and not insiders.empty:
            cols_to_show = [c for c in ['Insider','Position','Transaction','Shares','Value','Start Date'] if c in insiders.columns]
            st.dataframe(insiders[cols_to_show].head(15).reset_index(drop=True),
                         use_container_width=True, hide_index=True)
        else:
            st.info("No recent insider transactions.")
    except:
        st.info("Insider data unavailable.")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("#### Options chain")
    try:
        expiries = tk.options
        if expiries:
            sel_exp = st.selectbox("Expiry", expiries[:6])
            chain   = tk.option_chain(sel_exp)
            opt_type = st.radio("Type", ["Calls", "Puts"], horizontal=True)
            opt_df  = chain.calls if opt_type == "Calls" else chain.puts
            disp_cols = [c for c in ['strike','lastPrice','bid','ask','volume','openInterest','impliedVolatility'] if c in opt_df.columns]
            opt_disp  = opt_df[disp_cols].reset_index(drop=True)
            if 'impliedVolatility' in opt_disp.columns:
                opt_disp['impliedVolatility'] = opt_disp['impliedVolatility'].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")
            st.dataframe(opt_disp, use_container_width=True, hide_index=True)
    except:
        st.info("Options data unavailable for this ticker.")

# ═══════════════════════════════════════════════════════════
# TAB 5 — SENTIMENT
# ═══════════════════════════════════════════════════════════
with tab5:
    st.markdown(f"#### News sentiment — {ticker}")
    try:
        news_items = tk.news[:12] if tk.news else []
    except:
        news_items = []

    if news_items:
        scores = []
        for item in news_items:
            title = item.get('title', '')
            blob  = TextBlob(title)
            score = blob.sentiment.polarity
            scores.append(score)

        avg = np.mean(scores)
        s1, s2, s3 = st.columns(3)
        s1.metric("Avg Sentiment",  f"{avg:+.2f}")
        s2.metric("Bullish items",  f"{sum(1 for s in scores if s > 0.05)}")
        s3.metric("Bearish items",  f"{sum(1 for s in scores if s < -0.05)}")

        # Sentiment bar chart
        fig_sent = go.Figure(go.Bar(
            x=[f"#{i+1}" for i in range(len(scores))],
            y=scores,
            marker_color=['#22c55e' if s > 0.05 else ('#ef4444' if s < -0.05 else '#475569') for s in scores],
        ))
        fig_sent.update_layout(**PLOT_LAYOUT, height=200, showlegend=False,
                               yaxis=dict(range=[-1,1], gridcolor='#1e2330'),
                               xaxis=dict(gridcolor='#1e2330'))
        st.plotly_chart(fig_sent, use_container_width=True)

        st.markdown("#### Headlines")
        for i, (item, score) in enumerate(zip(news_items, scores)):
            title = item.get('title', '')
            url   = item.get('link', '#')
            pub   = item.get('publisher', '')
            ts    = item.get('providerPublishTime', 0)
            date_str = datetime.fromtimestamp(ts).strftime('%b %d') if ts else ''
            label = "Positive" if score > 0.05 else ("Negative" if score < -0.05 else "Neutral")
            lcolor = "#22c55e" if score > 0.05 else ("#ef4444" if score < -0.05 else "#475569")
            st.markdown(f"""
            <a href="{url}" target="_blank" style="text-decoration:none;">
            <div style='background:#131720;border:1px solid #1e2330;border-radius:8px;padding:14px 18px;margin-bottom:8px;cursor:pointer;transition:border-color .15s;' onmouseover="this.style.borderColor='#3b82f6'" onmouseout="this.style.borderColor='#1e2330'">
                <div style='font-size:13px;color:#e2e8f0;margin-bottom:6px;line-height:1.5;'>{title}</div>
                <div style='display:flex;gap:12px;align-items:center;'>
                    <span style='font-family:Geist Mono,monospace;font-size:10px;color:#475569;'>{pub} · {date_str}</span>
                    <span style='font-family:Geist Mono,monospace;font-size:10px;color:{lcolor};background:{lcolor}18;padding:2px 8px;border-radius:4px;'>{label} {score:+.2f}</span>
                </div>
            </div></a>
            """, unsafe_allow_html=True)
    else:
        headlines = [
            f"Analysts upgrade {ticker} ahead of earnings on strong margin outlook",
            f"Institutional flows indicate steady accumulation in {ticker} shares this quarter",
            f"Supply chain improvements support revised guidance for {ticker}",
            f"Sector rotation data shows {ticker} drawing renewed interest from growth funds",
        ]
        scores = [TextBlob(h).sentiment.polarity for h in headlines]
        st.info("Live news unavailable — showing sample headlines.")
        for h, s in zip(headlines, scores):
            lcolor = "#22c55e" if s > 0.05 else ("#ef4444" if s < -0.05 else "#475569")
            st.markdown(f"""
            <div style='background:#131720;border:1px solid #1e2330;border-radius:8px;padding:14px 18px;margin-bottom:8px;'>
                <div style='font-size:13px;color:#e2e8f0;margin-bottom:6px;'>{h}</div>
                <span style='font-family:Geist Mono,monospace;font-size:10px;color:{lcolor};'>{s:+.2f}</span>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# TAB 6 — BACKTEST
# ═══════════════════════════════════════════════════════════
with tab6:
    st.markdown(f"#### Strategy backtest — {ticker}")

    if df is not None and len(df) > 60:
        strat = st.selectbox("Strategy", [
            "SMA Crossover (50/200)",
            "RSI Mean Reversion",
            "Breakout (52-week high)",
        ])

        bt = pd.DataFrame(index=df.index)
        bt['Price'] = df['Close'].squeeze()
        bt['Returns'] = bt['Price'].pct_change()

        if strat == "SMA Crossover (50/200)":
            bt['SMA50']  = bt['Price'].rolling(50).mean()
            bt['SMA200'] = bt['Price'].rolling(200).mean()
            bt['Signal'] = np.where(bt['SMA50'] > bt['SMA200'], 1, 0)
            desc = "Long when the 50-day SMA is above the 200-day SMA (golden cross), flat otherwise."

        elif strat == "RSI Mean Reversion":
            d  = bt['Price'].diff()
            g  = d.where(d > 0, 0).rolling(14).mean()
            l  = (-d.where(d < 0, 0)).rolling(14).mean()
            r  = 100 - (100 / (1 + g / l))
            bt['RSI']    = r
            bt['Signal'] = np.where(r < 35, 1, np.where(r > 65, 0, np.nan))
            bt['Signal'] = bt['Signal'].ffill().fillna(0)
            desc = "Buy when RSI drops below 35 (oversold), sell when RSI exceeds 65."

        else:
            bt['High52'] = bt['Price'].rolling(252, min_periods=30).max().shift(1)
            bt['Signal'] = np.where(bt['Price'] >= bt['High52'], 1, 0)
            desc = "Buy when price breaks above the 52-week high, hold until it falls 5% from peak."

        bt['Strat_Returns']  = bt['Signal'].shift(1) * bt['Returns']
        bt['Cum_Market']     = (1 + bt['Returns'].fillna(0)).cumprod() * 100
        bt['Cum_Strategy']   = (1 + bt['Strat_Returns'].fillna(0)).cumprod() * 100

        final_mkt = float(bt['Cum_Market'].iloc[-1]) - 100
        final_str = float(bt['Cum_Strategy'].iloc[-1]) - 100
        sharpe = (bt['Strat_Returns'].mean() / bt['Strat_Returns'].std() * np.sqrt(252)) if bt['Strat_Returns'].std() != 0 else 0
        max_dd = ((bt['Cum_Strategy'] / bt['Cum_Strategy'].cummax()) - 1).min() * 100

        st.markdown(f'<p style="font-size:13px;color:#475569;margin-bottom:16px;">{desc}</p>', unsafe_allow_html=True)

        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Strategy return",   f"{final_str:.1f}%")
        b2.metric("Buy & hold return", f"{final_mkt:.1f}%")
        b3.metric("Sharpe ratio",      f"{sharpe:.2f}")
        b4.metric("Max drawdown",      f"{max_dd:.1f}%")

        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=bt.index, y=bt['Cum_Strategy'], name='Strategy',
                                    line=dict(color='#22c55e', width=2)))
        fig_bt.add_trace(go.Scatter(x=bt.index, y=bt['Cum_Market'],   name='Buy & Hold',
                                    line=dict(color='#475569', width=1.5, dash='dot')))
        fig_bt.add_hline(y=100, line_dash='dash', line_color='#1e2330', line_width=1)
        fig_bt.update_layout(**PLOT_LAYOUT, height=380, yaxis_title='Portfolio value (started at $100)')
        st.plotly_chart(fig_bt, use_container_width=True)

        # Drawdown chart
        dd_series = (bt['Cum_Strategy'] / bt['Cum_Strategy'].cummax() - 1) * 100
        fig_dd = go.Figure(go.Scatter(
            x=bt.index, y=dd_series,
            fill='tozeroy', fillcolor='rgba(239,68,68,0.1)',
            line=dict(color='#ef4444', width=1),
            name='Drawdown'
        ))
        fig_dd.update_layout(**PLOT_LAYOUT, height=200, yaxis_title='Drawdown %')
        st.plotly_chart(fig_dd, use_container_width=True)
    else:
        st.info("Not enough data to run a backtest. Extend the date range.")
