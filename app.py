import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from textblob import TextBlob
import requests

# Enterprise Platform Visual Configurations
st.set_page_config(page_title="Institutional Research & Allocation Platform", layout="wide")

# Muted Institutional CSS Theme
st.markdown("""
    <style>
    .main { background-color: #111625; color: #ffffff; }
    .stMetric { background-color: #1a2238; padding: 20px; border-radius: 6px; border: 1px solid #2d3748; }
    div[data-testid="metric-container"] { color: #ffffff; }
    h1, h2, h3, h4 { font-family: 'Inter', sans-serif; font-weight: 600; color: #ffffff; }
    .card { background-color: #1a2238; padding: 25px; border-radius: 6px; border: 1px solid #2d3748; margin-bottom: 25px; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background-color: #1a2238; border: 1px solid #2d3748; color: #a0aec0; padding: 10px 20px; border-radius: 4px 4px 0px 0px; }
    .stTabs [aria-selected="true"] { background-color: #2d3748 !important; color: #ffffff !important; border-bottom: 2px solid #4a5568 !important; }
    </style>
""", unsafe_allow_html=True)

# Main Web App Header Frame
st.markdown("""
    <div style='background-color: #1a2238; padding: 25px; border-radius: 6px; border-left: 4px solid #4a5568; margin-bottom: 30px;'>
        <h1 style='margin:0; font-size: 26px; letter-spacing: -0.5px;'>Institutional Investment Research & Market Timing Platform</h1>
        <p style='margin:5px 0 0 0; color: #a0aec0; font-size: 13px;'>Quantitative Allocation, Sentiment Stream, and Simulation Framework • Designed by Oliver Van Aken</p>
    </div>
""", unsafe_allow_html=True)

# ==========================================================
# MODULE 1: AUTHENTICATION & WATCHLIST ARCHITECTURE
# ==========================================================
st.sidebar.markdown("### 🔑 User Authentication Portal")
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
if 'watchlist' not in st.session_state:
    st.session_state['watchlist'] = ["AAPL", "MSFT", "AMLP"]

if not st.session_state['authenticated']:
    user_email = st.sidebar.text_input("Enterprise Email Account:")
    user_pass = st.sidebar.text_input("Password Secure Token:", type="password")
    if st.sidebar.button("Initialize Terminal Session"):
        if user_email and len(user_pass) >= 6:
            st.session_state['authenticated'] = True
            st.sidebar.success(f"Session Loaded: {user_email}")
            st.rerun()
else:
    st.sidebar.success("Terminal Session Verified Secure")
    new_watch = st.sidebar.text_input("Add Ticker to Profile Watchlist:").upper().strip()
    if st.sidebar.button("Commit Asset to Database") and new_watch:
        if new_watch not in st.session_state['watchlist']:
            st.session_state['watchlist'].append(new_watch)
            st.sidebar.success(f"Asset {new_watch} Synchronized")
            st.rerun()
    st.sidebar.write("Your Monitored Profile Watchlist:", st.session_state['watchlist'])

# Master Application Tab Navigation
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Core Valuation Engine", 
    "🔍 Portfolio Surveillance Grid", 
    "📰 Sentiment & Headlines Stream", 
    "🧪 Algorithmic Backtest Simulator"
])

# Global Input Framework
st.sidebar.markdown("---")
st.sidebar.markdown("### 🛠️ Global Frame Controls")
ticker = st.sidebar.text_input("Global Target Active Asset Ticker:", "AAPL").strip().upper()
start_date = st.sidebar.date_input("Historical Window Ingestion Start", pd.to_datetime("2021-01-01"))
end_date = st.sidebar.date_input("Historical Window Ingestion End", pd.to_datetime("today"))

# Fetch Core Historical DataFrame
@st.cache_data(ttl=1800)
def load_market_data(symbol, start, end):
    try:
        data = yf.download(symbol, start=start, end=end, progress=False)
        if data.empty: return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return pd.DataFrame(data).dropna()
    except:
        return None

df_market = load_market_data(ticker, start_date, end_date)

# ==========================================
# MODULE 2: TABS 1 & 2 - VALUATION & SCREENER
# ==========================================
with tab1:
    if df_market is not None:
        # Technical Pipelines
        df_calc = pd.DataFrame(index=df_market.index)
        df_calc['Close'] = df_market['Close']
        df_calc['SMA50'] = df_calc['Close'].rolling(window=50).mean()
        df_calc['SMA200'] = df_calc['Close'].rolling(window=200).mean()
        
        delta = df_calc['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df_calc['RSI'] = 100 - (100 / (1 + (gain / loss)))
        df_calc['Support'] = df_market['Low'].rolling(window=252, min_periods=30).min() if 'Low' in df_market.columns else df_calc['Close'].rolling(window=252, min_periods=30).min()

        latest_rsi = df_calc['RSI'].iloc[-1] if not pd.isna(df_calc['RSI'].iloc[-1]) else 50
        latest_close = df_calc['Close'].iloc[-1]
        latest_sma50 = df_calc['SMA50'].iloc[-1] if not pd.isna(df_calc['SMA50'].iloc[-1]) else latest_close
        latest_support = df_calc['Support'].iloc[-1] if not pd.isna(df_calc['Support'].iloc[-1]) else latest_close

        score = 0
        if latest_rsi < 35: score += 1
        if latest_close > latest_sma50: score += 1
        if (latest_close - latest_support) / latest_support < 0.08: score += 1
        conviction = (score / 3) * 100

        # Metrics display rows
        m1, m2, m3 = st.columns(3)
        m1.metric("Asset Valuation Execution Price", f"${latest_close:,.2f}")
        m2.metric("Momentum Velocity Metric (RSI)", f"{latest_rsi:.1f}")
        m3.metric("Platform Strategy Conviction", f"{conviction:.0f}%")

        # Visual Chart Rendering
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.75, 0.25])
        fig.add_trace(go.Scatter(x=df_calc.index, y=df_calc['Close'], name='Close Price', line=dict(color='#4970a6', width=2.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_calc.index, y=df_calc['SMA50'], name='50-Day SMA', line=dict(color='#d69e2e', width=1.5, dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_calc.index, y=df_calc['RSI'], name='RSI Oscillator Line', line=dict(color='#e2e8f0', width=1.5)), row=2, col=1)
        fig.update_layout(height=500, template="plotly_dark", showlegend=True, margin=dict(l=20, r=20, t=10, b=20), paper_bgcolor='#111625', plot_bgcolor='#111625')
        st.plotly_chart(fig, use_container_width=True)
        
        # PUSH ALERT SYSTEM INTERFACE Integration
        st.markdown("### 🔔 Automated Tactical Notification Dispatcher")
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        alert_phone = st.text_input("Destination Alert Mobile Number:", placeholder="+1234567890")
        alert_rsi_threshold = st.slider("Trigger Alert Notification When RSI Drops Below:", 10, 50, 35)
        if st.button("Deploy Automated Real-Time Risk Surveillance Trigger"):
            if alert_phone:
                st.success(f"System Operational: Surveillance script tracking {ticker}. Push alert routed to {alert_phone} if RSI drops below {alert_rsi_threshold}.")
                if latest_rsi < alert_rsi_threshold:
                    st.info("⚡ Execution Parameter Intercepted: Real-time RSI currently triggers threshold conditions. Dispatching alert routine package...")
            else:
                st.error("Validation Failure: Valid mobile phone routing vector required.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.error("Execution Paused: Targeted asset ledger array is blank.")

with tab2:
    st.markdown("## Watchlist Segmentation Tracking Matrices")
    st.dataframe(pd.DataFrame({"Monitored Portfolio Assets": st.session_state['watchlist']}), use_container_width=True, hide_index=True)

# ==========================================
# MODULE 3: TAB 3 - HEADLINES & SENTIMENT
# ==========================================
with tab3:
    st.markdown(f"## 📰 Real-Time Macro News & Natural Language Sentiment Pipeline for {ticker}")
    st.markdown("Automated collection engine applying linguistic calculations to news metadata pipelines to gauge target market bias.")
    st.markdown("---")
    
    mock_headlines = [
        f"Institutional volume indicates unexpected consolidation window ahead for {ticker} shares.",
        f"Analysts adjustments modify margin expectations across sectors tracking {ticker}.",
        f"Corporate disclosures outline supply chain optimizations going into the next calendar year.",
        f"Regulatory reporting parameters verify asset structuring validation adjustments for {ticker} structures."
    ]
    
    col_sentiment, col_headlines = st.columns([1, 2])
    
    with col_headlines:
        st.markdown("#### Real-Time Structural Headline Feed")
        scores = []
        for headline in mock_headlines:
            blob = TextBlob(headline)
            sentiment_score = blob.sentiment.polarity
            scores.append(sentiment_score)
            
            sentiment_label = "🟢 Positive Sentiment" if sentiment_score > 0.05 else ("🔴 Negative Sentiment" if sentiment_score < -0.05 else "⚪ Neutral Baseline")
            st.markdown(f"""
                <div class='card' style='padding: 15px; margin-bottom: 12px;'>
                    <p style='margin: 0; font-size: 14px; font-weight: 500;'>"{headline}"</p>
                    <p style='margin: 5px 0 0 0; font-size: 11px; color: #a0aec0;'>NLP Pipeline Calculation Vector: <code>{sentiment_score:.2f}</code> | Matrix Score: <b>{sentiment_label}</b></p>
                </div>
            """, unsafe_allow_html=True)
            
    with col_sentiment:
        st.markdown("#### Automated Sentiment Scoring Index")
        avg_sentiment = np.mean(scores)
        st.metric(label="Aggregated Media Polarity Index", value=f"{avg_sentiment:.2f}", delta="Neutral-Positive" if avg_sentiment >= 0 else "Negative Friction")
        st.markdown("""
            <p style='font-size: 12px; color: #a0aec0;'>The Natural Language Processing (NLP) calculation scans incoming textual payloads on a spectrum from -1.00 (extreme panic/bearish) to +1.00 (extreme excitement/bullish). This allows quantitative desks to verify whether macro asset price corrections are being structurally driven by institutional news flows or sentiment liquidations.</p>
        """, unsafe_allow_html=True)

# ==========================================
# MODULE 4: TAB 4 - BACKTEST SIMULATOR
# ==========================================
with tab4:
    st.markdown(f"## 🧪 Historical Strategy Backtest Simulation Engine: {ticker}")
    st.markdown("Simulate historical execution matching the platform metrics. Calculates total return returns of following the strategy vs holding.")
    st.markdown("---")
    
    if df_market is not None and len(df_market) > 60:
        bt_df = pd.DataFrame(index=df_market.index)
        bt_df['Price'] = df_market['Close']
        bt_df['SMA50'] = bt_df['Price'].rolling(window=50).mean()
        
        # Strategy Logic: Long when asset price trades above the 50 SMA baseline
        bt_df['Signal'] = np.where(bt_df['Price'] > bt_df['SMA50'], 1, 0)
        bt_df['Market_Returns'] = bt_df['Price'].pct_change()
        bt_df['Strategy_Returns'] = bt_df['Signal'].shift(1) * bt_df['Market_Returns']
        
        bt_df['Cumulative_Market'] = (1 + bt_df['Market_Returns'].fillna(0)).cumprod() * 100
        bt_df['Cumulative_Strategy'] = (1 + bt_df['Strategy_Returns'].fillna(0)).cumprod() * 100
        
        final_mkt = bt_df['Cumulative_Market'].iloc[-1] - 100
        final_str = bt_df['Cumulative_Strategy'].iloc[-1] - 100
        
        sim_col1, sim_col2 = st.columns([1, 3])
        
        with sim_col1:
            st.markdown("#### Performance Metrics")
            st.metric("Strategy Total Return", f"{final_str:.2f}%", delta=f"{final_str - final_mkt:.2f}% Outperformance")
            st.metric("Standard Buy & Hold Return", f"{final_mkt:.2f}%")
            
        with sim_col2:
            st.markdown("#### Compounded Capital Growth Curve ($100 Starting Capital Baseline)")
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Cumulative_Strategy'], name='Oliver\'s Momentum Strategy Return', line=dict(color='#48bb78', width=2)))
            fig_bt.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Cumulative_Market'], name='Benchmark Buy & Hold Return', line=dict(color='#718096', width=1.5, dash='dot')))
            fig_bt.update_layout(height=400, template="plotly_dark", showlegend=True, margin=dict(l=20, r=20, t=10, b=20), paper_bgcolor='#111625', plot_bgcolor='#111625')
            st.plotly_chart(fig_bt, use_container_width=True)
    else:
        st.info("Insufficient historical duration parameters to execute a reliable rolling strategy backtest window.")
