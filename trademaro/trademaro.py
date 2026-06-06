import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.svm import SVR, SVC
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from datetime import date, timedelta, datetime
from binance.client import Client
import time

# --- Page Config ---
st.set_page_config(
    page_title="TradeMaro",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📈"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric {
        background-color: #1e2127;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #30333d;
    }
    h1, h2, h3 { font-family: 'Helvetica Neue', sans-serif; }
    /* Metric label color */
    div[data-testid="stMetricLabel"] > label { color: #b2b5be; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.title("🛠️ Settings")

# 1. Market & Symbol
st.sidebar.subheader("📍 Asset Selection")
market_type = st.sidebar.selectbox("Market Type", ["Stocks (NSE/BSE)", "Cryptocurrencies", "Forex"])

symbol = ""
@st.cache_data
def load_csv_data(exchange):
    try:
        if exchange == "NSE":
            df = pd.read_csv("nse_stocks.csv")
            df['Display'] = df['SYMBOL'] + " - " + df['NAME OF COMPANY']
        else:
            df = pd.read_csv("bse_stocks.csv")
            df['Symbol'] = df['Symbol'].astype(str)
            df['Display'] = df['Symbol'] + " - " + df['CompanyName']
        return df
    except:
        return None

if market_type == "Stocks (NSE/BSE)":
    exchange = st.sidebar.radio("Exchange", ["NSE", "BSE"], horizontal=True)
    csv_data = load_csv_data(exchange)
    search_query = st.sidebar.text_input("Search Stock", placeholder="e.g. Reliance")
    
    if csv_data is not None and search_query:
        if exchange == "NSE":
            mask = (csv_data['NAME OF COMPANY'].str.contains(search_query, case=False, na=False) | 
                    csv_data['SYMBOL'].str.contains(search_query, case=False, na=False))
        else:
            mask = (csv_data['CompanyName'].str.contains(search_query, case=False, na=False) | 
                    csv_data['Symbol'].str.contains(search_query, case=False, na=False))
        filtered_df = csv_data[mask]
        if not filtered_df.empty:
            sel = st.sidebar.selectbox("Select", filtered_df['Display'].tolist())
            symbol = sel.split(" - ")[0] + (".NS" if exchange == "NSE" else ".BO")
        else:
            symbol = st.sidebar.text_input("Custom Symbol", value="RELIANCE.NS")
    else:
        symbol = st.sidebar.text_input("Enter Symbol", value="RELIANCE.NS")

elif market_type == "Cryptocurrencies":
    popular = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
    if st.sidebar.checkbox("Use Popular List", value=True):
        symbol = st.sidebar.selectbox("Select", popular)
    else:
        symbol = st.sidebar.text_input("Enter Symbol", "BTCUSDT").upper()

elif market_type == "Forex":
    symbol = st.sidebar.text_input("Enter Pair", "EURUSD=X").upper()

# 2. Date Range (Hidden from sidebar, using defaults)
today = date.today()
start_date = today - timedelta(days=365)
end_date = today

# 3. Auto Refresh (Sidebar)
st.sidebar.markdown("---")
auto_refresh = st.sidebar.checkbox("🔄 Auto-Refresh (Live)", value=False)


# --- MAIN AREA ---
st.title(f"📈 {symbol if symbol else 'TradeMaro Analytics'}")

# Top Toolbar: Timeframe Only
col_tf, col_spacer = st.columns([1, 3])
with col_tf:
    timeframe_label = st.selectbox("⏱️ Timeframe", 
        ["1 Minute", "5 Minutes", "15 Minutes", "30 Minutes", "1 Hour", "1 Day", "1 Week", "1 Month"],
        index=5
    )

# Logic
yf_intervals = {
    "1 Minute": "1m", "5 Minutes": "5m", "15 Minutes": "15m", "30 Minutes": "30m",
    "1 Hour": "1h", "1 Day": "1d", "1 Week": "1wk", "1 Month": "1mo"
}
binance_intervals = {
    "1 Minute": Client.KLINE_INTERVAL_1MINUTE, "5 Minutes": Client.KLINE_INTERVAL_5MINUTE,
    "15 Minutes": Client.KLINE_INTERVAL_15MINUTE, "30 Minutes": Client.KLINE_INTERVAL_30MINUTE,
    "1 Hour": Client.KLINE_INTERVAL_1HOUR, "1 Day": Client.KLINE_INTERVAL_1DAY,
    "1 Week": Client.KLINE_INTERVAL_1WEEK, "1 Month": Client.KLINE_INTERVAL_1MONTH
}
interval = yf_intervals[timeframe_label]
is_intraday = interval in ["1m", "5m", "15m", "30m", "1h"]


# --- Data Fetching ---
def get_data(symbol, m_type, interval_label, start_dt, end_dt):
    try:
        yf_int = yf_intervals[interval_label]
        bin_int = binance_intervals[interval_label]

        if m_type == "Cryptocurrencies":
            client = Client("", "") 
            klines = client.get_historical_klines(
                symbol, bin_int, 
                str(int(datetime.combine(start_dt, datetime.min.time()).timestamp() * 1000)), 
                str(int(datetime.combine(end_dt + timedelta(days=1), datetime.min.time()).timestamp() * 1000) - 1)
            )
            if not klines: return None
            df = pd.DataFrame(klines, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'ct', 'qav', 'nt', 'tbav', 'tqav', 'ig'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
            return df
        else:
            data = yf.download(symbol, start=start_dt, end=end_dt + timedelta(days=1), interval=yf_int, progress=False, auto_adjust=True)
            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
            return data if not data.empty else None
    except: return None

# --- Main Display Loop ---
placeholder = st.empty()

while True:
    with placeholder.container():
        if symbol:
            df = get_data(symbol, market_type, timeframe_label, start_date, end_date)
            
            if df is not None and not df.empty:
                # Metrics
                last = df['Close'].iloc[-1]
                prev = df['Close'].iloc[-2] if len(df) > 1 else last
                chg = last - prev
                pct = (chg / prev) * 100
                
                # Metrics Row
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Price", f"{last:.2f}", f"{chg:.2f} ({pct:.2f}%)")
                m2.metric("High", f"{df['High'].max():.2f}")
                m3.metric("Low", f"{df['Low'].min():.2f}")
                m4.metric("Volume", f"{df['Volume'].iloc[-1]:,.0f}")

                # Indicators (Calculated globally for Prediction & Chart)
                if len(df) > 20:
                    df['MA20'] = df['Close'].rolling(20).mean()
                    df['Upper'] = df['MA20'] + (df['Close'].rolling(20).std() * 2)
                    df['Lower'] = df['MA20'] - (df['Close'].rolling(20).std() * 2)
                    
                    delta = df['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                    rs = gain / loss
                    df['RSI'] = 100 - (100 / (1 + rs))

                # Tabs
                tab_chart, tab_ai, tab_compare, tab_funds, tab_data = st.tabs(["📈 Chart", "🔮 Prediction", "🧠 Research", "🏢 Info", "📋 Data"])
                
                with tab_chart:
                    # Plot
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.02)
                    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
                    
                    if len(df) > 20:
                        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='orange', width=1), name="MA20"), row=1, col=1)
                        fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], line=dict(color='gray', dash='dot'), name="BB Upper"), row=1, col=1)
                        fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], line=dict(color='gray', dash='dot'), name="BB Lower"), row=1, col=1)
                        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple'), name="RSI"), row=2, col=1)
                        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

                    fig.update_layout(height=600, margin=dict(t=0, b=0, l=0, r=0), template="plotly_dark")
                    fig.update_xaxes(showgrid=False, rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)

                with tab_compare:
                    st.subheader("🔬 Automated Model Research")
                    st.write("Compare multiple algorithms to find the best predictor for this specific asset.")
                    
                    if st.button("Run Full Model Comparison Study"):
                        with st.spinner("Training multiple models... This might take a moment."):
                            # --- Data Preparation for Research ---
                            df_res = df.copy()
                            df_res.dropna(inplace=True)
                            
                            # Features
                            feat_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MA20']
                            if not all(c in df_res.columns for c in feat_cols):
                                st.error("Not enough data to calculate indicators for research.")
                            else:
                                # 1. Regression Setup (Predict Price)
                                df_res['Target_Reg'] = df_res['Close'].shift(-7)
                                
                                # 2. Classification Setup (Predict Direction: 1 if Up, 0 if Down)
                                df_res['Target_Class'] = (df_res['Close'].shift(-7) > df_res['Close']).astype(int)
                                
                                df_res.dropna(inplace=True)
                                
                                X_res = df_res[feat_cols].values
                                y_reg = df_res['Target_Reg'].values
                                y_class = df_res['Target_Class'].values
                                
                                # Split
                                xr_train, xr_test, yr_train, yr_test = train_test_split(X_res, y_reg, test_size=0.2, random_state=42)
                                xc_train, xc_test, yc_train, yc_test = train_test_split(X_res, y_class, test_size=0.2, random_state=42)

                                # --- Regression Study ---
                                st.markdown("### 1. Regression Models (Predicting Exact Price)")
                                reg_models = {
                                    "Linear Regression": LinearRegression(),
                                    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                                    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                                    "Ridge Regression": Ridge(),
                                    "SVR (Support Vector)": make_pipeline(StandardScaler(), SVR(kernel='rbf')),
                                    "KNN Regressor": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5))
                                }
                                
                                reg_results = []
                                for name, model in reg_models.items():
                                    try:
                                        model.fit(xr_train, yr_train)
                                        preds = model.predict(xr_test)
                                        rmse = np.sqrt(mean_squared_error(yr_test, preds))
                                        r2 = r2_score(yr_test, preds)
                                        reg_results.append({"Model": name, "RMSE (Error)": rmse, "R2 Score": r2})
                                    except Exception as e:
                                        reg_results.append({"Model": name, "RMSE (Error)": 9999, "R2 Score": -9999})

                                reg_df = pd.DataFrame(reg_results).sort_values(by="RMSE (Error)")
                                
                                # Chart: Regression Error
                                fig_reg = go.Figure(go.Bar(
                                    x=reg_df['Model'], y=reg_df['RMSE (Error)'],
                                    text=reg_df['RMSE (Error)'].apply(lambda x: f"{x:.2f}"), textposition='auto',
                                    marker_color='#ff4b4b'
                                ))
                                fig_reg.update_layout(title="Model Error Comparison (RMSE - Lower is Better)", template="plotly_dark", height=300)
                                st.plotly_chart(fig_reg, use_container_width=True)

                                st.dataframe(reg_df.style.format({"RMSE (Error)": "{:.2f}", "R2 Score": "{:.4f}"}), use_container_width=True)
                                
                                best_reg = reg_df.iloc[0]['Model']
                                st.success(f"🏆 Best Regression Model: **{best_reg}**")

                                # --- Classification Study ---
                                st.markdown("---")
                                st.markdown("### 2. Classification Models (Predicting Direction: ⬆️ Up / ⬇️ Down)")
                                class_models = {
                                    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
                                    "Random Forest Class.": RandomForestClassifier(n_estimators=100, random_state=42),
                                    "Gradient Boosting Class.": GradientBoostingClassifier(random_state=42),
                                    "SVC (Support Vector)": make_pipeline(StandardScaler(), SVC()),
                                    "KNN Classifier": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
                                }
                                
                                class_results = []
                                for name, model in class_models.items():
                                    try:
                                        model.fit(xc_train, yc_train)
                                        preds = model.predict(xc_test)
                                        acc = accuracy_score(yc_test, preds) * 100
                                        class_results.append({"Model": name, "Accuracy (%)": acc})
                                    except:
                                        class_results.append({"Model": name, "Accuracy (%)": 0})

                                class_df = pd.DataFrame(class_results).sort_values(by="Accuracy (%)", ascending=False)
                                
                                # Chart: Classification Accuracy
                                fig_class = go.Figure(go.Bar(
                                    x=class_df['Model'], y=class_df['Accuracy (%)'],
                                    text=class_df['Accuracy (%)'].apply(lambda x: f"{x:.1f}%"), textposition='auto',
                                    marker_color='#00c853'
                                ))
                                fig_class.update_layout(title="Model Accuracy Comparison (Higher is Better)", template="plotly_dark", height=300)
                                st.plotly_chart(fig_class, use_container_width=True)

                                st.dataframe(class_df.style.format({"Accuracy (%)": "{:.2f}%"}), use_container_width=True)
                                
                                best_class = class_df.iloc[0]['Model']
                                st.success(f"🏆 Best Classification Model: **{best_class}**")


                with tab_ai:
                    if is_intraday:
                        st.warning("Prediction optimized for Daily timeframe (data volatility is high in intraday).")
                    elif len(df) > 50:
                        st.subheader("Random Forest Price Prediction")
                        
                        # Data Prep
                        df_t = df.copy()
                        df_t['Target'] = df_t['Close'].shift(-7)
                        df_t.dropna(inplace=True)
                        
                        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MA20']
                        # Ensure features exist (might be missing if len < 20, but len>50 check above covers it generally, though safe to be sure)
                        if all(col in df_t.columns for col in features):
                            X = df_t[features].values
                            y = df_t['Target'].values
                            xt, xte, yt, yte = train_test_split(X, y, test_size=0.2, random_state=42)

                            # Single Model: Random Forest
                            model = RandomForestRegressor(n_estimators=100, random_state=42)
                            model.fit(xt, yt)
                            rmse = np.sqrt(mean_squared_error(yte, model.predict(xte)))
                            st.info(f"Model RMSE (Error Margin): {rmse:.2f}")

                            # Forecast
                            last_X = df[features].tail(7).values
                            forecast = model.predict(last_X)
                            dates = [df.index[-1] + timedelta(days=i) for i in range(1, 8)]

                            # Chart
                            fig_p = go.Figure()
                            fig_p.add_trace(go.Scatter(x=df.index[-60:], y=df['Close'].tail(60), name="History"))
                            fig_p.add_trace(go.Scatter(x=dates, y=forecast, name="Forecast", mode='lines+markers', line=dict(color='#ff4b4b')))
                            fig_p.update_layout(template="plotly_dark", height=400, title="7-Day Price Forecast")
                            st.plotly_chart(fig_p, use_container_width=True)

                            # Table
                            pred_df = pd.DataFrame({'Date': dates, 'Predicted Price': forecast})
                            st.dataframe(pred_df.style.format({'Predicted Price': '{:.2f}'}), use_container_width=True)
                        else:
                            st.error("Not enough data for indicators (RSI/MA20) calculation.")

                    else:
                        st.error("Not enough data to train models.")

                with tab_funds:
                    if market_type == "Stocks (NSE/BSE)":
                        try:
                            info = yf.Ticker(symbol).info
                            st.write(f"**Name:** {info.get('longName', symbol)}")
                            st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                            st.write(info.get('longBusinessSummary', ''))
                        except: st.warning("No Info.")
                    else: st.info("Stocks only.")

                with tab_data:
                    st.dataframe(df.sort_index(ascending=False), use_container_width=True)

            else:
                st.warning("No data found for this range.")
    
    if auto_refresh:
        time.sleep(30 if is_intraday else 60)
        st.rerun()
    else:
        break