import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats

# Supabase configuration
SUPABASE_URL = "https://jehgpwipfqzcqnkbbvol.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImplaGdwd2lwZnF6Y3Fua2Jidm9sIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDAwMzc1MzEsImV4cCI6MjA1NTYxMzUzMX0.8hFhcysb_NP8y6kYDSE9w2w3l8ydyLcCRLZrG21ck-o"
TABLE_NAME = "bridge"

# Function to fetch data from Supabase
@st.cache_data(ttl=5)
def fetch_data():
    url = f"{SUPABASE_URL}/rest/v1/{TABLE_NAME}"
    headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}", "Content-Type": "application/json"}
    response = requests.get(url, headers=headers)
    return response.json() if response.status_code == 200 else []

# Calculate trend (enhanced with long-term analysis)
def get_trend(data, column, short_term_window=5, long_term_window=50):
    if len(data) < 2:
        return "→", "Stable", "→", "Stable"
    
    # Short-term trend (last 5 measurements)
    short_data = data.tail(short_term_window)
    y_short = pd.to_numeric(short_data[column], errors='coerce').dropna()
    if len(y_short) >= 2:
        x_short = np.arange(len(y_short))
        slope_short = np.polyfit(x_short, y_short, 1)[0]
        short_trend = ("↑", "Increasing") if slope_short > 0.1 else ("↓", "Decreasing") if slope_short < -0.1 else ("→", "Stable")
    else:
        short_trend = ("→", "Stable")

    # Long-term trend (last 50 measurements)
    long_data = data.tail(long_term_window)
    y_long = pd.to_numeric(long_data[column], errors='coerce').dropna()
    if len(y_long) >= 2:
        x_long = np.arange(len(y_long))
        slope_long = np.polyfit(x_long, y_long, 1)[0]
        long_trend = ("↑", "Increasing") if slope_long > 0.05 else ("↓", "Decreasing") if slope_long < -0.05 else ("→", "Stable")
    else:
        long_trend = ("→", "Stable")

    return short_trend[0], short_trend[1], long_trend[0], long_trend[1]

# Enhanced prediction with confidence interval and rate of change
def predict_next(data, column, window=5):
    if len(data) < window:
        return None, None, None
    X = np.arange(len(data)).reshape(-1, 1)
    y = pd.to_numeric(data[column], errors='coerce').dropna().tail(window).values
    if len(y) < 2:
        return None, None, None
    model = LinearRegression().fit(X[-len(y):], y)
    pred = model.predict([[len(data)]])[0]
    y_pred = model.predict(X[-len(y):])
    residual = y - y_pred
    std_error = np.std(residual)
    conf_interval = stats.t.ppf(0.975, len(y)-2) * std_error
    rate_of_change = (y[-1] - y[-2]) if len(y) >= 2 else 0  # Per measurement
    return pred, conf_interval, rate_of_change

# Calculate failure probability
def calculate_failure_prob(value, threshold, historical_mean, historical_std, reverse=False):
    if historical_std == 0:
        return 0
    if reverse:  # For water level (distance), lower values are critical
        z_score = (value - threshold) / historical_std
        prob = stats.norm.cdf(z_score)
        return (1 - prob) * 100
    else:  # For weight and vibration, higher values are critical
        z_score = (threshold - value) / historical_std
        prob = stats.norm.cdf(z_score)
        return (1 - prob) * 100

# Status check
def get_status(value, thresholds, reverse=False):
    if reverse:
        if value <= thresholds['critical']:
            return "Critical", "red"
        elif value <= thresholds['warning']:
            return "Warning", "yellow"
        return "Safe", "green"
    else:
        if value >= thresholds['critical']:
            return "Critical", "red"
        elif value >= thresholds['warning']:
            return "Warning", "yellow"
        return "Safe", "green"

# Calculate correlation between metrics
def get_correlation(data, col1, col2):
    if len(data) < 2:
        return None
    corr = data[[col1, col2]].corr().iloc[0, 1]
    return corr if not pd.isna(corr) else None

# Initialize DataFrames
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=['created_at', 'distance', 'weight', 'temperature', 'accel', 'gyro'])
    st.session_state.df['created_at'] = pd.to_datetime(st.session_state.df['created_at'])
if 'historical_df' not in st.session_state:
    st.session_state.historical_df = pd.DataFrame(columns=['created_at', 'distance', 'weight', 'temperature', 'accel', 'gyro'])
    st.session_state.historical_df['created_at'] = pd.to_datetime(st.session_state.historical_df['created_at'])

def main():
    st.title("Bridge Monitoring Dashboard with Predictive Maintenance")

    # Sidebar with lock/unlock functionality
    st.sidebar.header("Controls")
    refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 1, 10, 2)
    
    if 'locked' not in st.session_state:
        st.session_state.locked = True
    st.session_state.locked = st.sidebar.toggle("Unlock Thresholds", value=st.session_state.locked)
    
    st.sidebar.header("Thresholds")
    st.sidebar.markdown("**Water Level**: Lower values mean water is closer to the bridge.")
    
    water_critical = st.sidebar.number_input("Water Level Critical (cm)", value=5.0, step=0.1, disabled=st.session_state.locked)
    water_warning = st.sidebar.number_input("Water Level Warning (cm)", value=10.0, step=0.1, disabled=st.session_state.locked)
    weight_critical = st.sidebar.number_input("Weight Critical (g)", value=80.0, step=1.0, disabled=st.session_state.locked)
    weight_warning = st.sidebar.number_input("Weight Warning (g)", value=60.0, step=1.0, disabled=st.session_state.locked)
    vib_critical = st.sidebar.number_input("Vibration Critical (m/s²)", value=3.9, step=0.1, disabled=st.session_state.locked)
    vib_warning = st.sidebar.number_input("Vibration Warning (m/s²)", value=3.88, step=0.1, disabled=st.session_state.locked)

    THRESHOLDS = {
        'distance': {'critical': water_critical, 'warning': water_warning},
        'weight': {'critical': weight_critical, 'warning': weight_warning},
        'accel': {'critical': vib_critical, 'warning': vib_warning}
    }

    # Placeholders
    status_placeholder = st.empty()
    metrics_placeholder = st.empty()
    charts_placeholder = st.empty()
    recent_placeholder = st.empty()
    predictions_placeholder = st.empty()
    maintenance_placeholder = st.empty()

    figs = {key: go.Figure() for key in ['water', 'weight', 'temp', 'vib', 'gyro']}
    max_retries = 5
    retry_count = 0
    warning_displayed = False

    if 'chart_iteration' not in st.session_state:
        st.session_state.chart_iteration = 0

    while True:
        data = fetch_data()
        
        if not data:
            retry_count += 1
            if not warning_displayed:
                with status_placeholder.container():
                    st.warning(f"No data available. Retrying {max_retries - retry_count} more times...")
                warning_displayed = True
            if retry_count >= max_retries:
                with status_placeholder.container():
                    st.error("Maximum retries reached. No data available from Supabase.")
                    if st.button("Retry Now"):
                        retry_count = 0
                        warning_displayed = False
                        continue
                    if st.button("Exit"):
                        st.write("Dashboard stopped.")
                        break
            time.sleep(refresh_rate)
            continue

        retry_count = 0
        if warning_displayed:
            status_placeholder.empty()
            warning_displayed = False

        # Process data
        new_df = pd.DataFrame(data)
        if 'created_at' in new_df.columns:
            new_df['created_at'] = pd.to_datetime(new_df['created_at'])
        else:
            new_df['created_at'] = pd.Timestamp.now()
        
        st.session_state.df = pd.concat([st.session_state.df, new_df]).drop_duplicates(subset=['created_at']).sort_values('created_at').tail(5)
        recent_df = st.session_state.df

        st.session_state.historical_df = pd.DataFrame(data)
        st.session_state.historical_df['created_at'] = pd.to_datetime(st.session_state.historical_df['created_at'])
        historical_df = st.session_state.historical_df.tail(50)

        # Metrics and Status
        with metrics_placeholder.container():
            st.header("Latest Measurements")
            latest = recent_df.iloc[-1]
            col1, col2, col3 = st.columns(3)
            col4, col5 = st.columns(2)
            water_status, water_color = get_status(latest['distance'], THRESHOLDS['distance'], reverse=True)
            weight_status, weight_color = get_status(latest['weight'], THRESHOLDS['weight'])
            vib_status, vib_color = get_status(latest['accel'], THRESHOLDS['accel'])
            with col1:
                st.metric("Water Level", f"{latest['distance']:.2f} cm", water_status, delta_color="off")
                st.markdown(f"<p style='color:{water_color}'>{water_status}</p>", unsafe_allow_html=True)
            with col2:
                st.metric("Weight", f"{latest['weight']:.2f} g", weight_status, delta_color="off")
                st.markdown(f"<p style='color:{weight_color}'>{weight_status}</p>", unsafe_allow_html=True)
            with col3:
                st.metric("Temperature", f"{latest['temperature']:.2f} °C")
            with col4:
                st.metric("Vibration", f"{latest['accel']:.2f} m/s²", vib_status, delta_color="off")
                st.markdown(f"<p style='color:{vib_color}'>{vib_status}</p>", unsafe_allow_html=True)
            with col5:
                st.metric("Gyro", f"{latest['gyro']:.2f} rad/s")

        # Charts
        with charts_placeholder.container():
            st.header("Time Series Data (Last 5 Measurements)")
            st.session_state.chart_iteration += 1
            iteration = st.session_state.chart_iteration

            for fig_key, key, title, y_label, trace_type in [
                ('water', 'distance', 'Water Levels', 'Distance to Bridge (cm)', go.Bar),
                ('weight', 'weight', 'Weight', 'Weight (g)', go.Bar),
                ('temp', 'temperature', 'Temperature', 'Temperature (°C)', go.Scatter),
                ('vib', 'accel', 'Vibrations', 'Vibration (m/s²)', go.Scatter),
                ('gyro', 'gyro', 'Gyro', 'Angular Velocity (rad/s)', go.Scatter)
            ]:
                fig = figs[fig_key]
                fig.data = []
                short_arrow, short_text, long_arrow, long_text = get_trend(historical_df, key)
                mean_val = historical_df[key].mean()
                std_val = historical_df[key].std()
                if trace_type == go.Bar:
                    fig.add_trace(go.Bar(x=recent_df['created_at'], y=recent_df[key], name=title.split()[0]))
                else:
                    fig.add_trace(go.Scatter(x=recent_df['created_at'], y=recent_df[key], mode='lines+markers', name=title.split()[0]))
                fig.update_layout(
                    title=f"{title} (Short: {short_arrow} {short_text}, Long: {long_arrow} {long_text}, Avg: {mean_val:.2f})",
                    xaxis_title='Time',
                    yaxis_title=y_label,
                    template='plotly_dark'
                )
                if key == 'weight':
                    fig.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Threshold (80g)")
                if std_val > 0 and abs(latest[key] - mean_val) > 2 * std_val:
                    fig.add_annotation(x=latest['created_at'], y=latest[key], text="Anomaly!", showarrow=True, arrowhead=1, ax=20, ay=-30)
                try:
                    st.plotly_chart(fig, use_container_width=True, key=f"{fig_key}_{iteration}")
                except Exception as e:
                    print(f"Suppressed error for chart {fig_key}: {str(e)}")
                    st.write(f"Chart for {title} is temporarily unavailable.")

        # Recent Data
        with recent_placeholder.container():
            st.header("Most Recent 5 Measurements")
            st.dataframe(recent_df[['created_at', 'distance', 'weight', 'temperature', 'accel', 'gyro']],
                         column_config={
                             "created_at": "Time", "distance": "Water Level (cm)", "weight": "Weight (g)",
                             "temperature": "Temp (°C)", "accel": "Vib (m/s²)", "gyro": "Gyro (rad/s)"
                         })

        # Predictions
        with predictions_placeholder.container():
            st.header("Predictions for Next Measurement")
            col1, col2, col3 = st.columns(3)
            next_water, water_ci, water_rate = predict_next(historical_df, 'distance')
            next_weight, weight_ci, weight_rate = predict_next(historical_df, 'weight')
            next_vib, vib_ci, vib_rate = predict_next(historical_df, 'accel')
            with col1:
                st.write("Water Level:")
                if next_water:
                    st.write(f"{next_water:.2f} ± {water_ci:.2f} cm (Rate: {water_rate:.2f} cm/meas)")
                else:
                    st.write("Insufficient data")
            with col2:
                st.write("Weight:")
                if next_weight:
                    st.write(f"{next_weight:.2f} ± {weight_ci:.2f} g (Rate: {weight_rate:.2f} g/meas)")
                else:
                    st.write("Insufficient data")
            with col3:
                st.write("Vibration:")
                if next_vib:
                    st.write(f"{next_vib:.2f} ± {vib_ci:.2f} m/s² (Rate: {vib_rate:.2f} m/s²/meas)")
                else:
                    st.write("Insufficient data")

        # Predictive Maintenance Recommendations
        with maintenance_placeholder.container():
            st.header("Predictive Maintenance Analysis")
            latest = recent_df.iloc[-1]
            water_mean = historical_df['distance'].mean()
            water_std = historical_df['distance'].std()
            weight_mean = historical_df['weight'].mean()
            weight_std = historical_df['weight'].std()
            vib_mean = historical_df['accel'].mean()
            vib_std = historical_df['accel'].std()

            water_prob = calculate_failure_prob(latest['distance'], water_critical, water_mean, water_std, reverse=True)
            weight_prob = calculate_failure_prob(latest['weight'], weight_critical, weight_mean, weight_std)
            vib_prob = calculate_failure_prob(latest['accel'], vib_critical, vib_mean, vib_std)

            # Correlations
            weight_vib_corr = get_correlation(historical_df, 'weight', 'accel')
            water_weight_corr = get_correlation(historical_df, 'distance', 'weight')

            recs = []
            anomaly_count = 0

            # Water Level analysis
            if water_status == "Critical":
                recs.append(f"CRITICAL: Water level failure probability {water_prob:.1f}%. Immediate flood risk inspection.")
            if next_water and next_water - water_ci < water_critical:
                days_to_failure = (latest['distance'] - water_critical) / abs(water_rate) if water_rate != 0 else float('inf')
                recs.append(f"WARNING: Water level may reach critical in ~{days_to_failure:.1f} measurements ({water_prob:.1f}% probability)")
            if water_rate and abs(water_rate) > 0.5:  # Rapid change detection
                recs.append(f"ALERT: Rapid water level change ({water_rate:.2f} cm/meas). Investigate potential flooding.")
            if water_std > 0 and abs(latest['distance'] - water_mean) > 3 * water_std:
                recs.append("ANOMALY: Unusual water level detected.")
                anomaly_count += 1

            # Weight analysis
            if weight_status == "Critical":
                recs.append(f"CRITICAL: Weight failure probability {weight_prob:.1f}%. Immediate structural inspection.")
            if next_weight and next_weight + weight_ci > weight_critical:
                days_to_failure = (weight_critical - latest['weight']) / weight_rate if weight_rate > 0 else float('inf')
                recs.append(f"WARNING: Weight may exceed critical in ~{days_to_failure:.1f} measurements ({weight_prob:.1f}% probability)")
            if weight_rate and abs(weight_rate) > 2.0:  # Rapid change detection
                recs.append(f"ALERT: Rapid weight change ({weight_rate:.2f} g/meas). Check load distribution.")
            if weight_std > 0 and abs(latest['weight'] - weight_mean) > 3 * weight_std:
                recs.append("ANOMALY: Unusual weight detected.")
                anomaly_count += 1

            # Vibration analysis
            if vib_status == "Critical":
                recs.append(f"CRITICAL: Vibration failure probability {vib_prob:.1f}%. Immediate inspection.")
            if next_vib and next_vib + vib_ci > vib_critical:
                days_to_failure = (vib_critical - latest['accel']) / vib_rate if vib_rate > 0 else float('inf')
                recs.append(f"WARNING: Vibration may exceed critical in ~{days_to_failure:.1f} measurements ({vib_prob:.1f}% probability)")
            if vib_rate and abs(vib_rate) > 0.2:  # Rapid change detection
                recs.append(f"ALERT: Rapid vibration change ({vib_rate:.2f} m/s²/meas). Inspect structural stability.")
            if vib_std > 0 and abs(latest['accel'] - vib_mean) > 3 * vib_std:
                recs.append("ANOMALY: Unusual vibration detected.")
                anomaly_count += 1

            # Cross-metric correlations
            if weight_vib_corr and abs(weight_vib_corr) > 0.7:
                recs.append(f"INSIGHT: Strong correlation between weight and vibration ({weight_vib_corr:.2f}). Heavy loads may be causing vibrations.")
            if water_weight_corr and abs(water_weight_corr) > 0.7:
                recs.append(f"INSIGHT: Strong correlation between water level and weight ({water_weight_corr:.2f}). Flooding may affect load.")

            # Anomaly clustering
            if anomaly_count >= 2:
                recs.append(f"CRITICAL: Multiple anomalies detected ({anomaly_count}). Systemic issue likely—urgent inspection recommended.")

            if not recs:
                st.success("All systems operating normally. Schedule routine maintenance in 30 days.")
            else:
                for rec in recs:
                    if "CRITICAL" in rec:
                        st.error(rec)
                    elif "WARNING" in rec or "ALERT" in rec:
                        st.warning(rec)
                    else:
                        st.info(rec)

        time.sleep(refresh_rate)

if __name__ == "__main__":
    main()
