import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import time
import numpy as np
from sklearn.linear_model import LinearRegression

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

# Calculate trend
def get_trend(data, column):
    if len(data) < 2:
        return "→", "Stable"
    y = data[column].values
    x = np.arange(len(y))
    slope = np.polyfit(x, y, 1)[0]
    return ("↑", "Increasing") if slope > 0.1 else ("↓", "Decreasing") if slope < -0.1 else ("→", "Stable")

# Predict next value
def predict_next(data, column):
    if len(data) < 3:
        return None
    X = np.arange(len(data)).reshape(-1, 1)
    y = data[column].values
    model = LinearRegression().fit(X, y)
    return model.predict([[len(data)]])[0]

# Status check (adjusted for water level interpretation)
def get_status(value, thresholds, reverse=False):
    if reverse:  # For distance (water level), lower is critical (near bridge)
        if value <= thresholds['critical']:
            return "Critical", "red"
        elif value <= thresholds['warning']:
            return "Warning", "yellow"
        return "Safe", "green"
    else:  # For weight, accel, etc., higher is critical
        if value >= thresholds['critical']:
            return "Critical", "red"
        elif value >= thresholds['warning']:
            return "Warning", "yellow"
        return "Safe", "green"

# Initialize DataFrames with proper datetime type for created_at
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=['created_at', 'distance', 'weight', 'temperature', 'accel', 'gyro'])
    st.session_state.df['created_at'] = pd.to_datetime(st.session_state.df['created_at'])
if 'historical_df' not in st.session_state:
    st.session_state.historical_df = pd.DataFrame(columns=['created_at', 'distance', 'weight', 'temperature', 'accel', 'gyro'])
    st.session_state.historical_df['created_at'] = pd.to_datetime(st.session_state.historical_df['created_at'])

def main():
    st.title("Bridge Monitoring Dashboard")

    # Sidebar for controls
    st.sidebar.header("Controls")
    refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 1, 10, 2)
    st.sidebar.header("Thresholds")
    st.sidebar.markdown("**Water Level**: Lower values mean water is closer to the bridge.")
    water_critical = st.sidebar.number_input("Water Level Critical (cm, near bridge)", value=5.0, step=0.1)
    water_warning = st.sidebar.number_input("Water Level Warning (cm, near bridge)", value=10.0, step=0.1)
    weight_critical = st.sidebar.number_input("Weight Critical (g)", value=80.0, step=1.0)
    weight_warning = st.sidebar.number_input("Weight Warning (g)", value=60.0, step=1.0)
    vib_critical = st.sidebar.number_input("Vibration Critical (m/s²)", value=1.0, step=0.1)
    vib_warning = st.sidebar.number_input("Vibration Warning (m/s²)", value=0.5, step=0.1)

    THRESHOLDS = {
        'distance': {'critical': water_critical, 'warning': water_warning},
        'weight': {'critical': weight_critical, 'warning': weight_warning},
        'accel': {'critical': vib_critical, 'warning': vib_warning}
    }

    # Placeholders (removed export_placeholder)
    metrics_placeholder = st.empty()
    charts_placeholder = st.empty()
    recent_placeholder = st.empty()
    predictions_placeholder = st.empty()
    recommendations_placeholder = st.empty()

    # Initialize figures
    figs = {key: go.Figure() for key in ['water', 'weight', 'temp', 'vib', 'gyro']}

    while True:
        data = fetch_data()
        if not data:
            st.warning("No data available")
            time.sleep(refresh_rate)
            continue

        # Process recent data
        new_df = pd.DataFrame(data)
        if 'created_at' in new_df.columns:
            new_df['created_at'] = pd.to_datetime(new_df['created_at'])
        else:
            new_df['created_at'] = pd.Timestamp.now()
        
        st.session_state.df = pd.concat([st.session_state.df, new_df]).drop_duplicates(subset=['created_at']).sort_values('created_at').tail(5)
        recent_df = st.session_state.df

        # Process historical data (last 50 entries for context)
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
                st.metric("Water Level (Distance to Bridge)", f"{latest['distance']:.2f} cm", water_status, delta_color="off")
                st.markdown(f"<p style='color:{water_color}'>{water_status}</p>", unsafe_allow_html=True)
            with col2:
                st.metric("Weight", f"{latest['weight']:.2f} g", weight_status, delta_color="off")
                st.markdown(f"<p style='color:{weight_color}'>{weight_status}</p>", unsafe_allow_html=True)
            with col3:
                st.metric("Temperature", f"{latest['temperature']:.2f} °C")
            with col4:
                st.metric("Vibration Avg", f"{latest['accel']:.2f} m/s²", vib_status, delta_color="off")
                st.markdown(f"<p style='color:{vib_color}'>{vib_status}</p>", unsafe_allow_html=True)
            with col5:
                st.metric("Gyro Avg", f"{latest['gyro']:.2f} rad/s")

        # Charts with Trends and Historical Context
        with charts_placeholder.container():
            st.header("Time Series Data (Last 5 Measurements)")
            for fig, key, title, y_label, trace_type in [
                (figs['water'], 'distance', 'Water Levels (Distance to Bridge)', 'Distance to Bridge (cm)', go.Bar),
                (figs['weight'], 'weight', 'Weight Over Time', 'Weight (g)', go.Bar),
                (figs['temp'], 'temperature', 'Temperature Over Time', 'Temperature (°C)', go.Scatter),
                (figs['vib'], 'accel', 'Vibrations Over Time', 'Vibration (m/s²)', go.Scatter),
                (figs['gyro'], 'gyro', 'Average Gyro Over Time', 'Angular Velocity (rad/s)', go.Scatter)
            ]:
                fig.data = []
                trend_arrow, trend_text = get_trend(recent_df, key)
                mean_val = historical_df[key].mean()
                std_val = historical_df[key].std()
                if trace_type == go.Bar:
                    fig.add_trace(go.Bar(x=recent_df['created_at'], y=recent_df[key], name=title.split()[0]))
                else:
                    fig.add_trace(go.Scatter(x=recent_df['created_at'], y=recent_df[key], mode='lines+markers', name=title.split()[0]))
                fig.update_layout(
                    title=f"{title} ({trend_arrow} {trend_text}, Avg: {mean_val:.2f})",
                    xaxis_title='Time',
                    yaxis_title=y_label,
                    template='plotly_dark'
                )
                if key == 'weight':
                    fig.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Threshold (80g)")
                if std_val > 0 and abs(latest[key] - mean_val) > 2 * std_val:
                    fig.add_annotation(x=latest['created_at'], y=latest[key], text="Anomaly!", showarrow=True, arrowhead=1, ax=20, ay=-30)
                st.plotly_chart(fig, use_container_width=True)

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
            next_water = predict_next(recent_df, 'distance')
            next_weight = predict_next(recent_df, 'weight')
            next_vib = predict_next(recent_df, 'accel')
            with col1:
                st.write(f"Water Level: {next_water:.2f} cm" if next_water else "Insufficient data")
            with col2:
                st.write(f"Weight: {next_weight:.2f} g" if next_weight else "Insufficient data")
            with col3:
                st.write(f"Vibration: {next_vib:.2f} m/s²" if next_vib else "Insufficient data")

        # Recommendations (adjusted for water level interpretation)
        with recommendations_placeholder.container():
            st.header("Maintenance Recommendations")
            recs = []
            if water_status == "Critical":
                recs.append("Urgent: Water level too close to bridge (high flood risk). Inspect immediately.")
            elif water_status == "Warning":
                recs.append("Caution: Water level approaching bridge. Monitor flood risk.")
            if weight_status == "Critical":
                recs.append("Urgent: Excessive weight detected. Inspect bridge supports.")
            elif weight_status == "Warning":
                recs.append("Caution: Weight approaching limit. Reduce load if possible.")
            if vib_status == "Critical":
                recs.append("Urgent: High vibrations detected. Check structural integrity.")
            elif vib_status == "Warning":
                recs.append("Caution: Vibrations increasing. Schedule inspection.")
            if not recs:
                st.success("All systems normal. Continue regular monitoring.")
            else:
                for rec in recs:
                    st.warning(rec)

        time.sleep(refresh_rate)

if __name__ == "__main__":
    main()
