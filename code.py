import streamlit as st
import pandas as pd
from influxdb_client import InfluxDBClient, Point
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# Initialize InfluxDB client once
INFLUX_CONFIG = {
    "url": "https://us-east-1-1.aws.cloud2.influxdata.com",
    "token": "ELdquVkstVtEKJ9QygEo9n5RGbOuGxPJ9VZnP_pyLVDWtXcmf7cC5IKtQFsJCWmBwVTzfgt5CmXyluB4B39tGA==",
    "org": "84724340f460ff21",
    "bucket": "april"
}
measurement = "csv_data99"
client = InfluxDBClient(**INFLUX_CONFIG)
write_api = client.write_api()

st.set_page_config(page_title="File to InfluxDB Uploader", page_icon=":bar_chart:", layout="wide")
st.title(":new_moon_with_face: File to InfluxDB Uploader")
st.markdown('<style>div.block-container{padding-top:2rem;}</style>', unsafe_allow_html=True)

@st.cache_data
def read_file(file):
    try:
        if file.name.endswith(".csv"):
            return pd.read_csv(file, dayfirst=True)
        elif file.name.endswith(".xlsx"):
            return pd.read_excel(file, sheet_name=None, engine="openpyxl").pipe(lambda x: pd.concat(x.values(), ignore_index=True))
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

@st.cache_data
def filter_dataframe(df):
    for column in df.columns:
        unique_values = df[column].dropna().unique()
        selected_values = st.sidebar.multiselect(f"Filter by {column}", unique_values)
        if selected_values:
            df = df[df[column].isin(selected_values)]

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%Y", errors='coerce')
        valid_dates = df['Date'].dropna()
        if not valid_dates.empty:
            date_range = st.sidebar.date_input("Filter by Date Range", [valid_dates.min().date(), valid_dates.max().date()])
            start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    return df

def upload_to_influxdb(df):
    points = []
    for _, row in df.iterrows():
        point = Point(measurement).tag("source", "streamlit_upload")
        for col, value in row.items():
            point.field(col, str(value) if isinstance(value, pd.Timestamp) else value)
        points.append(point)
    write_api.write(bucket=INFLUX_CONFIG['bucket'], org=INFLUX_CONFIG['org'], record=points)

uploaded_file = st.file_uploader(":file_folder: Please upload your file", type=["csv", "xlsx"])
if uploaded_file:
    df = read_file(uploaded_file)
    if df is not None:
        st.write("### Preview of Uploaded Data:")
        st.write(df.head())
        df = filter_dataframe(df)
        st.write("### Filtered Data:")
        st.write(df)

        if 'Amount' in df.columns:
            st.write(f"Total Amount: {df['Amount'].sum()}")

        if st.button("Upload to InfluxDB"):
            try:
                upload_to_influxdb(df)
                st.success("Data uploaded successfully to InfluxDB!")
            except Exception as e:
                st.error(f"Error uploading data: {e}")

client.close()
