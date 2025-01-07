import streamlit as st
import pandas as pd
from influxdb_client import InfluxDBClient, Point, WritePrecision
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# Initialize InfluxDB client using URL, token, org, and bucket
url = "https://us-east-1-1.aws.cloud2.influxdata.com"
token = "ELdquVkstVtEKJ9QygEo9n5RGbOuGxPJ9VZnP_pyLVDWtXcmf7cC5IKtQFsJCWmBwVTzfgt5CmXyluB4B39tGA=="
org = "84724340f460ff21"
bucket = "april"
measurement = "csv_data99"
client = InfluxDBClient(url=url, token=token, org=org)
write_api = client.write_api()

st.set_page_config(page_title="File to InfluxDB Uploader", page_icon=":bar_chart", layout="wide")
st.title(":new_moon_with_face: File to InfluxDB Uploader")
st.markdown('<style>div.block-container{padding-top:2rem;}</style>', unsafe_allow_html=True)

@st.cache_data
def read_file(file):
    try:
        if file.name.endswith(".csv"):
            return pd.read_csv(file, dayfirst=True)  # Ensure dd/mm/yyyy format
        elif file.name.endswith(".xlsx"):
            # Read all sheets into a single DataFrame
            all_sheets = pd.read_excel(file, sheet_name=None, engine="openpyxl")
            return pd.concat(all_sheets.values(), ignore_index=True)
        else:
            st.error("Unsupported file format. Please upload CSV or XLSX.")
            return None
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

# File uploader
uploaded_file = st.file_uploader(":file_folder: Please upload your file", type=["csv", "xlsx"])

if uploaded_file is not None:
    df = read_file(uploaded_file)
    if df is not None:
        st.write("Preview of Uploaded Data:")
        st.write(df.head())

        # Filter Options
        filtered_df = df.copy()
        selected_columns = st.sidebar.multiselect("Filter by Columns", df.columns)
        if selected_columns:
            filtered_df = df[selected_columns]

        # Date Filtering
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%Y", errors='coerce')
            valid_dates = df['Date'].dropna()
            if not valid_dates.empty:
                date_range = st.sidebar.date_input(
                    "Filter by Date Range", [valid_dates.min().date(), valid_dates.max().date()]
                )
                filtered_df = df[(df['Date'] >= pd.to_datetime(date_range[0])) & (df['Date'] <= pd.to_datetime(date_range[1]))]

        st.write("Filtered Data:")
        st.write(filtered_df)

        # Convert DataFrame to InfluxDB JSON format and upload
        def upload_to_influxdb(df):
            for _, row in df.iterrows():
                point = Point(measurement).tag("source", "streamlit_upload")
                for col, value in row.items():
                    if isinstance(value, pd.Timestamp):
                        value = value.strftime("%d/%m/%Y")
                    point.field(col, value)
                write_api.write(bucket=bucket, org=org, record=point)

        # Upload to InfluxDB
        if st.button("Upload to InfluxDB"):
            try:
                upload_to_influxdb(filtered_df)
                st.success("Data uploaded successfully to InfluxDB!")
            except Exception as e:
                st.error(f"Error uploading data: {e}")

# Close the connection on app stop
client.close()
