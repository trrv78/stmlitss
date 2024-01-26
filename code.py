import pandas as pd 
import streamlit as st 
import plotly.express as px
import time

from supabase import create_client

API_URL = 'https://xcveldffznwantuastlu.supabase.co'
API_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InhjdmVsZGZmem53YW50dWFzdGx1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3MDI2MzQ1MDYsImV4cCI6MjAxODIxMDUwNn0.jfjqBFAMrdumZ8_S5BPmzAadKcvN9BZjm02xUcyIkPQ'
supabase = create_client(API_URL, API_KEY)

@st.cache_data(ttl=60)  # Cache the data for 60 seconds
def fetch_data():
    supabaseList = supabase.table('maintable').select('*').execute().data
    df = pd.DataFrame(supabaseList)
    df["DateTime"] = pd.to_datetime(df["created_at"])  # Convert "DateTime" column to datetime data type
    return df

st.set_page_config(page_title="Dashboard", layout='centered', initial_sidebar_state='collapsed')

st.markdown('### Smoke Sensor')

while True:
    df = fetch_data()
    
    # Get the most recent 5 entries
    df_recent = df.tail(5)
    
    st.plotly_chart(px.line(df_recent, x="DateTime", y="mq2", title='Smoke Sensor Readings', markers=True), use_container_width=True)

    st.plotly_chart(px.scatter(df_recent, x="DateTime", y="mq2", title='Smoke Sensor Readings'), use_container_width=True)

    st.plotly_chart(px.bar(df_recent, x="DateTime", y="mq2", title='Smoke Sensor Readings'), use_container_width=True)
    
    time.sleep(5)  # Wait for 5 seconds before fetching new data
    
    # Calculate maximum, minimum, and average values
    max_value = df_recent["mq2"].max()
    min_value = df_recent["mq2"].min()
    avg_value = df_recent["mq2"].mean()
    
    st.write("Maximum Value:", max_value)
    st.write("Minimum Value:", min_value)
    st.write("Average Value:", avg_value)
    
    st.rerun()  # Rerun the script to update the page

