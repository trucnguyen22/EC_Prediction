# pip install streamlit fbprophet yfinance plotly
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from datetime import datetime, time
from plotly import graph_objs as go


color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')
st.title('Xây dựng mô hình dự đoán năng lượng tiêu thụ (Đơn vị: MegaWatt)')


st.sidebar.text('')
st.sidebar.text('')
st.sidebar.text('')


st.file_uploader('File uploader')


# Plot Data Function
def plot_raw_data(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.iloc[:, 0], y=df.iloc[:, 1]))
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


# Initial Data
df = pd.read_csv('./Data/SortData.csv')
# Create Streamlit slider for selecting date range
df['Thời gian'] = pd.to_datetime(df['Thời gian'])
start_date = df['Thời gian'].min().to_pydatetime()
end_date = df['Thời gian'].max().to_pydatetime()
select_time = st.slider(
    "When do you start?",
    value=(start_date, end_date),
)
st.write("Start time:", select_time[0], select_time[1])
# Filter data based on date range
filtered_df = df[(df['Thời gian'] >= select_time[0]) &
                 (df['Thời gian'] <= select_time[1])]
st.header('Năng lượng tiêu thụ ban đầu')
plot_raw_data(filtered_df)


# Predicted Data
df_prediction = pd.read_csv('./Data/PredData.csv')
# Create Streamlit slider for selecting date range
df_prediction['Thời gian'] = pd.to_datetime(df_prediction['Thời gian'])
start_date_prediction = df_prediction['Thời gian'].min().to_pydatetime()
end_date_prediction = df_prediction['Thời gian'].max().to_pydatetime()
select_time_prediction = st.slider(
    "When do you start?",
    value=(start_date_prediction, end_date_prediction),
)
st.write("Start time:", select_time_prediction[0], select_time_prediction[1])
# Filter data based on date range
filtered_df_prediction = df_prediction[(df_prediction['Thời gian'] >= select_time_prediction[0]) &
                                       (df_prediction['Thời gian'] <= select_time_prediction[1])]
st.header('Năng lượng tiêu thụ được dự đoán')
plot_raw_data(filtered_df_prediction)
