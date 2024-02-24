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


st.date_input('Date input')
select_time = st.slider(
    "When do you start?",
    value=(datetime(2002, 1, 1, 1, 00), datetime(2018, 8, 3, 0, 00)),)
# format="MM/DD/YY - hh:mm")
st.write("Start time:", select_time[0], select_time[1])


# Plot Data Function
def plot_raw_data(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.iloc[:, 0], y=df.iloc[:, 1]))
    title_text = 'Dữ liệu năng lượng tiêu thụ ban đầu' if df.columns[
        1] == 'Năng lượng' else 'Dữ liệu năng lượng được dự đoán'
    fig.layout.update(title_text=title_text, xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


# Initial Data
df = pd.read_csv('./Data/SortData.csv')
plot_raw_data(df)


# Predicted Data
df_pred = pd.read_csv('./Data/PredData.csv')
plot_raw_data(df_pred)
