# pip install streamlit fbprophet yfinance plotly
import streamlit as st
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from datetime import date
from plotly import graph_objs as go


color_pal = sns.color_palette()
# plt.style.use('fivethirtyeight')

st.title('Xây dựng mô hình dự đoán năng lượng tiêu thụ (Đơn vị: MegaWatt)')

# Plot Data Function


def plot_raw_data(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.iloc[:, 0], y=df.iloc[:, 1]))
    fig.layout.update(
        title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


# Initial Data
df = pd.read_csv('./PJME_hourly.csv')
plot_raw_data(df)


# Predicted Data
df_pred = pd.read_csv('./PredictOP.csv')
plot_raw_data(df_pred)