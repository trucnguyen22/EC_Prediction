# pip install streamlit fbprophet yfinance plotly
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import Model_XGBoost

from datetime import datetime, time
from plotly import graph_objs as go

color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')


# ----------------------------------------------------


st.title('Xây dựng mô hình dự đoán')
st.subheader('Mô hình dự đoán năng lượng tiêu thụ (Đơn vị: MegaWatt)')
# Edit sidebar
st.sidebar.text('')
st.sidebar.markdown(
    """
    <h1 style='text-align: center; color: #008080;'>My Streamlit App</h1>
    <p style='text-align: center;'>
        <img src='https://upload.wikimedia.org/wikipedia/commons/f/fd/Logo_of_the_Tr%E1%BA%A7n_%C4%90%E1%BA%A1i_Ngh%C4%A9a_High_School_for_the_Gifted.svg' alt='Logo' style='width: 200px;'>
    </p>
    """, unsafe_allow_html=True
)
# Add sidebar buttons
st.sidebar.button("Button1")
st.sidebar.button("Button2")

uploaded_file = st.file_uploader('Tệp dữ liệu cần được dự đoán')
if uploaded_file is not None:
    df_name = uploaded_file
    df_input = pd.read_csv(uploaded_file)

Model_XGBoost.process(df_name, df_input)

# ----------------------------------------------------
# Plot Data Function


def plotData(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.iloc[:, 0], y=df.iloc[:, 1]))
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


# ----------------------------------------------------
# Initial Data

# df = pd.read_csv('./Data/PJME_SortData.csv')
df = pd.read_csv('./Data/Sort_{df_name}.csv')
df['Thời gian'] = pd.to_datetime(df['Thời gian'])
select_time = st.slider(
    "When do you start?",
    value=(df['Thời gian'].min().to_pydatetime(),
           df['Thời gian'].max().to_pydatetime()),
)
filtered_df = df[(df['Thời gian'] >= select_time[0]) &
                 (df['Thời gian'] <= select_time[1])]
st.header('Năng lượng tiêu thụ ban đầu')
plotData(filtered_df)


# ----------------------------------------------------
# Predicted Data


# df_prediction = pd.read_csv('./Data/PJME_PredData.csv')
df_prediction = pd.read_csv('./Data/Pred_{df_name}.csv')
df_prediction['Thời gian'] = pd.to_datetime(df_prediction['Thời gian'])
select_time_prediction = st.slider(
    "When do you start?",
    value=(df_prediction['Thời gian'].min().to_pydatetime(),
           df_prediction['Thời gian'].max().to_pydatetime()),
)
filtered_df_prediction = df_prediction[(df_prediction['Thời gian'] >= select_time_prediction[0]) &
                                       (df_prediction['Thời gian'] <= select_time_prediction[1])]
st.header('Năng lượng tiêu thụ được dự đoán')
plotData(filtered_df_prediction)
