import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import pickle

from sklearn.metrics import mean_squared_error

color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')

# types of time series data
# read raw data file
df = pd.read_csv('./Data/Sorted_PJME_hourly.csv')
df = df.set_index('Thời gian')
df.index = pd.to_datetime(df.index)
df.plot(style='.',
        figsize=(15, 5),
        color=color_pal[0],
        title='Năng lượng điện tiêu thụ (Đơn vị: MegaWatt)')

# train / test split
train = df.loc[df.index < '01-01-2015']
test = df.loc[df.index >= '01-01-2015']

fig, ax = plt.subplots(figsize=(15, 5))
train.plot(ax=ax, label='Bộ Huấn luyện',
           title='Bộ dữ liệu Huấn luyện / Kiểm tra')
test.plot(ax=ax, label='Bộ Kiểm tra')
ax.axvline('01-01-2015', color='black', ls='--')
ax.legend(['Bộ Huấn luyện', 'Bộ Kiểm tra'])

# Week data
df.loc[(df.index > '01-01-2010') & (df.index < '01-08-2010')] \
    .plot(figsize=(15, 5), title='Dữ liệu theo Tuần')


def create_features(df):
    """
    Create time series features based on time series index.
    """
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df


train = create_features(train)
test = create_features(test)

FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year']
TARGET = 'Năng lượng'

X_train = train[FEATURES]
y_train = train[TARGET]
X_test = test[FEATURES]
y_test = test[TARGET]

reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
                       n_estimators=1000,
                       early_stopping_rounds=50,
                       objective='reg:linear',
                       max_depth=3,
                       learning_rate=0.01)
reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100)

# feature importance
fi = pd.DataFrame(data=reg.feature_importances_,
                  index=reg.feature_names_in_,
                  columns=['importance'])
fi.sort_values('importance').plot(kind='barh', title='Feature Importance')

# forecast on test
test['prediction'] = reg.predict(X_test)
df = df.merge(test[['prediction']], how='left',
              left_index=True, right_index=True)
ax = df[['Năng lượng']].plot(figsize=(15, 5))
df['prediction'].plot(ax=ax, style='.')
plt.legend(['Dữ liệu chuẩn', 'Dữ liệu được dự đoán'])
ax.set_title('Dữ liệu và Dự đoán thô')

# write prediction data file
df_prediction = df['prediction'].loc[df.index >= '01-01-2015']
df_prediction.to_csv('Data/YearData.csv')
