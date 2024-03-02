import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

from sklearn.metrics import mean_squared_log_error

color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')

# initialize
df_name = "none"
df_input = "none"


def process(df_name, df_input):
    # ----------------------------------------------------

    df = df_input
    columns = df.columns.tolist()
    columns[0] = 'Thời gian'
    columns[1] = 'Năng lượng'
    df.columns = columns

    df['Thời gian'] = pd.to_datetime(df['Thời gian'])
    df_sorted = df.sort_values(by='Thời gian')
    df_sorted['Thời gian'] = df_sorted['Thời gian'].astype('int64')
    df_sorted['Thời gian'] = pd.to_datetime(df_sorted['Thời gian'])
    df = df_sorted
    df = df.set_index('Thời gian')
    # df.to_csv('Data/AEP_SortData.csv')
    df.to_csv('Data/Sort_{df_name}.csv')

    # ----------------------------------------------------

    train = df.loc[df.index < '01-01-2015']
    test = df.loc[df.index >= '01-01-2015']

    # ----------------------------------------------------

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

    # ----------------------------------------------------

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

    # ----------------------------------------------------

    '''
    fi = pd.DataFrame(data=reg.feature_importances_,
                    index=reg.feature_names_in_,
                    columns=['importance'])
    fi.sort_values('importance').plot(kind='barh', title='Feature Importance')
    '''

    # ----------------------------------------------------

    test['prediction'] = reg.predict(X_test)
    df = df.merge(test[['prediction']], how='left',
                  left_index=True, right_index=True)

    # ----------------------------------------------------

    df_prediction = df['prediction'].loc[df.index >= '01-01-2015']
    # df_prediction.to_csv('Data/AEP_PredData.csv')
    df_prediction.to_csv('Data/Pred_{df_name}.csv')
