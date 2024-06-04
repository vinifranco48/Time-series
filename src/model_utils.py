import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from src.exception import CustomException

def feature_engineering(df, target, tranformation_log=False):
    try:
        # Garantir que o índice é um DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
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
    except Exception as e:
        raise CustomException(e, sys)
    
    if tranformation_log:
        df[target] = np.log1p[target]
    
    if lag:
        df.sort_values(by=to_sort, axis=0, inplace=True)
        for lag in lags:
            df[f"sales_lag{lag}"] = df.groupby(to_group)[target].transform(lambda x: x.shift(lag))

        return df
        
def time_split(df, data_limit, t):
    try:
        train = df.loc[df.index < data_limit]
        test = df.loc[df.index >= data_limit]
        return train, test
    except Exception as e:
        raise CustomException(e, sys)
    
    

def time_split_plot(train, test, data_limit):
    try:
        figure, ax = plt.subplots(figsize=(20, 7))

        train.plot(ax=ax, label="Train", y="sales")
        test.plot(ax=ax, label="Test",y="sales")

        ax.axvline(data_limit, color="black",ls="--")

        plt.title("Time series train and teste split", fontsize=27, fontweight="bold",loc="left", pad=25)
        plt.xlabel("Date", loc="left",labelpad=25)
        plt.ylabel("Sales", loc="top", labelpad=25)
        plt.xticks(rotation=0)
        plt.legend(loc="upper left")
        plt.show()
    except Exception as e:
        raise CustomException(e, sys)

