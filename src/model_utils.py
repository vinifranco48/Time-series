import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from src.exception import CustomException
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.feature_selection import RFECV

def feature_engineering(df, target, tranformation_log=False, roll=False, ewm=False, roll_mean=False, roll_std=False, roll_min=False, roll_max=False, to_sort=None, to_group=None, lags=None, windows=None, weights=None, min_periods=None, win_type=None, date_related=True, lag=False):
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
    except Exception as e:
        raise CustomException(e, sys)
    
    if tranformation_log:
        df[target] = np.log1p[target]
    
    if lag:
        df.sort_values(by=to_sort, axis=0, inplace=True)
        for lag in lags:
            df[f"sales_lag{lag}"] = df.groupby(to_group)[target].transform(lambda x: x.shift(lag))

        return df
    if roll:
        df.sort_values(by=to_sort, axis=0, inplace=True)

        if roll_mean:
            for window in windows:
                df['sales_roll_mean'+ str(window)]=df.groupby(to_group)[target].transform(lambda x: x.shift(1).rolling(window=window, min_periods=min_periods, win_type=win_type).mean())

        if roll_std:
            for window in windows:
                 df['sales_roll_mean'+ str(window)]=df.groupby(to_group)[target].transform(lambda x: x.shift(1).rolling(window=window, min_periods=min_periods, win_type=win_type).std())
        
        if roll_min:
            for window in windows:
                df['sales_roll_mean'+ str(window)]=df.groupby(to_group)[target].transform(lambda x: x.shift(1).rolling(window=window, min_periods=min_periods, win_type=win_type).min())

        if roll_max:
            for window in windows:
                df['sales_roll_mean'+ str(window)]=df.groupby(to_group)[target].transform(lambda x: x.shift(1).rolling(window=window, min_periods=min_periods, win_type=win_type).max())
        

        if ewm:
            for weight in weights:
                for lag in lags:
                    df['sales_ewm_w_' + str(weight) + '_lag_' + str(lag)] = df.groupby(to_group)[target].transform(lambda x: x.shift(lag).ewm(alpha=weight).mean())

        return df
def time_split(df, data_limit):
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

# Essa função cria features utilizando os lags, e util para treina modelos supervisionados
def series_to_supervised(data,window=1, lag=1, dropna=True):
    cols, name = list(), list()
    for i in range(window, 0, -1):
        cols.append(data.shift(i))
        names += [('%s(t)' % (col,i)) for col in data.columns]

    cols.append(data)
    names += [('%s(t)' % (col)) for col in data.columns]

    cols.append(data.shift(-lag))
    names += [('%s(t)' % (col, lag)) for col in data.columns]

    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropna:
        agg.drop(inplace=True)
    return agg

def cross_time_series(data, model, target, teste_size=None, gap=0, n_splits=5, log=False, verbose=False, display_score=True):
    try:

        t_split = TimeSeriesSplit(n_splits=n_splits, test_size=teste_size ,gap=gap)
        pontuação = []

        for fold, (train_index, val_index) in enumerate(t_split.split(data)):

            train = data.iloc[train_index]
            val = data.iloc[val_index]

            X_train = train.drop(columns=[target])
            y_train = train[target].copy()

            X_val = val.drop(columns=[target])
            y_val = val[target].copy()

            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)

            if log:
                score = np.sqrt(mean_squared_error(np.expm1(y_val), np.expm1(y_pred)))
            else:
                score = np.sqrt(mean_squared_error(y_val, y_pred))

            pontuação.append(score)

            if verbose:
                print('-'*30)
                print(f'Fold {fold}')
                print(f'Score (RMSE) = {round(score, 4)}')

            if not display_score:
                return pontuação
            
            print('-'*60)
            print(f'{type(model).__name__} s time series cross validation results:')
            print(f'Average validation score = {round(np.mean(pontuação), 4)}')
            print(f'Standard validation score = {round(np.std(pontuação), 4)}')

            return pontuação



    except Exception as e:
        raise CustomException(e, sys)

def cross_time_series_CNN_LSTM(data, model, target, test_size=None, gap=0, n_splits=5, log=False, verbose=False, display_score=True):
    try:
        t_split = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
        scores = []

        for fold, (train_index, val_index) in enumerate(t_split.split(data)):
            train = data.iloc[train_index]
            val = data.iloc[val_index]

            # Preparar dados de treino
            X_train = train.drop(columns=[target])
            y_train = train[target].copy()

            # Preparar dados de validação
            X_val = val.drop(columns=[target])
            y_val = val[target].copy()

            # Reshape dos dados para o formato [samples, subsequences, timesteps, features]
            subsequences = 2  # dividir cada amostra em 2 subsequências
            timesteps = X_train.shape[1] // subsequences
            X_train = X_train.values.reshape((X_train.shape[0], subsequences, timesteps, 1))
            X_val = X_val.values.reshape((X_val.shape[0], subsequences, timesteps, 1))

            # Compilar o modelo
            model.compile(optimizer=Adam(), loss='mse')

            # Treinar o modelo
            model.fit(X_train, y_train, epochs=50, verbose=0)

            # Fazer previsões
            y_pred = model.predict(X_val)

            # Calcular a métrica de desempenho (RMSE)
            if log:
                score = np.sqrt(mean_squared_error(np.expm1(y_val), np.expm1(y_pred)))
            else:
                score = np.sqrt(mean_squared_error(y_val, y_pred))

            scores.append(score)

            # Exibir informações de cada fold se verbose=True
            if verbose:
                print('-' * 30)
                print(f'Fold {fold}')
                print(f'Score (RMSE) = {round(score, 4)}')

        # Exibir resultados finais da validação cruzada
        if display_score:
            print('-' * 60)
            print(f'{type(model).__name__} time series cross validation results:')
            print(f'Average validation score = {round(np.mean(scores), 4)}')
            print(f'Standard validation score = {round(np.std(scores), 4)}')

        return scores

    except Exception as e:
        print(f'Error during cross-validation: {str(e)}')
        return None