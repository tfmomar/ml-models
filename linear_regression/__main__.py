import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
import utils as U
import argparse

if __name__ == '__main__':
    arguments = argparse.ArgumentParser()
    arguments.add_argument('--path')
    arguments.add_argument('--index_col', default=None)
    arguments.add_argument('--window_size', default='10')

    df = U.read_df(**vars(arguments.parse_args()))
    df_features = U.roll_window(df.iloc[:, :-1], **vars(arguments.parse_args()))
    df_target = U.roll_window(df.iloc[:, -1], **vars(arguments.parse_args()))

    i = 0
    lr = LinearRegression()
    x_train, y_train = [], []
    for x_pred, y_pred in zip(df_features['feature 5'], df_target):
        x_pred = np.array([x_pred]).reshape(-1, 1)

        if i > 0:
            lr.fit(x_train, y_train)
            y_mse = lr.predict(np.array([x_pred]).reshape(-1, 1))

            print(f"ITER {i}\n", y_mse)
            print("MSE:", mse(y_true=y_pred, y_pred=y_mse))
        else:
            i += 1
        x_train, y_train = x_pred, y_pred
    """
    lr = LinearRegression()
    lr.fit([df_features['feature 1'].values], df_target.values)
    print(lr.coef_)
    print(lr.predict([df_features['feature 1'].values[500]]))
    """


