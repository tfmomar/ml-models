import pandas as pd
import os


def read_df(**kwargs):
    path, index_col = kwargs['path'], int(kwargs['index_col']) if kwargs['index_col'] is not None else None
    return pd.read_csv(path, index_col=index_col)


def roll_window(df: pd.DataFrame, **kwargs):
    window_size = int(kwargs['window_size'])
    return df.rolling(window_size)


def roll_std(df: pd.DataFrame.rolling):
    return df.std()


def df_to_csv(df: pd.DataFrame, path, file_name):
    assert os.path.isdir(path), f"Path {path} does not exist"
    df.to_csv(path+file_name)
