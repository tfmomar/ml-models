import sklearn as sk
import pandas as pd


def create_window(df: pd.DataFrame, n_window):
    df.rolling(20)
