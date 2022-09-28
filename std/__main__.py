import pandas as pd

import utils as U
import argparse

if __name__ == '__main__':
    arguments = argparse.ArgumentParser()
    arguments.add_argument('--path')
    arguments.add_argument('--index_col', default=None)
    arguments.add_argument('--window_size', default='10')

    df = U.read_df(**vars(arguments.parse_args()))
    df_roll = U.roll_window(df, **vars(arguments.parse_args()))
    df_rol_std = U.roll_std(df_roll)

    # print([*df_roll])
    # print(df_rol_std)
    df_features_std = df_rol_std.iloc[:, :-1]
    print("df_features_std", df_features_std)

    feature_values = {}
    for value in df_features_std.columns:
        feature_values[value] = []

    y_values = df_rol_std.iloc[:, -1]
    print("y_values", y_values)

    for i in range(len(df_features_std)):
        print(df_features_std.values[i])
        for col, value in zip(feature_values, df_features_std.values[i]):
            feature_values[col].append(value/y_values[i])

    df_std_final = pd.DataFrame.from_dict(feature_values).dropna()
    print("feature_values\n", df_std_final)
    df_std_final['target'] = df_rol_std.iloc[:, -1]
    U.df_to_csv(df_std_final, './results/', 'df_std.csv')

