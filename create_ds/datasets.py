import random

import numpy as np
import os
import pandas as pd


class CreateDataset:
    def __init__(self, n_features, n_relation, formula, time, min_value=-100, max_value=100, path='./datasets/'):
        assert n_features > n_relation, f'Error: n_features cannot be lower or equal than n_relation:' + \
                                            f'{n_features} < {n_relation}'
        assert os.path.isdir(path), f'Path {path} does not exist'
        assert path.endswith('/'), 'Invalid path reference'
        self.n_features = range(n_features)
        self.n_relation = range(n_relation)
        self.time = time
        self.equation = formula
        self.min = min_value
        self.max = max_value
        self.amplitude = np.random.uniform(min_value, max_value)
        self.data = []
        self.b = np.random.uniform(min_value, max_value) / 10
        self.path = path

    def create_features(self):
        self.data = []
        for t in range(self.time):
            each_row = []
            for col in self.n_features:
                if col in self.n_relation:
                    each_row.append(self.related_data(t, col))
                else:
                    each_row.append(self.non_related_data())
            each_row.append(self.create_y(t))
            self.data.append(each_row)

        self.export_ds('ds_test')

    def create_y(self, t):
        # TODO create a function to eval the formula given in the input
        return self.amplitude * np.sin(np.pi * t / 100)

    def related_data(self, t, col):
        # TODO create a function to eval the formula given in the input
        if col % 2 == 0:
            return self.amplitude * np.cos(np.pi * t / 100) + self.b
        else:
            return self.amplitude * np.sin(np.pi * t / 100) + self.b

    def non_related_data(self):
        return np.random.uniform(self.min, self.max)

    def export_ds(self, name):
        assert '.csv' not in name, 'Name of the file does contain .csv, please remove it'
        pd.DataFrame(self.data, index=None, columns=[f'feature {i}' for i in self.n_features] + ['target']) \
            .to_csv(self.path+name+'.csv')
