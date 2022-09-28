from create_ds.datasets import CreateDataset

if __name__ == '__main__':
    ds = CreateDataset(7, 2, '', 1000)
    ds.create_features()
