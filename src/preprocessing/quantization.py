import pandas as pd
import os


def count_func(x, cat):
    if cat == 'count':
        try:
            i = int(x)
        except ValueError:
            return x
        if i == 0:
            return '0'
        elif i >= 1 and i <= 10:
            return 'between 1 and 10'
        elif i >= 11 and i <= 100:
            return 'between 11 and 100'
        elif i >= 101 and i <= 1000:
            return 'between 101 and 1000'
        elif i > 1000:
            return 'more than 1000'
    else:
        return x


def area_func(x, cat):
    if cat == 'area':
        area = x.split('m2')[0]
        area = int(area)

        if area == 0:
            return '0'
        elif area >= 1 and area <= 10:
            return 'between 1 and 10'
        elif area >= 11 and area <= 100:
            return 'between 11 and 100'
        elif area >= 101 and area <= 1000:
            return 'between 101 and 1000'
        elif area > 1000:
            return 'more than 1000'
    else:
        return x


datasets = ["RSVQA-LR", "RSVQA-HR", "RSVQAxBEN"]
for data_dir in datasets:
    traindf = pd.read_csv(os.path.join("datasets", data_dir, 'traindf.csv'))
    valdf = pd.read_csv(os.path.join("datasets", data_dir, 'valdf.csv'))
    testdf = pd.read_csv(os.path.join("datasets", data_dir, 'testdf.csv'))
    dfs = {'train': traindf, 'val': valdf, 'test': testdf}

    categories = traindf['category'].unique()

    if 'area' in categories:
        print('Processing area category')
        for split, df in dfs.items():
            df['answer'] = df.apply(lambda x: area_func(x['answer'], x['category']), axis=1)

    elif 'count' in categories:
        print('Processing count category')
        for split, df in dfs.items():
            df['answer'] = df.apply(lambda x: count_func(x['answer'], x['category']), axis=1)

    dfs['train'].to_csv(os.path.join("datasets", data_dir, 'traindf.csv'), index=False)
    dfs['val'].to_csv(os.path.join("datasets", data_dir, 'valdf.csv'), index=False)
    dfs['test'].to_csv(os.path.join("datasets", data_dir, 'testdf.csv'), index=False)
