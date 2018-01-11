import pandas as pd

def one_hot_encoding(df):
    cats = df.ix[:, df.dtypes == object]
    one_hot = pd.get_dummies(cats)
    df = df.ix[:, df.dtypes != object]
    df = pd.concat([df, one_hot], axis = 1)

    return df
