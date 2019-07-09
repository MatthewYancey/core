import pandas as pd

# does just what it says
def one_hot_encoding(df):
    cats = df.loc[:, df.dtypes == object]
    one_hot = pd.get_dummies(cats)
    df = df.loc[:, df.dtypes != object]
    df = pd.concat([df, one_hot], axis = 1)

    return df
