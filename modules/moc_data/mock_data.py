import pandas as pd
import numpy as np
import datetime
import random as rn

def mock_data(df, n):
    df_new = pd.DataFrame(index=range(n), columns=df.columns)

    for i in range(df.shape[1]):
        if str(df.dtypes[i])[:10] == 'datetime64':
            print('date')
            new_dates = pd.date_range('1/1/2000', periods=n, freq='D')
            df_new.ix[:, i] = new_dates

        elif df.dtypes[i] == 'object':
            print('string')
            N = len(set(list(df.ix[:, i])))
            new_strings = ['factor' + str(N_i) for N_i in range(N)]
            new_strings = [rn.choice(new_strings) for n_i in range(n)]
            df_new.ix[:, i] = new_strings

        else:
            print('numeric')
            max_num = max(df.ix[:, i])
            min_num = min(df.ix[:, i])
            new_num = [rn.uniform(min_num, max_num) for n_i in range(n)]
            df_new.ix[:, i] = new_num

    return(df_new)
