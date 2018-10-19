import pandas as pd
import os

# does just what it says
def one_hot_encoding(df):
    cats = df.ix[:, df.dtypes == object]
    one_hot = pd.get_dummies(cats)
    df = df.ix[:, df.dtypes != object]
    df = pd.concat([df, one_hot], axis = 1)

    return df


# function for saving the model results
def save_model_results(file_name, accuracy):
    # if it exists
    if os.path.isfile(file_name):
        df_res = pd.read_csv(file_name)
        count = df_res.loc[df_res.shape[0] - 1, 'model_number'] + 1
        i = df_res.shape[0]
        df_res.loc[i, 'model_number'] = count
        df_res.loc[i, 'accuracy'] = accuracy
        print('Appending Results')
        df_res.to_csv(file_name, index=False)
    # if it doesn't exist
    else:
        df_res = pd.DataFrame(columns=['model_number', 'accuracy'])
        df_res.loc[0] = [1, accuracy]
        df_res.to_csv(file_name, index=False)

    print(df_res.head())