# taken from https://gist.github.com/mick001/9db3609e49e98069316267349abc37b5#file-preprocessing_b-py

import pandas as pd
import numpy as np
from sklearn import preprocessing

# cols_to_delete = ["day", "month", "contact"]
ordinal_vars = ["education", "poutcome"]
csvs = ["bank", "bank-additional", "bank-additional-full", "bank-full"]
# Load dfs
for csv in csvs:
    df = pd.read_csv('../Data/%s.csv' % (csv), sep=";")
    # df = pd.read_csv('../Data/bank-additional-full.csv', sep =";")

    # for col in cols_to_delete:
    #     if col in df:
    #         del df[col]

    # Variables names
    col_names = df.columns.tolist()

    # Categorical vars
    cat_vars = []
    for col in df:
        if df[col].dtype == object:
            cat_vars.append(col)

    # cat_vars = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact',
    #           'month','poutcome','y', "day_of_week"]

    # Quantitative vars
    quantit = [i for i in col_names if i not in cat_vars]

    df_cat = pd.DataFrame()
    for i, cat_var in enumerate(cat_vars):
        if cat_var != "y" and cat_var not in ordinal_vars:
            df_cat = pd.concat([df_cat, pd.get_dummies(df[cat_var], prefix=cat_vars[i], drop_first=True)], axis=1)

    # df.education.replace("unknown", df.education.mode()[0], inplace=True)
    if len(ordinal_vars)>0:
        if csv == "bank-additional" or "bank-additional-full":
            df["poutcome"] = df["poutcome"].map({"nonexistent": 0, "failure": -1, "success": 1})
            df_cat = pd.concat([df_cat, pd.get_dummies(df["education"], prefix="education", drop_first=True)], axis=1)
            ordinal_dfs = df["poutcome"]
        else:
            df["education"] = df["education"].map({"primary":0, "secondary":1, "tertiary":2, "unknown":1})
            df["poutcome"] = df["poutcome"].map({"other": 0, "failure": -1, "success": 1, "unknown": 0})
            ordinal_dfs = [df["education"] ,df["poutcome"]]

    # Map variable to predict
    dict_map = dict()
    y_map = {'yes' : 1,'no' : 0}
    dict_map['y'] = y_map
    df = df.replace(dict_map)
    df_y = df['y']
    # df["y"] = df["y"].astype(bool)
    # df["y"] = df["y"].astype(int)
    # df_y = pd.DataFrame(df["y"])

    df_num = df[quantit]
    df1_names = df_num.keys().tolist()

    # Scale quantitative variables
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(df_num)
    df_num = pd.DataFrame(x_scaled)
    df_num.columns = df1_names

    # Get final df
    final_df = pd.concat([ordinal_dfs,df_num, df_cat, df_y], axis=1)


    final_df.to_csv('../Data/Cleaned_ordinals/%s_cleaned.csv'%(csv), index = False)