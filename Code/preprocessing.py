# taken from https://gist.github.com/mick001/9db3609e49e98069316267349abc37b5#file-preprocessing_b-py

import pandas as pd
import numpy as np
from sklearn import preprocessing


def clean_banking():

    csvs = ["bank", "bank-additional", "bank-additional-full", "bank-full"]
    # Load dfs
    for csv in csvs:
        df = pd.read_csv('../Data/%s.csv' % (csv), sep=";")
        # df = pd.read_csv('../Data/bank-additional-full.csv', sep =";")

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
            if cat_var != "y":
                df_cat = pd.concat([df_cat, pd.get_dummies(df[cat_var], prefix=cat_vars[i], drop_first=True)], axis=1)

        # # Get dummy variables for categorical vars
        # job = pd.get_dummies(df['job'])
        # marital = pd.get_dummies(df['marital'])
        # education = pd.get_dummies(df['education'])
        # default = pd.get_dummies(df['default'])
        # housing = pd.get_dummies(df['housing'])
        # loan = pd.get_dummies(df['loan'])
        # contact = pd.get_dummies(df['contact'])
        # month = pd.get_dummies(df['month'])
        # day = pd.get_dummies(df['day_of_week'])
        # poutcome = pd.get_dummies(df['poutcome'])

        # Map variable to predict
        dict_map = dict()
        y_map = {'yes' : 1,'no' : 0}
        dict_map['y'] = y_map
        df = df.replace(dict_map)
        df_y = df['y']

        df_num = df[quantit]
        df1_names = df_num.keys().tolist()

        # Scale quantitative variables
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(df_num)
        df_num = pd.DataFrame(x_scaled)
        df_num.columns = df1_names

        # Get final df
        final_df = pd.concat([df_num, df_cat, df_y], axis=1)
                              # job,
                              # marital,
                              # education,
                              # default,
                              # housing,
                              # loan,
                              # contact,
                              # day,
                              # month,
                              # poutcome,

        # Quick check
        # print(final_df.head())

        # Save df
        # final_df.to_csv('../Data/bank_normalized.csv', index = False)
        # final_df.to_csv('../Data/bank-additional-full_cleaned.csv', index = False)

        final_df.to_csv('../Data/Cleaned/%s_cleaned.csv'%(csv), index = False)

if __name__ == "main":
    clean_banking()