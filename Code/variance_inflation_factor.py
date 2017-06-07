import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

df = pd.read_csv("../Data/bank-additional_normalized.csv")

dfX = df.drop('y',axis=1)
# dfX = dfX.as_matrix()
# dfX = dfX.values
# dfX = df.reset_index().values


def calculate_vif_(X, thresh=5.0):
    variables = range(X.shape[1])
    dropped=True
    while dropped:
        dropped=False
        vif = [variance_inflation_factor(X.values, ix) for ix in range(X.shape[1])]
        print(vif)
        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + X.columns[maxloc] + '\' at index: ' + str(maxloc))
            X = X.drop(X.columns[maxloc], axis=1)
            dropped=True

    print('Remaining variables:')
    print(X.columns)
    return X

dfvif = calculate_vif_(dfX)

final_df = pd.concat([dfvif, df['y']], axis = 1)

final_df.to_csv('../Data/bank-additional-full_normalized_vif.csv', index =
False)

# # For each X, calculate VIF and save in dataframe
# vif = pd.DataFrame()
# vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
# vif["features"] = X.columns
# # Step 3: Inspect VIF Factors
# vif.round(1)