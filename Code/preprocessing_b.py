# taken from https://gist.github.com/mick001/9db3609e49e98069316267349abc37b5#file-preprocessing_b-py

import pandas as pd
import numpy as np
from sklearn import preprocessing

# Load data
# data = pd.read_csv('../Data/bank.csv', sep = ";")
data = pd.read_csv('../Data/bank-additional-full.csv', sep = ";")

# Variables names
var_names = data.columns.tolist()

# Categorical vars
categs = ['job','marital','education','default','housing','loan','contact',
          'month','poutcome','y', "day_of_week"]
# Quantitative vars
quantit = [i for i in var_names if i not in categs]

# Get dummy variables for categorical vars
job = pd.get_dummies(data['job'])
marital = pd.get_dummies(data['marital'])
education = pd.get_dummies(data['education'])
default = pd.get_dummies(data['default'])
housing = pd.get_dummies(data['housing'])
loan = pd.get_dummies(data['loan'])
contact = pd.get_dummies(data['contact'])
month = pd.get_dummies(data['month'])
day = pd.get_dummies(data['day_of_week'])
poutcome = pd.get_dummies(data['poutcome'])

# Map variable to predict
dict_map = dict()
y_map = {'yes':1,'no':0}
dict_map['y'] = y_map
data = data.replace(dict_map)
label = data['y']

df1 = data[quantit]
df1_names = df1.keys().tolist()

# Scale quantitative variables
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(df1)
df1 = pd.DataFrame(x_scaled)
df1.columns = df1_names

# Get final df
final_df = pd.concat([df1,
                      job,
                      marital,
                      education,
                      default,
                      housing,
                      loan,
                      contact,
                      day,
                      month,
                      poutcome,
                      label], axis=1)

# Quick check
print(final_df.head())

# Save df
# final_df.to_csv('../Data/bank_normalized.csv', index = False)
final_df.to_csv('../Data/bank-additional-full_normalized.csv', index = False)
