import pandas as pd

df = pd.read_csv("../Data/bank-additional.csv", sep =";")
i = 0

cat_vars = []

for col in df:
    if df[col].dtype == object:
        cat_vars.append(col)

print(cat_vars)