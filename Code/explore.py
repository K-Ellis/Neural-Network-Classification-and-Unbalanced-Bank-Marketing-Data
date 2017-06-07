import pandas as pd

df = pd.read_csv("../Data/bank-additional_normalized.csv")

final_df = pd.concat([df["age"], df['y']], axis = 1)

print(final_df.head())