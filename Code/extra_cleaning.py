# taken from https://gist.github.com/mick001/9db3609e49e98069316267349abc37b5#file-preprocessing_b-py

import pandas as pd
import numpy as np
from sklearn import preprocessing


def get_df(csv_name):
	df = pd.read_csv('../Data/%s.csv' % (csv_name), sep=";")
	return df


def remove_cols(df, cols):
	for col in cols:
		del df[col]
	return df


def get_cats_and_quants(df, cols_to_be_mapped):
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
	df_num = df[quantit]

	df_cat = pd.DataFrame()
	df_to_map = pd.DataFrame()
	for i, cat_var in enumerate(cat_vars):
		if cat_var not in cols_to_be_mapped:
		# if cat_var != "y":
			df_cat = pd.concat([df_cat, pd.get_dummies(df[cat_var], prefix=cat_vars[i], drop_first=True)], axis=1)
		else:
			df_to_map = pd.concat([df_cat, df[cat_var]], axis=1)

	return df_cat, df_num, df_to_map


def binary(df, var):
	# Map variable to predict
	# dict_map = dict()
	# y_map = {'yes': 1, 'no': 0}
	# dict_map['y'] = y_map
	# df = df.replace(dict_map)
	# df_y = df[var]
	df[var] = df[var].astype(bool)
	df[var] = df[var].astype(int)
	df_out = pd.DataFrame(df[var])
	return df_out


def scale_quants(df_num):
	df1_names = df_num.keys().tolist()

	# Scale quantitative variables
	min_max_scaler = preprocessing.MinMaxScaler()
	x_scaled = min_max_scaler.fit_transform(df_num)
	df_num2 = pd.DataFrame(x_scaled)
	df_num2.columns = df1_names
	return df_num2


def concat_dfs(dfs):
	# Get final df
	# for df in dfs:
	# 	print(df.head())
	final_df = pd.concat([dfs], axis=1)
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

	return final_df


def save_csv(df, csv_name):
	df.to_csv('../Data/Cleaned2/%s_cleaned.csv' % (csv_name), index=False)


if __name__ == "__main__":
	# csv_names = ["bank", "bank-additional", "bank-additional-full", "bank-full"]
	csv_names = ["bank"]
	for csv_name in csv_names:
		df = get_df(csv_name)

		cols_to_be_removed = ["poutcome"]
		df = remove_cols(df, cols_to_be_removed)

		target_var = "y"
		cols_to_be_mapped = [target_var]
		df_cat, df_num, df_to_map = get_cats_and_quants(df, cols_to_be_mapped)

		df_y = binary(df, target_var)

		df_num = scale_quants(df_num)

		final_df = concat_dfs([df_cat, df_num, df_y])

		save_csv(final_df, csv_name)

		print(final_df.head())