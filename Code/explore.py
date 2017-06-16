import pandas as pd
pd.set_option('display.max_columns', 55)
# df = pd.read_csv("../Data/bank-full.csv", sep =";")
df = pd.read_csv("../Data/Cleaned_ordinals/bank-additional-full_cleaned.csv")
# df = pd.read_csv("../Data/bank-additional-full.csv", sep =";")
print(df["education_basic.9y"])
# print(df["poutcome"].value_counts())
print(df.head())
# print(df.isnull().sum())
# print(df.education.value_counts())
#
# unknown_edu = df[df.education == "unknown"]
# # print(unknown_edu.job.value_counts())
#
# # jobs = df.job.columns.tolist()
# # for job in jobs:
# df.education.replace("unknown", df.education.mode()[0], inplace=True)
# print(df.education.value_counts())
# # job_student = df[df.job == "student"]
# print(job_student.education.mode()[0])


#link education to job