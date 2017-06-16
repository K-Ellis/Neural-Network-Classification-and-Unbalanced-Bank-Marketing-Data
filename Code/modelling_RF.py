import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import winsound
import numpy as np
np.random.seed(1)  # Set seed

df = pd.read_csv("../Data/Cleaned_ordinals/bank-additional-full_cleaned.csv")
X = df.drop('y',axis=1)
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

clf = RandomForestClassifier(n_estimators=50, random_state=1)

clf.fit(X_train, y_train)
train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)
predictions = clf.predict(X_test)

print("train score =", train_score)
print("test score =", test_score)
print(confusion_matrix(y_test,predictions))

with open("../Logs/RF_outputs_ordinals/RF_%s.txt"%(test_score), "w") as f:
    f.write("train score ="+ str(train_score)+"\n")
    f.write("test score ="+ str(test_score)+"\n")
    f.write(str(confusion_matrix(y_test, predictions))+"\n")
    f.write(str(clf) + "\n")

extra_reports = False
if extra_reports:
    print(classification_report(y_test,predictions))

    # coefs_ is a list of weight matrices, where weight matrix at index i represents the weights between layer i and
    # layer i+1.
    print(len(clf.coefs_))
    print(len(clf.coefs_[0]))
    # intercepts_ is a list of bias vectors, where the vector at index i represents  the bias values added to layer i+1.
    print(len(clf.intercepts_[0]))

    # MLP can fit a non-linear model to the training data. clf.coefs_ contains the weight matrices that constitute the
    # model parameters:
    print([coef.shape for coef in clf.coefs_])

# winsound.Beep(300,300)
winsound.Beep(400,300)
winsound.Beep(300,300)