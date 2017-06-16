import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import winsound
import numpy as np
np.random.seed(1)  # Set seed

cleaning = ""
# cleaning = "_ordinals"
# cleaning = "_3cols"

input_location = "../Data/Cleaned" + cleaning + "/bank-additional-full_cleaned.csv"
output_location = "../Logs/MLP_outputs" + cleaning + "/MLP_%s.txt"

df = pd.read_csv(input_location)
X = df.drop('y', axis=1)
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

clf = MLPClassifier(hidden_layer_sizes=(10, 8, 5),
                    solver="lbfgs", #{‘lbfgs’, ‘sgd’, ‘adam’}, default ‘adam’,
                    max_iter=500,
                    alpha=0.1)
mlp_info = {"hidden_layer_sizes":clf.hidden_layer_sizes,
           "solver": clf.solver,
           "max_iter": clf.max_iter,
           "alpha":clf.alpha}
clf.fit(X_train, y_train)
train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)
predictions = clf.predict(X_test)
conf_matrix = confusion_matrix(y_test,predictions)
print("train score =", train_score)
print("test score =", test_score)
print(conf_matrix)

TN = conf_matrix[0][0]
TP = conf_matrix[1][1]
FN = conf_matrix[1][0]
FP = conf_matrix[0][1]
Accuracy = (TP+TN)/(TN+TP+FN+FP)
Sensitivity = TP/(TP+FN)
Specificity = TN/(TN+FP)
print("Accuracy = ", Accuracy)
print("Sensitivity = ", Sensitivity)
print("Specificity = ", Specificity)

with open(output_location%test_score, "w") as f:
    f.write(str(mlp_info) + "\n")
    f.write("train score ="+ str(train_score)+"\n")
    f.write("test score ="+ str(test_score)+"\n")
    f.write(str(conf_matrix)+"\n")
    f.write("Accuracy = "+ str(Accuracy)+"\n")
    f.write("Sensitivity = "+ str(Sensitivity)+"\n")
    f.write("Specificity = "+ str(Specificity)+"\n")
    f.write(str(clf))

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