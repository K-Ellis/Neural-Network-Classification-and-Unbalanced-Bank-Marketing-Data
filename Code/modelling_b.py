import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix

# df = pd.read_csv("../Data/bank-additional-full_normalized.csv")
df = pd.read_csv("../Data/bank-additional-full_normalized_vif.csv")


X = df.drop('y',axis=1)
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y)

mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)
mlp.fit(X_train,y_train)

predictions = mlp.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

# coefs_ is a list of weight matrices, where weight matrix at index i
# represents the weights between layer i and layer i+1.
print(len(mlp.coefs_))
print(len(mlp.coefs_[0]))
# intercepts_ is a list of bias vectors, where the vector at index i
# represents  the bias values added to layer i+1.
print(len(mlp.intercepts_[0]))