import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
import winsound

# df = pd.read_csv("../Data/bank-additional-full_normalized.csv")
 # [[8759  371]
 # [ 545  622]]

df = pd.read_csv("../Data/Cleaned/bank-additional-full_cleaned.csv")
# hidden_layer_sizes=(13,13,13)
# train score = 0.916707131527
# test score = 0.910653588424
# [[8776  346]
#  [ 574  601]]

# hidden_layer_sizes=(10,10),
# train score = 0.915023793338
# test score = 0.912693017384
# [[8918  256]
#  [ 643  480]]

# hidden_layer_sizes=(10,10,10)
# train score = 0.916448156421
# test score = 0.911042051083
# [[8785  380]
#  [ 536  596]]

# hidden_layer_sizes=(15,15,15)
# train score = 0.917257453627
# test score = 0.910265125765
# [[8786  357]
#  [ 567  587]]


# hidden_layer_sizes=(20,20,20),
# train score = 0.918066750834
# test score = 0.914635330679
# [[8804  342]
#  [ 537  614]]

# hidden_layer_sizes=(20,20,20,20)
# train score = 0.923796575054
# test score = 0.911333398077
# [[8820  306]
#  [ 607  564]]

# hidden_layer_sizes=(50,50,50)
# train score = 0.952380952381
# test score = 0.900067980965
# [[8666  508]
#  [ 521  602]]

# hidden_layer_sizes=(172, 172, 172, 172, 172)
# train score = 0.985950600499
# test score = 0.889967951831
# [[8585  535]
#  [ 598  579]]

# hidden_layer_sizes=(30,20,10)
# train score = 0.928263895633
# test score = 0.902398756919
# [[8579  541]
#  [ 464  713]]

# hidden_layer_sizes=(20,15,10)
# train score = 0.921433427212
# test score = 0.911042051083
# [[8741  406]
#  [ 510  640]]

# hidden_layer_sizes=(15,10,5)
# train score = 0.91887604804
# test score = 0.908711275129
# [[8719  382]
#  [ 558  638]]

# hidden_layer_sizes=(10,8,5)
# train score = 0.913923149137
# test score = 0.912790133048
# [[8679  445]
#  [ 453  720]]

# hidden_layer_sizes=(8,6,4)
# train score = 0.907901977922
# test score = 0.909099737788
# [[8658  467]
#  [ 469  703]]

# hidden_layer_sizes=(10,5,5)
# train score = 0.913113851931
# test score = 0.910265125765
# [[8813  295]
#  [ 629  560]]

# hidden_layer_sizes=(10,5)
# train score = 0.916933734745
# test score = 0.916771875303
# [[8832  365]
#  [ 492  608]]

# hidden_layer_sizes=(5,5)
# train score = 0.91146288563
# test score = 0.918034378945
# [[8910  259]
#  [ 585  543]]

# hidden_layer_sizes=(13,13,13), solver=lbfgs
# score = 0.912693017384
# [[8750  378]
#  [ 521  648]]

# hidden_layer_sizes=(13,13,13), solver=lbfgs, alpha=0.1
# score = 0.907254540157
# [[8719  399]
#  [ 507  672]]

# hidden_layer_sizes=(5,5),solver="sgd"
# train score = 0.911009679195
# test score = 0.91201320773
# [[8816  354]
#  [ 552  575]]

# hidden_layer_sizes=(10,8,5),solver="sgd"
# train score = 0.911592373183
# test score = 0.911916092066
# [[8781  362]
#  [ 545  609]]

# hidden_layer_sizes=(10,8,5), solver="sgd", max_iter=500,alpha=0.1)
# train score = 0.9107183322
# test score = 0.915703602991
# [[8931  249]
#  [ 619  498]]

# hidden_layer_sizes=(10,8,5), alpha=0.1)
# train score = 0.915282768444
# test score = 0.911916092066
# [[8744  342]
#  [ 565  646]]

# df = pd.read_csv("../Data/bank-additional-full_normalized_vif.csv")
 # [[868  53]
 #  [ 57  52]]

X = df.drop('y',axis=1)
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y)

mlp = MLPClassifier(hidden_layer_sizes=(10,8,5),
                    # solver="sgd", #{‘lbfgs’, ‘sgd’, ‘adam’}, default ‘adam’,
                    max_iter=500,
                    alpha=0.1)
mlp.fit(X_train,y_train)

score = mlp.score(X_train, y_train)
print("train score =", score)
score = mlp.score(X_test, y_test)
print("test score =", score)


predictions = mlp.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

# coefs_ is a list of weight matrices, where weight matrix at index i represents the weights between layer i and
# layer i+1.
print(len(mlp.coefs_))
print(len(mlp.coefs_[0]))
# intercepts_ is a list of bias vectors, where the vector at index i represents  the bias values added to layer i+1.
print(len(mlp.intercepts_[0]))

# MLP can fit a non-linear model to the training data. clf.coefs_ contains the weight matrices that constitute the
# model parameters:
print([coef.shape for coef in mlp.coefs_])

winsound.Beep(300,300)
winsound.Beep(400,300)
winsound.Beep(300,300)