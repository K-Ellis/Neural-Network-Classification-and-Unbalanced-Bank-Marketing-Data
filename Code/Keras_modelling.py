from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import winsound

# fix random seed for reproducibility

# http://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

np.random.seed(7)

# load data
cleaning = ""
# cleaning = "_ordinals"
# cleaning = "_3cols"

input_location = "../Data/Cleaned" + cleaning + "/bank-additional-full_cleaned.csv"
output_location = "../Logs/CNN_outputs" + cleaning + "/CNN_%s.txt"

df = pd.read_csv(input_location)

X = np.array(df.drop('y', axis=1))
y = np.array(df['y'])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# create model
model = Sequential()
model.add(Dense(12, input_dim=53, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, epochs=10, batch_size=10)

train_scores = model.evaluate(X_train, y_train)
print("\nmetric names = ", model.metrics_names)
print("train scores = ", train_scores)
print("%s: %.2f%%" % (model.metrics_names[1], train_scores[1]*100))

scores = model.evaluate(X_test, y_test)
print("\nmetric names = ", model.metrics_names)
print("test scores = ", scores)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# calculate predictions
predictions = model.predict(X_test)
# print(predictions)
# round predictions
rounded = [round(x[0]) for x in predictions]
# print(rounded)


# df_y_test = pd.DataFrame(y_test)
# df_y_test_predictions = pd.DataFrame(df_y_test)
conf_matrix = confusion_matrix(y_test,rounded)
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

with open(output_location%Accuracy, "w") as f:
    f.write("train score ="+ str(train_scores[1])+"\n")
    f.write("test score ="+ str(Accuracy)+"\n")
    f.write(str(conf_matrix)+"\n")
    f.write("Accuracy = "+ str(Accuracy)+"\n")
    f.write("Sensitivity = "+ str(Sensitivity)+"\n")
    f.write("Specificity = "+ str(Specificity)+"\n")
    # f.write(str(model.summary()) + "\n")
    f.write(str(model.get_config()))


winsound.Beep(400,300)
winsound.Beep(300,300)