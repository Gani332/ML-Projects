import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")

print(data.head())

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]  # Attribute: unique to each student
predict = "G3"  # We want labels based on attributes
x = np.array(data.drop([predict], 1))  # All features/attributes
y = np.array(data[predict])  # All labels
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# Splits x and y into 4 different arrays
# x_train deals with x array and y_train deals with y array

# x_test and y_test tests accuracy of the model we create
# test_size splits 10% of data so comp can't see the data so it can actually predict instead of memorise ans
"""
best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)  # Giving a best fit line to data
    acc = linear.score(x_test, y_test)  # Going to return a value that tests the accuracy of model
    print(acc)

    if acc > best:
        best = acc
        with open("studentmodel.pickle", "wb") as f:  #Saves a pickle file so we can use it - don't need to retrain model every time
            pickle.dump(linear, f) """

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

print("co: " '\n', linear.coef_)
print("Intercept" '\n', linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

p = "G1"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()
