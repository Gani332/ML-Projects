import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics

cancer = datasets.load_breast_cancer()  # Data set

#print(cancer.feature_names)
#print(cancer.target_names)

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)    # Training Model

#print(x_train, y_train)

classes = ['malignant', 'benign']

clf = svm.SVC(kernel="linear", C=2)  # ** Look here for more: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html, Higher C value = softer margin
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)    # Look here for more: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#:~:text=In%20multilabel%20classification%2C%20this%20function,set%20of%20labels%20in%20y_true.

print(acc)

