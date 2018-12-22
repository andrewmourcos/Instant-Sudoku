from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

digits = datasets.load_digits()
classifier = svm.SVC(gamma = 0.001)

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.33, random_state=5)

logisticRegr = LogisticRegression()

classifier = logisticRegr.fit(x_train, y_train)

score = classifier.score(x_test, y_test)
print("The accuracy is: " + str(score))

