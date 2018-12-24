from sklearn import datasets, neighbors, linear_model
from sklearn.externals import joblib
from sklearn.svm import LinearSVC

digits = datasets.load_digits()
X_digits = digits.data / digits.data.max()
y_digits = digits.target

n_samples = len(X_digits)

X_train = X_digits[:int(.9 * n_samples)]
y_train = y_digits[:int(.9 * n_samples)]
X_test = X_digits[int(.9 * n_samples):]
y_test = y_digits[int(.9 * n_samples):]

knn = neighbors.KNeighborsClassifier()
logistic = linear_model.LogisticRegression(solver='lbfgs', max_iter=3000,
                                           multi_class='multinomial')
svm = LinearSVC()

print('KNN score: %f' % knn.fit(X_train, y_train).score(X_test, y_test))
print('LogisticRegression score: %f'
      % logistic.fit(X_train, y_train).score(X_test, y_test))
print('Svm score: %f' % svm.fit(X_train, y_train).score(X_test, y_test))

joblib.dump(knn, "classifier.pkl", compress=3)