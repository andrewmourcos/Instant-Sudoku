from sklearn import datasets, neighbors, linear_model
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import cv2
import os
import numpy as np

# Acquire dataset to train
training_path = "CustomDataset/Train/"
list_folders = os.listdir(training_path)
trainset = []
trainlabels = []
for folder in list_folders:
	list_files = os.listdir(os.path.join(training_path, folder))
	for file in list_files:
		if(file != ".DS_Store"):
			print("\r" + os.path.join(training_path,folder,file), end="")
			img = cv2.imread(os.path.join(training_path, folder, file))
			img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
			img = cv2.resize(img, (36,36))
			trainset.append(img)
			trainlabels.append(int(folder))

# prepare testing dataset
testing_path = "CustomDataset/Test/"
list_test_folders = os.listdir(testing_path)
testset=[]
testlabels=[]
for folder in list_test_folders:
	if(folder != ".DS_Store"):
		list_files = os.listdir(os.path.join(testing_path, folder))
	for file in list_files:
		if(folder != ".DS_Store") and (file != ".DS_Store"):
			print("\r" + os.path.join(testing_path,folder, file), end="")
			img = cv2.imread(os.path.join(testing_path,folder,file))
			img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
			img = cv2.resize(img, (36,36))
			testset.append(img)
			testlabels.append(int(folder))

print("\n" + str(len(trainset)) + " elements to train on")
trainset = np.reshape(trainset, (len(trainset), -1))
testset = np.reshape(testset, (len(testset), -1))

svm = LinearSVC()
knn = neighbors.KNeighborsClassifier()
logistic = linear_model.LogisticRegression()

svm.fit(trainset, trainlabels)
knn.fit(trainset, trainlabels)
logistic.fit(trainset, trainlabels)
print("Training complete...")


X = svm.predict(testset)
print("SVM score: " + str(accuracy_score(testlabels, svm.predict(testset))))
print("KNN score: " + str(accuracy_score(testlabels, knn.predict(testset))))
print("Log Reg score: " + str(accuracy_score(testlabels, logistic.predict(testset))))

joblib.dump(knn, "classifier.pkl", compress=3)









