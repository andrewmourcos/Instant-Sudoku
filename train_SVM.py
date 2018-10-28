# source ~/virtualenv/cv/bin/activate

import numpy as np
from sklearn.svm import LinearSVC
import os
import cv2
import joblib

# Generate training set
TRAIN_PATH = "Custom-Dataset/Train/"
list_folder = os.listdir(TRAIN_PATH)
# removing bothersome files
list_folder.remove('.DS_Store') 
trainset = []
for folder in list_folder:
    flist = os.listdir(os.path.join(TRAIN_PATH, folder))
    for f in flist:
        file = os.path.join(TRAIN_PATH, folder, f)
        if (f != ".DS_Store"):
            print(file)
            im = cv2.imread(file, 0)
            im = cv2.resize(im, (36,36))
            trainset.append(im)

# Labeling for trainset
train_label = []
for i in range(0,10):
    temp = 500*[i]
    train_label += temp

# Generate testing set
TEST_PATH = "Custom-Dataset/Test/"
list_folder = os.listdir(TEST_PATH)
# removing bothersome files
list_folder.remove('.DS_Store') 
testset = []
test_label = []
for folder in list_folder:
    flist = os.listdir(os.path.join(TEST_PATH, folder))
    for f in flist:
        file = os.path.join(TEST_PATH, folder, f)
        if (f != ".DS_Store"):
            im = cv2.imread(file, 0)
            im = cv2.resize(im, (36,36))
            testset.append(im)
            test_label.append(int(folder))
trainset = np.reshape(trainset, (5000, -1))

# Create a linear SVM object
clf = LinearSVC()

# Perform the training
clf.fit(trainset, train_label)
print("Training finished successfully")

# Testing
testset = np.reshape(testset, (len(testset), -1))
y = clf.predict(testset)
print("Testing accuracy: " + str(clf.score(testset, test_label)))

# creating pickle
joblib.dump(clf, "classifier.pkl", compress=3)