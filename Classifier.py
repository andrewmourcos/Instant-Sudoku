import cv2
import numpy as np
import joblib

clf = joblib.load('classifier.pkl')
idx = 0
for i in range(0,50):
	idx += 1
	file = "Digits/"+ str(idx) + ".jpg"
	X = cv2.imread(file,0)
	X = cv2.resize(X, (36,36))
	num = clf.predict(np.reshape(X, (1,-1)))
	print(num[0], file)
