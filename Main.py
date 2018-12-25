# Using 3443 data points I collected
# + 4362 data points from Cap Huuquan
# + 
import cv2
import numpy as np
import joblib
from sklearn import datasets, neighbors, linear_model
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
import os
from helpers import *

# import image
sudoku = cv2.imread("ye.jpg", 0)
sudoku1 = cv2.imread("ye.jpg")

sudoku = cv2.resize(sudoku, (420,420))
sudoku1 = cv2.resize(sudoku1, (420,420))

#import classifier
clf = joblib.load('classifier.pkl') 

font = cv2.FONT_HERSHEY_SIMPLEX

# filtering 
cop = np.zeros(sudoku.shape, np.uint8)
blur = cv2.GaussianBlur(sudoku,(5,5),0)

outerBox = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)
outerBox = cv2.bitwise_not(outerBox)

# detect roi
outerBox, contours, hierarchy = cv2.findContours(outerBox, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
biggest = None
max_area = 0
for i in contours:
        area = cv2.contourArea(i)
        if area > 100:
                peri = cv2.arcLength(i,True)
                approx = cv2.approxPolyDP(i,0.02*peri,True)
                if area > max_area and len(approx)==4:
                        biggest = approx
                        max_area = area

biggest = rectify(biggest)
h = np.array([ [0,0],[449,0],[449,449],[0,449] ],np.float32)
retval = cv2.getPerspectiveTransform(biggest,h)
warped = cv2.warpPerspective(blur,retval,(450,450))
warped1 = cv2.warpPerspective(sudoku1,retval,(450,450))


# get grid
cop = np.zeros(warped.shape, np.uint8)
outerBox = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)
outerBox = cv2.bitwise_not(outerBox)
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
np.array([[0, 0, 1, 0, 0],
       [0, 0, 1, 0, 0],
       [1, 1, 1, 1, 1],
       [0, 0, 1, 0, 0],
       [0, 0, 1, 0, 0]], dtype=np.uint8)
outerBox = cv2.dilate(outerBox, kernel, iterations=1)
grid = np.zeros(warped.shape, np.uint8)
lines = cv2.HoughLines(outerBox,1,np.pi/180,300)
for line in lines:
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(grid,(x1,y1),(x2,y2),(255),2)

grid = cv2.dilate(grid, kernel, iterations=2)
# squareBit = cv2.bitwise_not(grid)
numbers = np.zeros((9,9))
# collect all squares
_, contours, _ = cv2.findContours(grid,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

# Iterate through all digits in sudoku puzzle
idx =0 
for cnt in contours[:81]:
    idx += 1
    x,y,w,h = cv2.boundingRect(cnt)
    digit=outerBox[y:y+h,x:x+w]
    digit = cv2.erode(digit, kernel, iterations=2)
    cv2.rectangle(warped1,(x,y),(x+w,y+h),(200,0,0),2)
    digit = cv2.resize(digit, (36,36))

    # Use classifier for digit
    num = clf.predict(np.reshape(digit, (1,-1)))
    if(cv2.countNonZero(digit) < 50):
        num = [0]

    # Place digit in 2d array
    row, col = locateEntry(x,y)
    numbers[row][col] = num[0]
    cv2.putText(warped1,str(num[0]),(x,y+h),font,1,(225,0,0),2)

# if(not solveSudoku(numbers)):
#     print("Sorry, this one's a toughie... please try again")

# display image
cv2.imshow("character recognition", warped1)
cv2.waitKey(0)
cv2.destroyAllWindows()
