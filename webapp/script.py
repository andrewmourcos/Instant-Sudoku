import cv2
import numpy as np
import joblib
from sklearn import datasets, neighbors, linear_model
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
import os

# ~~~ cleaning server ~~~
import shutil
import tempfile
import weakref

class FileRemover(object):
    def __init__(self):
        self.weak_references = dict()  # weak_ref -> filepath to remove

    def cleanup_once_done(self, response, filepath):
        wr = weakref.ref(response, self._do_cleanup)
        self.weak_references[wr] = filepath

    def _do_cleanup(self, wr):
        filepath = self.weak_references[wr]
        print('Deleting %s' % filepath)
        shutil.rmtree(filepath, ignore_errors=True)

file_remover = FileRemover()
# ~~~ cleaning server ~~~

def modify(array):
    doModify = int(input("Would you like to modify the scan? (0 or 1)"))
    if(doModify == 1):
        row = int(input("Which row? "))
        col = int(input("Which col? "))
        val = int(input("What should it be replaced with?"))
        array[row-1][col-1] = val
    return doModify

# ~~~ Sudoku solving functions by Hari ~~~
# https://stackoverflow.com/questions/1697334/algorithm-for-solving-sudoku
def findNextCellToFill(grid, i, j):
    for x in range(i,9):
        for y in range(j,9):
            if grid[x][y] == 0:
                return x,y
    for x in range(0,9):
        for y in range(0,9):
            if grid[x][y] == 0:
                return x,y
    return -1,-1

def isValid(grid, i, j, e):
    rowOk = all([e != grid[i][x] for x in range(9)])
    if rowOk:
        columnOk = all([e != grid[x][j] for x in range(9)])
        if columnOk:
            # finding the top left x,y co-ordinates of the section containing the i,j cell
            secTopX, secTopY = 3 *(i//3), 3 *(j//3) #floored quotient should be used here. 
            for x in range(secTopX, secTopX+3):
                for y in range(secTopY, secTopY+3):
                    if grid[x][y] == e:
                        return False
            return True
    return False

def solveSudoku(grid, i=0, j=0):
    i,j = findNextCellToFill(grid, i, j)
    if i == -1:
        return True
    for e in range(1,10):
        if isValid(grid,i,j,e):
            grid[i][j] = e
            if solveSudoku(grid, i, j):
                return True
            # Undo the current cell for backtracking
            grid[i][j] = 0
    return False
# ~~~ end of sudoku solver functions ~~~

# parameters: x and y coordinates of the 
# top left corner of contour
def locateEntry(xpx,ypx):
    for i in range(1,10):
        targY = (i*50)-25
        if(ypx < targY):
            for j in range(1,10):
                targX = (j*50)-25
                if(xpx < targX):
                    return i-1, j-1

def rectify(h):
    h = h.reshape((4,2))
    hnew = np.zeros((4,2),dtype = np.float32)
 
    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]
         
    diff = np.diff(h,axis = 1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]
  
    return hnew


def solve(fileName):
    # import image
    sudoku = cv2.imread("static/%s" %fileName, 0)
    sudoku1 = cv2.imread("static/%s" %fileName)

    sudoku = cv2.resize(sudoku, (420,420))
    sudoku1 = cv2.resize(sudoku1, (420,420))

    #import classifier
    clf = joblib.load('Classifiers/log-classifier.pkl') 

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

    # Iterate through digits in sudoku puzzle
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

    new_file_name = "solved_%s" %fileName
    # display sudoku puzzle with recognition
    cv2.imwrite("static/%s" %new_file_name, warped1)
    cv2.waitKey(0)
    print(new_file_name)
    return (new_file_name)

    # # Allow user to correct recognition
    # while(modify(numbers)):
    #     print(numbers)

    # # Solve sudoku puzzle
    # if(not solveSudoku(numbers)):
    #     print("Sorry, this one's a toughie... please try again")
    # else:
    #     print("Aha! Here's the solved puzzle:")
    #     print(numbers)

    # cv2.destroyAllWindows()
