import cv2
import numpy as np

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
