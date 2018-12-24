# this script takes pictures of sudoku games and
# extracts all the squares, saving those in /DigitExample

import cv2
import numpy as np

# functions
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
        
idx =0 

for x in range(1,41):
    # import image
    file_name = 'SudokuExample/sudo'+ str(x) +'.jpg'
    sudoku = cv2.imread(file_name, 0)
    sudoku1 = cv2.imread(file_name)

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
    lines = cv2.HoughLines(outerBox,1,np.pi/180,400)
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
    cv2.line(grid,(0,0),(0,450),(255),2)
    cv2.line(grid,(450,0),(450,450),(255),2)
    grid = cv2.dilate(grid, kernel, iterations=2)

    # collect all squares
    _, contours, _ = cv2.findContours(grid,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        idx += 1
        x,y,w,h = cv2.boundingRect(cnt)
        if (w > 350):
            pass
        else:
            digit=outerBox[y:y+h,x:x+w]
            digit = cv2.erode(digit, kernel, iterations=2)
            cv2.imwrite('DigitExample/'+ str(idx) + '.jpg', digit)
            cv2.rectangle(warped1,(x,y),(x+w,y+h),(200,0,0),2)
    print(file_name, idx)
        
