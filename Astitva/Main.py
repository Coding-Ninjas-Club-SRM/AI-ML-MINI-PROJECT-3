###########################################################################################################################################
#                                                          Libraries Import
###########################################################################################################################################


import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
from keras.models import load_model
from matplotlib import pyplot as plt
from skimage.segmentation import clear_border
from keras.preprocessing.image import img_to_array

###########################################################################################################################################
#                                                        Image Pre-Processing
###########################################################################################################################################

img=cv2.imread(r"D:\GIT\CN\Project\bc.jpeg")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #Image Grayscaled
gblur=cv2.GaussianBlur(gray, (11,11), 0) #Gaussian Blur to Remove Noise
Adapt = cv2.adaptiveThreshold(gblur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7,2)  #Adaptive Threshold To Clear the Image
bit = cv2.bitwise_not(Adapt) #bitwise To invert The Colours of the image
cv2.imshow("b",bit) #image show
cv2.imshow("a",Adapt) #image Show
cv2.waitKey(0) #waiting till Windows Closed

###########################################################################################################################################
#                                                  Contours Finding, Crop and Warping
###########################################################################################################################################

cnts = cv2.findContours(bit.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #Countours Finding
cnts = imutils.grab_contours(cnts) #Noting Down the Contours
cnts = sorted(cnts, key=cv2.contourArea, reverse=True) #Sorting the Contours
puzzleCnt = None 
for c in cnts:    #Going Through Each Contour
        
    peri = cv2.arcLength(c, True) #Finding Length of the whole Curve
    approx = cv2.approxPolyDP(c, 0.02 * peri, True) #Using Approx PolyDP in order to Find Approx Shape
    print(approx) #printing Approx
    if len(approx) == 4: #Limitting 4 Points
        puzzleCnt = approx  #Just Assigning Approx Value Tp Puzzlecnt
        break #Exit Loop

    if puzzleCnt is None: #Conditional Statement
        raise Exception(("Could not find Sudoku puzzle outline. "  #Exception Given
            "Try debugging your thresholding and contour steps."))
output = img.copy() #img copied to output
cv2.drawContours(output, [puzzleCnt], -1, (225, 0,  0), 2) #Drawing Contours
cv2.imshow("Puzzle Outline", output)


puzzle = four_point_transform(img, puzzleCnt.reshape(4, 2)) #Cropping and Warping
warped = four_point_transform(bit, puzzleCnt.reshape(4, 2)) #Cropping and Warping
warped = imutils.resize(warped,width=600) #Image Resize
cv2.imshow("Puzzle1", puzzle) #Cropping and Warping
cv2.imshow("Puzzle", warped) #Cropping and Warping

edge_h = np.shape(warped)[0] #Edge Finding
edge_w = np.shape(warped)[1] #Edge Finding
celledge_h = edge_h // 9 #Dividing Rows
celledge_w = edge_w // 9 #Dividing Columns

grid=warped.copy() #Image Coping
cv2.imshow("GRID",grid) #Image Copying
cv2.waitKey(0) #Wait Till Execution

###########################################################################################################################################
#                                                    Cell Extraction and Prediction
###########################################################################################################################################


def extract_digit(cell):  #function Defined to Extract Cell
        thresh = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY) #GrayScaling The Image
        thresh = cv2.threshold(thresh, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] #Thresholding to Clear the Image With Inversion
        thresh = clear_border(thresh) #Clearing The Borders
        cv2.imshow("Threshold Image",thresh) #Image Show
        cv2.waitKey(0) #Waiting the Execution

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #Countours Finding
        cnts = imutils.grab_contours(cnts) #Countours Grabbing
        if len(cnts) == 0: #Conditional Statement to Check No Contours
            return None #Cross
        c = max(cnts, key=cv2.contourArea) #finding Max Contours Area
        mask = np.zeros(thresh.shape, dtype="uint8") #Numpy Convert
        cv2.drawContours(mask, [c], -1, 255, -1) #Drawing Contours

        (h, w) = puzzle.shape #Getting Height and Width
        percentFilled = cv2.countNonZero(mask) / float(w * h) #Finding Percentage Filled
        if percentFilled < 0.03: ##Conditional Statement to Check Empty Cell
            return None #Cross
        digit = cv2.bitwise_and(thresh, mask=mask) #Getting Digits Inverted
        cv2.imshow("Digit", digit) #Image Show
        cv2.waitKey(0) #Wait till Execution
        return digit #function Returning Digit as Output

model=load_model('D:\GIT\model.h5') #loading Model

board = np.zeros((9, 9), dtype="int") #Creating a Board
stepX = puzzle.shape[1] // 9 #Finding Shape
stepY = puzzle.shape[0] // 9 #Finding Shape
cellLocs = [] #Creating a List

for y in range(0, 9): #For Loops
	row = [] #Creating A Row List
	for x in range(0, 9): #For Loops
		startX = x * stepX #Starting Value of Cell Extraction
		startY = y * stepY #Starting Value of Cell Extraction
		endX = (x + 1) * stepX #Ending Value of Cell Extarction
		endY = (y + 1) * stepY #Ending Value of Cell Extraction
		row.append((startX, startY, endX, endY)) #Giving Values To Row

		cell = puzzle[startY:endY, startX:endX] #Cell Marking
		digit = extract_digit(cell) #Function Call
		if digit is not None: #Condition Statement
			roi = cv2.resize(digit, (28, 28)) #Resize Cell
			roi = roi.astype("float") / 255.0 #Float Image
			roi = img_to_array(roi) #Image Converted To Array
			roi = np.expand_dims(roi, axis=0) 
			pred = model.predict(roi).argmax(axis=1)[0] #Prediction Marked
			board[y, x] = pred #entering Values In Board
	cellLocs.append(row) #Appending Values In CellLocs

print(board) #Printing Board

###########################################################################################################################################
#                                                          Puzzle Solving
###########################################################################################################################################

M = 9 #initialising a Variable with Value 9
def puzzle(a): #Function To Print Solved
    for i in range(M): 
        for j in range(M):
            print(a[i][j],end = " ") #Printing Statement
        print() #Pass

def solve(grid, row, col, num): # Solving Fuction For Sudoku
    for x in range(9): #Checking Every No. in Range 1 to 9
        if grid[row][x] == num: #If No. Is Present In Row
            return False #pass
             
    for x in range(9): #Checking Every No. in Range 1 to 9
        if grid[x][col] == num: #If No. Is Present In Row
            return False #pass
 
 
    startRow = row - row % 3 #Marking 3 Rows at a time
    startCol = col - col % 3 #Marking 3 Columns at a time
    for i in range(3): #Checking Every Row in Range 1 to 9
        for j in range(3): #Checking Every Coloumn in Range 1 to 9
            if grid[i + startRow][j + startCol] == num: #If No. Is Present In Row and Column of 3 sets
                return False #pass
    return True #Return True
 
def Suduko(grid, row, col): #Calling Functions
 
    if (row == M - 1 and col == M):  #Checking The Values
        return True #return true
    if col == M: #In Column
        row += 1 #In Row
        col = 0 
    if grid[row][col] > 0: #Conditional Statement
        return Suduko(grid, row, col + 1) #Returning Value
    for num in range(1, M + 1, 1):  #If Num is Given In Range 
     
        if solve(grid, row, col, num): #Calling Function Solver
         
            grid[row][col] = num #Putting Value 
            if Suduko(grid, row, col + 1): #Condition To Check empty Cell
                return True #if Yes, return True
        grid[row][col] = 0 #Then Append 0
    return False #Returning False


if (Suduko(board, 0, 0)): #Calling Main Function
    puzzle(board) #Printing Values as Output
else:
    print("Solution does not exist:(") #Solution Does Not Exist in Given Puzzle

###########################################################################################################################################
#                                                          Program Ended
###########################################################################################################################################

