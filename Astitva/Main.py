import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
from keras.models import load_model
import matplotlib.pyplot as plt

#Img Pre Processing
img=cv2.imread(r"D:\GIT\CN\Project\bc.jpeg")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gblur=cv2.GaussianBlur(gray, (11,11), 0)
Adapt = cv2.adaptiveThreshold(gblur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7,2)
bit = cv2.bitwise_not(Adapt)
cv2.imshow("b",bit)
cv2.imshow("a",Adapt)
cv2.waitKey(0)
#Countours
cnts = cv2.findContours(bit.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
puzzleCnt = None
for c in cnts:
        
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    print(approx)
    if len(approx) == 4:
        puzzleCnt = approx
        break

    if puzzleCnt is None:
        raise Exception(("Could not find Sudoku puzzle outline. "
            "Try debugging your thresholding and contour steps."))
output = img.copy()
cv2.drawContours(output, [puzzleCnt], -1, (225, 0,  0), 2)
# cv2.imshow("Puzzle Outline", output)


puzzle = four_point_transform(img, puzzleCnt.reshape(4, 2))
warped = four_point_transform(bit, puzzleCnt.reshape(4, 2))
# cv2.imshow("Puzzle1", puzzle)
# cv2.imshow("Puzzle", warped)

edge_h = np.shape(warped)[0]
edge_w = np.shape(warped)[1]
celledge_h = edge_h // 9
celledge_w = edge_w // 9

grid=warped.copy()

tempgrid = []
for i in range(celledge_h, edge_h + 1, celledge_h):
    for j in range(celledge_w, edge_w + 1, celledge_w):
        rows = grid[i - celledge_h:i]
        tempgrid.append([rows[k][j - celledge_w:j] for k in range(len(rows))])

#Initialisation of Array
finalgrid = []
for i in range(0, len(tempgrid) - 8, 9):
    finalgrid.append(tempgrid[i:i + 9])


# Creating 9x9 ARRAY
for i in range(9):
    for j in range(9):
        finalgrid[i][j] = np.array(finalgrid[i][j])
try:
    for i in range(9):
        for j in range(9):
            os.remove("BoardCells/cell" + str(i) + str(j) + ".jpg")

except:
    pass

for i in range(9):
    for j in range(9):
        cv2.imwrite(str("BoardCells/cell" + str(i) + str(j) + ".jpg"), finalgrid[i][j])

cv2.imshow("5",finalgrid[4][5])
cv2.waitKey(0)
##########################################################################################
#                          CONFIRMED   CELL EXTRACTION DONE                              #
##########################################################################################

model=load_model('D:\GIT\model.h5')

COLUMNS = 9
ROWS = 9
WIDTH = 0 
HEIGHT = 0

#create 9x9 matrix
matrix = [[0 for x in range(COLUMNS)] for y in range(ROWS)]
def numericallysudoku(matrix):
    for x in range(COLUMNS):
        for y in range(ROWS):
            image = finalgrid[x][y].copy()
            if (np.count_nonzero(image)>50):
                image = cv2.resize(image, (28,28))
                #configurate image format for prediction
                image = image.astype('float32')
                image = image.reshape(1, 28, 28, 1)
                image /= 255
                #prediction
                pred = (model.predict(image) > 0.9).astype("int32")
                matrix[x][y] = pred.argmax()
            else:
                matrix[x][y] = 0
                
numericallysudoku(matrix) 
 

for i in range(len(matrix)):
    for j in range(len(matrix[i])):
        print (matrix[i][j],end="")
    print("\n")


# cv2.waitKey(0)

##########################################################################################
#                            CONFIRMED    MODEL AND PREDICTION                           #
##########################################################################################

M = 9
def puzzle(a):
    for i in range(M):
        for j in range(M):
            print(a[i][j],end = " ")
        print()

def solve(grid, row, col, num):
    for x in range(9):
        if grid[row][x] == num:
            return False
             
    for x in range(9):
        if grid[x][col] == num:
            return False
 
 
    startRow = row - row % 3
    startCol = col - col % 3
    for i in range(3):
        for j in range(3):
            if grid[i + startRow][j + startCol] == num:
                return False
    return True
 
def Suduko(grid, row, col):
 
    if (row == M - 1 and col == M):
        return True
    if col == M:
        row += 1
        col = 0
    if grid[row][col] > 0:
        return Suduko(grid, row, col + 1)
    for num in range(1, M + 1, 1): 
     
        if solve(grid, row, col, num):
         
            grid[row][col] = num
            if Suduko(grid, row, col + 1):
                return True
        grid[row][col] = 0
    return False


if (Suduko(matrix, 0, 0)):
    puzzle(matrix)
else:
    print("Solution does not exist:(")