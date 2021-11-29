# Importing the libraries
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import sudokuSolver

#Setting the width and height of the image to make it a square
heightImg = 450
widthImg = 450

# Loading the CNN Model
model = load_model('MNIST_CNN.h5')

def preProcess(img):
    #Converting image to gray scale
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Adding Blur
    destination = cv2.GaussianBlur(imgGray, (9, 9), 1)
    #Applying Threshold
    source_px, imgThreshold = cv2.threshold(destination, 180, 255, cv2.THRESH_BINARY_INV)
    return imgThreshold

#Loading the Image
img = cv2.imread("SudokuImage.jpeg")
#Resizing the image to make it square
img = cv2.resize(img, (widthImg, heightImg))
# Creating a Blank Image for debugging
imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)
#Calling preProcess()
imgThreshold = preProcess(img)

#View Threshold Image
# plt.imshow(imgThreshold)
# plt.show()

imgContours = img.copy()
imgBigContour = img.copy()
# Finding all Contours
contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Drawing all detected contours
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3)

#A func to reorder th points as per warping technique demands
def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] =myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] =myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

#A func to get the biggest grid, i.e, the Sukoku
def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest,max_area

#Finding the biggest contour
biggest, maxArea = biggestContour(contours)
if biggest.size != 0:
    biggest = reorder(biggest)
    print(biggest)
    #Drawing the biggest contour
    cv2.drawContours(imgBigContour, biggest, -1, (0, 0, 255), 25)
    #Preparing points for warping
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]])
    #Performing Warp Perspective
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

    imgDetectedDigits = imgBlank.copy()
    #Converting Warp Image to gray scale
    imgWarpColored = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)

#A function to get 81 cell images
def splitBoxes(img):
    rows = np.vsplit(img,9)
    boxes=[]
    for r in rows:
        cols= np.hsplit(r,9)
        for box in cols:
            boxes.append(box)
    return boxes

#A function to predict all 81 images
def getPredection(boxes,model):
    result = []
    for image in boxes:
        ## PREPARE IMAGE
        img = np.asarray(image)
        img = img[7:img.shape[0] - 5, 5:img.shape[1] -5]
        source_px, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        img = cv2.resize(img, (28, 28))
        img = img.astype("float32")
        img = img / 255
        # plt.imshow(img)
        # plt.show()
        img = img.reshape(1, 28, 28, 1)

        classes = model.predict(img, batch_size=1)
        classes = classes.argmax()
        predictions = model.predict(img)
        probabilityValue = np.amax(predictions)
        if classes == [[0]]:
            if probabilityValue > 0.1:
                result.append((0))
            else:
                result.append(0)
        elif classes == [[1]]:
            if probabilityValue > 0.8:
                result.append(0)
            else:
                result.append(0)
        elif classes == [[2]]:
            if probabilityValue > 0.8:
                result.append(2)
            else:
                result.append(0)
        elif classes == [[3]]:
            if probabilityValue > 0.8:
                result.append(3)
            else:
                result.append(0)
        elif classes == [[4]]:
            if probabilityValue > 0.8:
                result.append(4)
            else:
                result.append(0)
        elif classes == [[5]]:
            if probabilityValue > 0.8:
                result.append(5)
            else:
                result.append(0)
        elif classes == [[6]]:
            if probabilityValue > 0.8:
                result.append(6)
            else:
                result.append(0)
        elif classes == [[7]]:
            if probabilityValue > 0.8:
                result.append(7)
            else:
                result.append(0)
        elif classes == [[8]]:
            if probabilityValue > 0.8:
                result.append(8)
            else:
                result.append(0)
        elif classes == [[9]]:
            if probabilityValue > 0.8:
                result.append(9)
            else:
                result.append(0)
    return result

#A function to display solution on th image
def displayNumbers(img,numbers,color = (0,255,0)):
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    for x in range (0,9):
        for y in range (0,9):
            if numbers[(y*9)+x] != 0 :
                 cv2.putText(img, str(numbers[(y*9)+x]),
                               (x*secW+int(secW/2)-10, int((y+0.8)*secH)), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            2, color, 2, cv2.LINE_AA)
    return img

#Finding each digit
imgSolvedDigits = imgBlank.copy()
boxes = splitBoxes(imgWarpColored)
print(len(boxes))
numbers = getPredection(boxes, model)
print(numbers)
imgDetectedDigits = displayNumbers(imgDetectedDigits, numbers, color=(255, 0, 255))
numbers = np.asarray(numbers)
#Keeping 1 in blank cells and 0 where numbers are present
posArray = np.where(numbers > 0, 0, 1)
print(posArray)

#Splitting the list into 9*9 and finding solution of the board
board = np.array_split(numbers,9)
print(board)
try:
    sudukoSolver.solve(board)
except:
    pass
print(board)

flatList = []
for sublist in board:
    for item in sublist:
        flatList.append(item)
solvedNumbers =flatList*posArray
imgSolvedDigits= displayNumbers(imgSolvedDigits,flatList)

#Drwaing Grids
def drawGrid(img):
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    for i in range (0,9):
        pt1 = (0,secH*i)
        pt2 = (img.shape[1],secH*i)
        pt3 = (secW * i, 0)
        pt4 = (secW*i,img.shape[0])
        cv2.line(img, pt1, pt2, (255, 255, 0),2)
        cv2.line(img, pt3, pt4, (255, 255, 0),2)
    return img

#Prepare points for warping
pts2 = np.float32(biggest)
pts1 =  np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
matrix = cv2.getPerspectiveTransform(pts1, pts2)  # GER
imgInvWarpColored = img.copy()
imgInvWarpColored = cv2.warpPerspective(imgSolvedDigits, matrix, (widthImg, heightImg))
inv_perspective = cv2.addWeighted(imgInvWarpColored, 1, img, 0.5, 1)
imgDetectedDigits = drawGrid(imgDetectedDigits)
imgSolvedDigits = drawGrid(imgSolvedDigits)

cv2.imshow('Solved Sudoku', inv_perspective)

cv2.waitKey(0)