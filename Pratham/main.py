from imutils.perspective import four_point_transform
from keras.preprocessing.image import img_to_array
from skimage.segmentation import clear_border
from tkinter import filedialog
from keras.models import load_model
import tkinter as tk 
import numpy as np
import imutils
import cv2
import os



MODEL = load_model('model\model.h5')

def selectImg():
    root = tk.Tk() 
    root.withdraw() 
    imgSrc = filedialog.askopenfilename(initialdir= os.getcwd(),title="Select Base Image: ")
    return imgSrc
    

def preprocessing(img):
    height = 540
    width = 540
    img = cv2.resize(img, (width, height))
    grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    bluredImg = cv2.GaussianBlur(grayImg, (9,9), 0)
    thresholdImg = cv2.adaptiveThreshold(bluredImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7,2)
    thresholdImg = cv2.bitwise_not(thresholdImg)
    
    return thresholdImg


def reorder(contour):
        contour = contour.reshape((4, 2))
        contourReordered = np.zeros((4, 1, 2), dtype=np.int32)
        add = contour.sum(1)
        contourReordered[0] = contour[np.argmin(add)]
        contourReordered[3] =contour[np.argmax(add)]
        diff = np.diff(contour, axis=1)
        contourReordered[1] =contour[np.argmin(diff)]
        contourReordered[2] = contour[np.argmax(diff)]
        
        return contourReordered

def findContours(processedImage):
    
    

    finalCountour = np.array([])
    contours , _ = cv2.findContours(processedImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 200 :
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if area > 0 and len(approx) == 4:
                finalCountour = approx
    
    if finalCountour.size != 0:
        finalCountour = reorder(finalCountour)
    
    
    return finalCountour

def extractDigit(cell):
    thresh = cv2.threshold(cell, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = clear_border(thresh)
    contour = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contour = imutils.grab_contours(contour)
    
    if len(contour) == 0:
        return None

    c = max(contour, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)
    (h, w) = thresh.shape
    percentFilled = cv2.countNonZero(mask) / float(w * h)
    
    if percentFilled < 0.03:
        return None

    digit = cv2.bitwise_and(thresh, thresh, mask=mask)
    return digit



def cells(croppedImage):
    board = np.zeros((9, 9), dtype="int")
    stepX = croppedImage.shape[1] // 9
    stepY = croppedImage.shape[0] // 9

    for x in range(0, 9):
        row = []
        for y in range(0, 9):
            
            startX = x * stepX
            endX = (x + 1) * stepX

            startY = y * stepY
            endY = (y + 1) * stepY
            
            cell = croppedImage[startY:endY, startX:endX]

            digit = extractDigit(cell)
    
            if digit is not None:
    
                cellImage = cv2.resize(digit, (28, 28))
                cellImage = cellImage.astype("float") / 255.0
                cellImage = img_to_array(cellImage)
                cellImage = np.expand_dims(cellImage, axis=0)
                prediction = MODEL.predict(cellImage).argmax(axis=1)
                board[ x , y ] = prediction
    return board

def find_empty(b):
    for i in range(len(b)):
        for j in range(len(b[0])):
            if b[i][j]==0:
                return (i,j)
    return None

def valid(b, num, pos):
    for i in range(len(b[0])):
        if b[pos[0]][i] == num and pos[1]!= i:
            return False

    for i in range(len(b)):
        if b[i][pos[1]] == num and pos[0]!= i:
            return False

    box_x = pos[1] // 3
    box_y = pos[0] // 3

    for i in range(box_y*3, box_y*3 + 3):
        for j in range(box_x * 3, box_x*3 + 3):
            if b[i][j] == num and (i,j) != pos:
                return False

    return True

def solve(board):
    
    f = find_empty(board)
    if not f:
        return True
    else:
        row,col = f
    for i in range(1,10):
        if valid(board, i, (row,col)):
            board[row][col]=i
            if solve(board):
                return True
            board[row][col] = 0
    return False


def main():
    img = cv2.imread(selectImg())
    processedImage = preprocessing(img)
    contour = findContours(processedImage)
    croppedImage = four_point_transform(processedImage, contour.reshape(4, 2))
    board = cells(croppedImage) 
    print(solve(board))
   
   
   
   


if __name__ == "__main__":
    main()

