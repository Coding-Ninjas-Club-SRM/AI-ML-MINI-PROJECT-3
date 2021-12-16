# Importing the necessary libraries
import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
from keras.preprocessing.image import img_to_array
from skimage.segmentation import clear_border
from keras.models import load_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint


#Fetching data
(X_train, y_train),(X_test, y_test) = mnist.load_data()

X_train = X_train.astype(np.float32)/255
X_test = X_test.astype(np.float32)/255

# Resizing the Dimension of image to (28,28,1)
X_train  = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)


y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

model = Sequential()

model.add(Conv2D(32,(3,3),input_shape = (28,28,1), activation='relu'))
model.add(MaxPool2D((2,2)))

model.add(Conv2D(64,(3,3), activation='relu'))
model.add(MaxPool2D((2,2)))

model.add(Conv2D(128,(3,3), activation='relu'))
model.add(MaxPool2D((2,2)))


model.add(Flatten())

model.add(Dropout(0.25))


model.add(Dense(100,activation="relu"))
model.add(Dense(10,activation="softmax"))

model.summary()

model.compile(optimizer='adam',
     loss = tf.keras.losses.categorical_crossentropy,
      metrics=['accuracy'])


es =EarlyStopping(monitor = 'val_acc',
         min_delta = 0.01, patience = 4, 
        verbose= 1)

mc = ModelCheckpoint("./model.h5" ,
                    monitor = "val_acc",
                    verbose = 1,
                    save_best_only= True)


callback = [es,mc]
history = model.fit(X_train, y_train, epochs =50, validation_split =0.3, callbacks= callback)

model_save = model.save("model.h5")


# Creating function to preprocess image
def find(img):
	# convert the image from colour to grayscale
	grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Blurring the grayscale image
	blur = cv2.GaussianBlur(grayscale, (11, 11), 0)
    # apply adaptive thresholding to invert the threshold map
	thresh = cv2.adaptiveThreshold(blur, 255,
		cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2)
	thresh = cv2.bitwise_not(thresh)

    # find contours in the thresholded image and sort them by size in descending order
	ext_contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	ext_contours = imutils.grab_contours(ext_contours)
	ext_contours = sorted(ext_contours, key=cv2.contourArea, reverse=True)
	
	puzzleCnt = None
	
	for c in ext_contours:
		# approximate the contour
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)
		
		if len(approx) == 4:
			puzzleCnt = approx
			break

    # if puzzle contour is empty then our script could not find the outline of the Sudoku puzzle, so raise an error
	if puzzleCnt is None:
		raise Exception(("Sudoku puzzle outline not found"
			"\nTry debugging your thresholding and contour steps."))
	
	# draw the contour of the puzzle
	grid = img.copy()
	cv2.drawContours(grid, [puzzleCnt], -1, (0, 255, 0), 2)
	cv2.imshow("Puzzle Outline", grid)
	cv2.waitKey(0)


    # apply a four point perspective transform to both the original image and grayscale image to obtain a top-down bird's eye view of the puzzle
	puzzle = four_point_transform(img, puzzleCnt.reshape(4, 2))
	warped = four_point_transform(grayscale, puzzleCnt.reshape(4, 2))

	
	cv2.imshow("Puzzle Transform", puzzle)
	cv2.waitKey(0)
	# return a 2-tuple of puzzle in both RGB and grayscale
	return (puzzle, warped)


def digit_extraction(cell):
	# apply automatic thresholding to the cell and clearing any connected borders that touch the border of the cell
	thresh = cv2.threshold(cell, 0, 255,
		cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	thresh = clear_border(thresh)
	
	cv2.imshow("Cell Thresh", thresh)
	cv2.waitKey(0)

    # finding contours
	ext_contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	ext_contours = imutils.grab_contours(ext_contours)
	if len(ext_contours) == 0:
		return None
	c = max(ext_contours, key=cv2.contourArea)
	mask = np.zeros(thresh.shape, dtype="uint8")
	cv2.drawContours(mask, [c], -1, 255, -1)


    # compute the percentage of masked pixels relative to the total area of the image
	(h, w) = thresh.shape
	percentFilled = cv2.countNonZero(mask) / float(w * h)
	if percentFilled < 0.03:
		return None
	# applying the mask to the thresholded cell
	digit = cv2.bitwise_and(thresh, thresh, mask=mask)
	# check to see if we should visualize the masking step
	cv2.imshow("Digit", digit)
	cv2.waitKey(0)
	# returns the digit to the function
	return digit


# loading the model from disk
print("Loading the model...")
model = load_model("model.h5")
# load the input image from disk and resize it

img = cv2.imread("sudoku1.jpeg")
img = imutils.resize(img, width=600)

# find the puzzle in the image
(puzzleImg, warped) = find(img)
# initialize 9x9 Sudoku board
board = np.zeros((9, 9), dtype="int")
stepX = warped.shape[1] // 9
stepY = warped.shape[0] // 9
cellLocs = []


# locating the grid locations
for y in range(0, 9):
	# initialize the current list of cell locations
	row = []
	for x in range(0, 9):
		# compute the starting and ending (x, y)-coordinates of the current cell
		startX = x * stepX
		startY = y * stepY
		endX = (x + 1) * stepX
		endY = (y + 1) * stepY
		row.append((startX, startY, endX, endY))


        # crop the cell from the warped transform image and then extract the digit from the cell
		cell = warped[startY:endY, startX:endX]
		digit = digit_extraction(cell)
		if digit is not None:
			ROI = cv2.resize(digit, (28, 28))
			ROI = ROI.astype("float") / 255.0
			ROI = img_to_array(ROI)
			ROI = np.expand_dims(ROI, axis=0)
			# classify the digit and update the Sudoku board with the prediction
			pred = model.predict(ROI).argmax(axis=1)[0]
			board[y, x] = pred
	# creating rows
	cellLocs.append(row)
 

print(board)

#Solving the given Sudoku puzzle
Z = 9
def print_puzzle(m):
    for i in range(Z):
        for j in range(Z):
            print(m[i][j],end = " ")
        print()

def solve_puzzle(cell, row, col, n):
    for i in range(9):
        if cell[row][i] == n:
            return False

    for i in range(9):
        if cell[i][col] == n:
            return False


    startR = row - row % 3
    startC = col - col % 3
    for x in range(3):
        for y in range(3):
            if cell[x + startR][y + startC] == n:
                return False
    return True

def Suduko(cell, row, col):

    if (row == Z - 1 and col == Z):
        return True
    if col == Z:
        row = row + 1
        col = 0
    if cell[row][col] > 0:
        return Suduko(cell, row, col + 1)
    for n in range(1, Z + 1, 1): 

        if solve_puzzle(cell, row, col, n):

            cell[row][col] = n
            if Suduko(cell, row, col + 1):
                return True
        cell[row][col] = 0
    return False


if (Suduko(board, 0, 0)):
    print_puzzle(board)
else:
    print("CANNOT BE SOLVED")