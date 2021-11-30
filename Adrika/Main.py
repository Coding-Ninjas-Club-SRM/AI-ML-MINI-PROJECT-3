#import the required libraries:
import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
from keras.preprocessing.image import img_to_array
from skimage.segmentation import clear_border
from keras.models import load_model


def find(img):
	# convert the image to grayscale and blur it slightly
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    # apply adaptive thresholding and then invert the threshold map
	thresh = cv2.adaptiveThreshold(blurred, 255,
		cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2)
	thresh = cv2.bitwise_not(thresh)

    # find contours in the thresholded image and sort them by size in descending order
	cnt = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnt = imutils.grab_contours(cnt)
	cnt = sorted(cnt, key=cv2.contourArea, reverse=True)
	# initialize a contour that corresponds to the puzzle outline
	puzzleCnt = None
	# loop over the contours
	for c in cnt:
		# approximate the contour
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)
		# if our approximated contour has four points, then assume that we have found the outline of the puzzle
		if len(approx) == 4:
			puzzleCnt = approx
			break

    # if puzzle contour is empty then our script could not find the outline of the Sudoku puzzle, so raise an error
	if puzzleCnt is None:
		raise Exception(("Sudoku puzzle outline not found"
			"\nTry debugging your thresholding and contour steps."))
	
	# draw the contour of the puzzle on the image and then display it to our screen for visualization purposes
	output = img.copy()
	cv2.drawContours(output, [puzzleCnt], -1, (0, 255, 0), 2)
	cv2.imshow("Puzzle Outline", output)
	cv2.waitKey(0)


    # apply a four point perspective transform to both the original image and grayscale image to obtain a top-down bird's eye view of the puzzle
	puzzle = four_point_transform(img, puzzleCnt.reshape(4, 2))
	warped = four_point_transform(gray, puzzleCnt.reshape(4, 2))

	# check to see if we are visualizing the perspective transform and show the output warped image (for debugging purposes)
	cv2.imshow("Puzzle Transform", puzzle)
	cv2.waitKey(0)
	# return a 2-tuple of puzzle in both RGB and grayscale
	return (puzzle, warped)


def digit_extraction(cell):
	# apply automatic thresholding to the cell and then clear any connected borders that touch the border of the cell
	thresh = cv2.threshold(cell, 0, 255,
		cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	thresh = clear_border(thresh)
	# check to see if we are visualizing the cell thresholding step
	cv2.imshow("Cell Thresh", thresh)
	cv2.waitKey(0)

    # find contours in the thresholded cell
	cnt = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnt = imutils.grab_contours(cnt)
	# if no contours were found than this is an empty cell
	if len(cnt) == 0:
		return None
	# otherwise, find the largest contour in the cell and create a mask for the contour
	c = max(cnt, key=cv2.contourArea)
	mask = np.zeros(thresh.shape, dtype="uint8")
	cv2.drawContours(mask, [c], -1, 255, -1)


    # compute the percentage of masked pixels relative to the total area of the image
	(h, w) = thresh.shape
	percentFilled = cv2.countNonZero(mask) / float(w * h)
	# if less than 3% of the mask is filled then we are looking at noise and so, can safely ignore the contour
	if percentFilled < 0.03:
		return None
	# apply the mask to the thresholded cell
	digit = cv2.bitwise_and(thresh, thresh, mask=mask)
	# check to see if we should visualize the masking step
	cv2.imshow("Digit", digit)
	cv2.waitKey(0)
	# return the digit to the calling function
	return digit


# load the model from disk
print("Loading model...")
model = load_model("model.h5")
# load the input image from disk and resize it
print("Processing image...")
img = cv2.imread("abc.jpeg")
img = imutils.resize(img, width=600)

# find the puzzle in the image
(puzzleImg, warped) = find(img)
# initialize 9x9 Sudoku board
board = np.zeros((9, 9), dtype="int")
# infer the location of each cell by dividing the warped image into a 9x9 grid
stepX = warped.shape[1] // 9
stepY = warped.shape[0] // 9
# initialize a list to store the (x, y)-coordinates of each cell location
cellLocs = []


# loop over the grid locations
for y in range(0, 9):
	# initialize the current list of cell locations
	row = []
	for x in range(0, 9):
		# compute the starting and ending (x, y)-coordinates of the current cell
		startX = x * stepX
		startY = y * stepY
		endX = (x + 1) * stepX
		endY = (y + 1) * stepY
		# add the (x, y)-coordinates to our cell locations list
		row.append((startX, startY, endX, endY))


        # crop the cell from the warped transform image and then extract the digit from the cell
		cell = warped[startY:endY, startX:endX]
		digit = digit_extraction(cell)
		# verify that the digit is not empty
		if digit is not None:
			# resize the cell to 28x28 pixels and then prepare the cell for classification
			roi = cv2.resize(digit, (28, 28))
			roi = roi.astype("float") / 255.0
			roi = img_to_array(roi)
			roi = np.expand_dims(roi, axis=0)
			# classify the digit and update the Sudoku board with the prediction
			pred = model.predict(roi).argmax(axis=1)[0]
			board[y, x] = pred
	# add the row to cell locations
	cellLocs.append(row)
 

print(board)

#Solve the given Sudoku puzzle
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
    print("NO SOLUTION!")


#############################END#################################