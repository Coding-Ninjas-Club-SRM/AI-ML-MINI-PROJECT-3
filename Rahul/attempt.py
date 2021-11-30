
import cv2
from matplotlib.pyplot import imread
import numpy as np
from numpy.lib.function_base import extract
import tensorflow as tf
import solver

# for taking image from webcam
cam= cv2.VideoCapture(0) 

def capture_image():
    while(True):
        ret,frame= cam.read() 

        cv2.imshow('frame',frame) #webcam shown in a window

        if cv2.waitKey(1) & 0xFF == ord('p'):  # p bound to capture the image
            image=cv2.imwrite('puzzle.png',frame)
            break

        elif cv2.waitKey(1) & 0xFF == ord('q'): # q bound to quitting the window
            break
    cam.release()
    cv2.destroyWindow('frame') #destroys the webcam window
    return image

def read_img(image):
    # processing image grayscale----->blur----->thresholding
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 3)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 3)

    #finding contours of the image

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    for c in cnts:
        peri = cv2.arcLength(c, True) 
        approx = cv2.approxPolyDP(c, 0.015 * peri, True)
        break
    
    
    def crop(corners):

        # The points obtained from contours may not be in order because of the skewness  of the image, or
        # because of the camera angle. This function returns a list of corners in the right order 
        sort_corners = [(corner[0][0], corner[0][1]) for corner in corners]
        sort_corners = [list(ele) for ele in sort_corners]
        x, y = [], []

        for i in range(len(sort_corners[:])):
            x.append(sort_corners[i][0])
            y.append(sort_corners[i][1])

       
        centroid = [sum(x) / len(x), sum(y) / len(y)]

        

        for _, item in enumerate(sort_corners):
            if item[0] < centroid[0]:
                if item[1] < centroid[1]:
                    top_left = item
                else:
                    bottom_left = item
            elif item[0] > centroid[0]:
                if item[1] < centroid[1]:
                    top_right = item
                else:
                    bottom_right = item

        ordered_corners = [top_left, top_right, bottom_right, bottom_left]

        width_A = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))
        width_B = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))
        width = max(int(width_A), int(width_B))


        # Construct new points to obtain top-down view of image in 
        # top_r, top_l, bottom_l, bottom_r order
        dimensions = np.array([[0, 0], [width, 0], [width , width], [0,width]], dtype = "float32")

        # Convert to Numpy format
        ordered_corners = np.array(ordered_corners, dtype="float32")

        # Find perspective transform matrix
        matrix = cv2.getPerspectiveTransform(ordered_corners, dimensions)

        # Return the transformed image
        sudoku=cv2.warpPerspective(image, matrix, (width, width))

        cv2.imwrite('final.png',sudoku)

    crop(approx)

    # func to extract cordinates of all 81 cells

    def extract():
        final=cv2.imread('final.png',)
        final=cv2.cvtColor(final,cv2.COLOR_RGB2GRAY)
        final=cv2.resize(final,(252,252))
        
        thresh1= cv2.adaptiveThreshold(final, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 6)
        thresh1= cv2.bitwise_not(thresh1)
        cv2.imwrite('threshed.png',thresh1)
        edgeh=np.shape(thresh1)[0]
        edgew=np.shape(thresh1)[1]
        cellh=edgeh//9
        cellw=edgew//9
        pos=[]

        x1,x2,y1,y2 = 0,0,0,0
        


        for i in range(9):
            y2 = y1 + cellh
            x1 = 0
            for j in range(9):
                x2 = x1 + cellw
                current_cell = [x1,x2,y1,y2]
                pos.append(current_cell)
                x1 = x2
            y1 = y2
        
        pos=np.array(pos,np.float32)
        
        return pos
    extract()

    def center_crop(img, dim):
        # Returns center cropped image
        # Args:
        # img: image to be center cropped
        # dim: dimensions (width, height) to be cropped
        
        width, height = img.shape[1], img.shape[0]

        # process crop width and height for max available dimension
        crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
        crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
        mid_x, mid_y = int(width/2), int(height/2)
        cw2, ch2 = int(crop_width/2), int(crop_height/2) 
        crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
        return crop_img
    
    #extracts each cell and runs it through the digit recognition model

    def trainer():
        model = tf.keras.models.load_model('ident_model.h5')
        num=[]
        pos=extract()
        final=cv2.imread('threshed.png')
        for i in range(len(pos)):
            body= final[int(pos[i][2]):int(pos[i][3]),int(pos[i][0]):int(pos[i][1])] # cropping each cell
            
            body=body[:,:,0] # grayscaling the img
            
            
            a=np.sum(body==255) #finding out how many white pixels are there in the image to filter out blank cells
            if a>150:
                body=center_crop(body,(25,26))
                body=cv2.medianBlur(body,3)
                body=cv2.resize(body,(28,28))
                body=body.reshape(1,28,28,1)
                body=np.float64(body)
                body /=255
                prediction=model.predict(body)
                num.append(np.argmax(prediction))
            else:
                num.append(0)
                
            
        return num
    
    
    x=trainer()
    print(x)
    x=np.asarray(x)
    board=np.array_split(x,9)
    
    solver.solver(board)

x=int(input('enter method of providing image\n\t 1. webcam \n\t2. path of file'))
if x==1:
    img1=capture_image()
    read_img(img1)
elif x==2:
    path=input('enter path of the image: ')
    img=cv2.imread(path)
    read_img(img)
else:
    print('please enter the correct number code!!!!')






