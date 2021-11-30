#!/usr/bin/env python
# coding: utf-8

# In[35]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import streamlit as st
from PIL import Image, ImageOps
import pandas as pd
# In[36]:


@st.cache(allow_output_mutation = True)
def load_model():
    model = tf.keras.models.load_model('model.h5')  # LOAD THE CNN MODEL
    return model
model = load_model()


# In[77]:


def order_corner_points(corners):
    corners = [(corner[0][0], corner[0][1]) for corner in corners]
    top_r, top_l, bottom_l, bottom_r = corners[0], corners[1], corners[2], corners[3]
    return top_l, top_r, bottom_r, bottom_l
def find_corners(img):
    ext_contours = cv2.findContours(img, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    ext_contours = ext_contours[0] if len(ext_contours) == 2 else ext_contours[1]
    ext_contours = sorted(ext_contours, key=cv2.contourArea, reverse=True)

    for c in ext_contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.015 * peri, True)
        if len(approx) == 4:
            return approx
def processing(img, skip_dilate=False):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    process = cv2.GaussianBlur(img.copy(), (9, 9), 0)
    process = cv2.adaptiveThreshold(process, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    process = cv2.bitwise_not(process, process)
    return process

def perspective_transform(image, corners):
    ordered_corners = order_corner_points(corners)
    top_l, top_r, bottom_r, bottom_l = ordered_corners

    
    width_A = np.sqrt(((bottom_r[0] - bottom_l[0]) ** 2) + ((bottom_r[1] - bottom_l[1]) ** 2))
    width_B = np.sqrt(((top_r[0] - top_l[0]) ** 2) + ((top_r[1] - top_l[1]) ** 2))
    width = max(int(width_A), int(width_B))

   
    height_A = np.sqrt(((top_r[0] - bottom_r[0]) ** 2) + ((top_r[1] - bottom_r[1]) ** 2))
    height_B = np.sqrt(((top_l[0] - bottom_l[0]) ** 2) + ((top_l[1] - bottom_l[1]) ** 2))
    height = max(int(height_A), int(height_B))
    dimensions = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1],
                           [0, height - 1]], dtype="float32")

    ordered_corners = np.array(ordered_corners, dtype="float32")

    grid = cv2.getPerspectiveTransform(ordered_corners, dimensions)

    return cv2.warpPerspective(image, grid, (width, height))
 
def isSafe(grid, row, col, num):
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

def solveSuduko(grid, row, col):
    if (row == N - 1 and col == N):
        return True
    if col == N:
        row += 1
        col = 0
    if grid[row][col] > 0:
        return solveSuduko(grid, row, col + 1)
    for num in range(1, N + 1, 1):
        if isSafe(grid, row, col, num):
            grid[row][col] = num
            if solveSuduko(grid, row, col + 1):
                return True
        grid[row][col] = 0
    return False


# In[71]:


uploaded_file = st.file_uploader("Upload your own image (supported types: jpg, jpeg, png)...", type=["jpg","jpeg","png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, width=500)
    
    input_image = np.array(img, dtype='uint8')
    print(input_image.shape)
    input_image = input_image[:,:,:3]
    try:
        img= processing(input_image)
        sudoku = find_corners(img)
        transformed = perspective_transform(img, sudoku)
        transformed = cv2.resize(transformed, (450,450))
        transformed = cv2.rotate(transformed, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        transformed = cv2.bitwise_not(transformed)
    except:
        st.error("There is something wrong with processing your image")
        st.stop()
        transformed = input_image
        transformed = cv2.resize(transformed, (450,450))
        transformed = cv2.bitwise_not(transformed)



    

    
   
# In[72]:



    rows = np.vsplit(transformed,9)
    boxes=[]
    for r in rows:
        cols= np.hsplit(r,9)
        for box in cols:
            boxes.append(box)


# In[73]:


    #plt.imshow(transformed, cmap="Greys_r")


# In[74]:


    result = []
    for i in boxes:
        
        number = i  
        number = number[4:number.shape[0] - 4, 4:number.shape[1] -4]
        number = cv2.resize(number, (32,32))
        #plt.imshow(number, cmap="Greys_r")
        n = number.astype('float32')
        n = n.reshape(1, 32, 32, 1)
    
        n = n/255.0
        pred = model.predict(n)
        probabilityValue = np.amax(pred)
        pred = np.argmax(pred, axis=-1)
        if probabilityValue > 0.8:
            result.append(pred[0])
        else:
            result.append(0)


# In[75]:


    result=np.array(result)
    result=np.split(result,9)
    print(result)


    # In[76]:


    N = 9
    grid = result
    with st.spinner("Working on the sudoku..."):
        if (solveSuduko(grid, 0, 0)):
            st.success("Result:")
            df = pd.DataFrame(grid)
            st.write(df)
        else:
            st.write("No Solution found")
            df = pd.DataFrame(grid)
            st.write(df)


