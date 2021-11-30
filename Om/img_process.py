import cv2
import numpy as np


def display_image(img):
    cv2.imshow('image', img) 
    cv2.waitKey(0)  
    cv2.destroyAllWindows()  



def processing(img, skip_dilate=False):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    process = cv2.GaussianBlur(img.copy(), (9, 9), 0)

    
    process = cv2.adaptiveThreshold(process, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

   
    process = cv2.bitwise_not(process, process)

    if not skip_dilate:
        
        kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
        process = cv2.dilate(process, kernel)

    return process


def find_corners(img):
    
    ext_contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ext_contours = ext_contours[0] if len(ext_contours) == 2 else ext_contours[1]
    ext_contours = sorted(ext_contours, key=cv2.contourArea, reverse=True)

    
    for c in ext_contours:
        peri = cv2.arcLength(c, True)
        
        approx = cv2.approxPolyDP(c, 0.015 * peri, True)
        if len(approx) == 4:
            
            return approx




def order_corner_points(corners):
    corners = [(corner[0][0], corner[0][1]) for corner in corners]
    top_r, top_l, bottom_l, bottom_r = corners[0], corners[1], corners[2], corners[3]
    return top_l, top_r, bottom_r, bottom_l



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


def create_image_grid(img):
    grid = np.copy(img)
    
    edge_h = np.shape(grid)[0]
    edge_w = np.shape(grid)[1]
    celledge_h = edge_h // 9
    celledge_w = np.shape(grid)[1] // 9

    grid = cv2.cvtColor(grid, cv2.COLOR_BGR2GRAY)

   
    grid = cv2.bitwise_not(grid, grid)


    tempgrid = []
    for i in range(celledge_h, edge_h + 1, celledge_h):
        for j in range(celledge_w, edge_w + 1, celledge_w):
            rows = grid[i - celledge_h:i]
            tempgrid.append([rows[k][j - celledge_w:j] for k in range(len(rows))])

    
    finalgrid = []
    for i in range(0, len(tempgrid) - 8, 9):
        finalgrid.append(tempgrid[i:i + 9])

    
    for i in range(9):
        for j in range(9):
            finalgrid[i][j] = np.array(finalgrid[i][j])

    try:
        for i in range(9):
            for j in range(9):
                np.os.remove("BoardCells1/cell" + str(i) + str(j) + ".jpg")
    except:
        pass
    for i in range(9):
        for j in range(9):
            cv2.imwrite(str("BoardCells1/cell" + str(i) + str(j) + ".jpg"), finalgrid[i][j])

    return finalgrid


def scale_and_centre(img, size, margin=20, background=0):
    """Scales and centres an image onto a new background square."""
    h, w = img.shape[:2]

    def centre_pad(length):
        """Handles centering for a given length that may be odd or even."""
        if length % 2 == 0:
            side1 = int((size - length) / 2)
            side2 = side1
        else:
            side1 = int((size - length) / 2)
            side2 = side1 + 1
        return side1, side2

    def scale(r, x):
        return int(r * x)

    if h > w:
        t_pad = int(margin / 2)
        b_pad = t_pad
        ratio = (size - margin) / h
        w, h = scale(ratio, w), scale(ratio, h)
        l_pad, r_pad = centre_pad(w)
    else:
        l_pad = int(margin / 2)
        r_pad = l_pad
        ratio = (size - margin) / w
        w, h = scale(ratio, w), scale(ratio, h)
        t_pad, b_pad = centre_pad(h)

    img = cv2.resize(img, (w, h))
    img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, None, background)
    return cv2.resize(img, (size, size))


def extract():
    # print("Enter image name: ")
    # image_url = input()
    img = cv2.imread('sudoku_1.jpg')
    processed_sudoku = processing(img)
    sudoku = find_corners(processed_sudoku)
    transformed = perspective_transform(img, sudoku)
    cropped = 'cropped_img.png'
    cv2.imwrite(cropped, transformed)
    transformed = cv2.resize(transformed, (450, 450))
    sudoku = create_image_grid(transformed)
    return sudoku

