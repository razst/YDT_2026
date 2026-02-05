import cv2

#DEVIDE INTO 3 PARTS

import numpy as np
# from matplotlib import pyplot as plt

# img = cv2.imread("photos\WIN_20251127_19_27_09_Pro.jpg", cv2.IMREAD_GRAYSCALE)

# print(img)

# columns = np.array_split(img, 3, axis=1)
# for i, col in enumerate(columns):
#     cv2.imwrite(f"col_{i}.jpg", col)

# rows = np.array_split(img, 6, axis=0)
# for i, row in enumerate(rows):
#     cv2.imwrite(f"row_{i}.jpg", row)

# small = cv2.resize(img, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
# cv2.imshow("image",small)
# cv2.waitKey(0)

#tile = [[[1,2,3],[1,2,3],[1,2,3]],[[4,5,6],[4,5,6],[4,5,6]],[[7,8,9],[7,8,9],[7,8,9]]]

def divide_image(image_path, tile_height, tile_width):
    img = cv2.imread(image_path)
    if img is None:
        print("image not found")
        return []

    # Convert to HSV once
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_h, img_w = hsv_img.shape[:2] # Safely get height and width
    
    grid = [] # This will be our nested list
    for y in range(0, img_h, tile_height):
        row_list = [] # Start a new row
        for x in range(0, img_w, tile_width):
            tile = hsv_img[y:y+tile_height, x:x+tile_width]
            row_list.append(tile)
        grid.append(row_list) # Add the full row to the grid
    
    return grid

def is_tile_wet(tile, percent_threshold):
    # 'tile' is an HSV array. Channel 2 is 'Value' (Brightness).
    v_channel = tile[:, :, 2]
    
    # Create a mask where True (1) is a dark pixel and False (0) is bright
    dark_pixels = np.count_nonzero(v_channel <= 150)
    
    # Total number of pixels in this tile
    total_pixels = v_channel.size 
    
    actual_percent = (dark_pixels / total_pixels) * 100

    print(actual_percent)
    return actual_percent >= percent_threshold

def divide_into_three_rows(image_path):
    img = cv2.imread(image_path)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h, w = img.shape[:2]
    
    row_h = h // 3
    
    top_row = hsv_img[0 : row_h, 0 : w]
    mid_row = hsv_img[row_h : row_h*2, 0 : w]
    bottom_row = hsv_img[row_h*2 : h, 0 : w]
    
    rows = [top_row, mid_row, bottom_row]
    return rows

rows = divide_into_three_rows("photos\WIN_20251127_19_27_09_Pro_EDIT.jpg")
img = cv2.imread("photos\WIN_20251127_19_27_09_Pro_EDIT.jpg")


print(is_tile_wet(rows[1], 30))




lower_bound = np.array([0, 0, 0])
upper_bound = np.array([180, 255, 152])

mask1 = cv2.inRange(rows[0], lower_bound, upper_bound)
mask2 = cv2.inRange(rows[1], lower_bound, upper_bound)
mask3 = cv2.inRange(rows[2], lower_bound, upper_bound)

combined = np.vstack((mask1, mask2, mask3))

resized_comb = cv2.resize(combined, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
resized_img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
cv2.imshow('comb', resized_comb)
cv2.imshow('original', resized_img)

# print(is_tile_wet(rows[0], 60))

# img = cv2.imread("photos\WIN_20251127_19_27_09_Pro_EDIT.jpg")
# hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# lower_bound = np.array([0, 0, 0]),
# upper_bound = np.array([180, 255, 150])
# mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
# resized_img = cv2.resize(mask, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

# cv2.imshow('wet mask', resized_img)


cv2.waitKey(0)