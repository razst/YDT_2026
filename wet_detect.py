import cv2
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

grid = divide_image("photos\WIN_20251127_19_27_09_Pro.jpg", 30, 30)

print(is_tile_wet(grid[0][2], 60))