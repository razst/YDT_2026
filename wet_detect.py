import cv2
import time
import numpy as np

wet_precent = 60 #TODO CAP LETTER
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

#TODO comments 
def is_tile_wet(tile, percent_threshold):
    # 'tile' is an HSV array. Channel 2 is 'Value' (Brightness).
    v_channel = tile[:, :, 2]
    
    # Create a mask where True (1) is a dark pixel and False (0) is bright
    dark_pixels = np.count_nonzero(v_channel <= 150)
    
    # Total number of pixels in this tile
    total_pixels = v_channel.size 
    
    actual_percent = (dark_pixels / total_pixels) * 100

    print(actual_percent)
    return actual_percent >= percent_threshold, actual_percent

def divide_into_three_rows(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h, w = img.shape[:2]
    
    row_h = h // 3
    
    top_row = hsv_img[0 : row_h, 0 : w]
    mid_row = hsv_img[row_h : row_h*2, 0 : w]
    bottom_row = hsv_img[row_h*2 : h, 0 : w]
    
    rows = [top_row, mid_row, bottom_row]
    return rows




def main_image(path):
    frame = cv2.imread(path)
    rows = divide_into_three_rows(frame)
    print(rows)
    frame_h = frame.shape[0]
    row_h = frame_h // 3
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, row in enumerate(rows):
        is_wet = is_tile_wet(row, wet_precent)
        y_position = int((row_h * i) + (row_h // 2))
        # print("Zone "+str(i)+", is_wet: "+str(is_wet)) 
        cv2.putText(frame, "Zone "+str(i)+", is_wet: "+str(is_wet), (20, y_position), font, 1, (255, 0, 0), 2)
        # cv2.putText(frame,"test",(10,10),font,10,4,(255,0,0))
    cv2.imshow("wet", frame)
    cv2.waitKey(0)

def main_movie(path):
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return 

    prev_time = 0
    fps_count=0
    fps_sum=0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read frame.")
            break
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        current_time = time.time()

        # FPS = 1 / time between frames
        fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
        prev_time = current_time
        fps_count +=1
        fps_sum+=fps

        # Resize once (0.5 is usually enough)
        frame = cv2.resize(frame, None, fx=0.2, fy=0.5, interpolation=cv2.INTER_AREA)
        rows = divide_into_three_rows(frame)
        frame_h = frame.shape[0]
        row_h = frame_h // 3
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, row in enumerate(rows):
            is_wet = is_tile_wet(row, wet_precent)
            y_position = int((row_h * i) + (row_h // 2))
            # print("Zone "+str(i)+", is_wet: "+str(is_wet)) 
            cv2.putText(frame, "Zone "+str(i)+", is_wet: "+str(is_wet), (20, y_position), font, 1, (255, 0, 0), 2)
            # cv2.putText(frame,"test",(10,10),font,10,4,(255,0,0))
        cv2.imshow("wet", frame)

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

def return_rows(path):
    row_list = []
    frame = cv2.imread(path)
    rows = divide_into_three_rows(frame)

    for row in rows:
        is_wet, precent = is_tile_wet(row, wet_precent)
        row_list.append((is_wet, precent))
    
    return row_list

list = return_rows("photos/wet1.jpg")
print(list)


# rows = divide_into_three_rows("photos\WIN_20251127_19_27_09_Pro_EDIT.jpg")
# img = cv2.imread("photos\WIN_20251127_19_27_09_Pro_EDIT.jpg")

# print(is_tile_wet(rows[1], 30))




# lower_bound = np.array([0, 0, 0])
# upper_bound = np.array([180, 255, 152])

# mask1 = cv2.inRange(rows[0], lower_bound, upper_bound)
# mask2 = cv2.inRange(rows[1], lower_bound, upper_bound)
# mask3 = cv2.inRange(rows[2], lower_bound, upper_bound)

# combined = np.vstack((mask1, mask2, mask3))

# resized_comb = cv2.resize(combined, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
# resized_img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
# cv2.imshow('comb', resized_comb)
# cv2.imshow('original', resized_img)





# print(is_tile_wet(rows[0], 60))

# img = cv2.imread("photos\WIN_20251127_19_27_09_Pro_EDIT.jpg")
# hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# lower_bound = np.array([0, 0, 0]),
# upper_bound = np.array([180, 255, 150])
# mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
# resized_img = cv2.resize(mask, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

# cv2.imshow('wet mask', resized_img)


cv2.waitKey(0)