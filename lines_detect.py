import cv2
import numpy as np
from enum import Enum
import time

#BUG what happens if we see two H lines?

class Prevrect:
    def __init__(self,x_pos,y_pos,horizontal_lines):
        self.horizontal_lines = horizontal_lines
        self.x_pos = x_pos
        self.y_pos = y_pos
# RED color mask
# Define a WIDER red color range in HSV
# This accounts for lighting variations and different shades of red tape.
lower_red_1 = np.array([0, 80, 50])    # Lowered Saturation and Value minimums
upper_red_1 = np.array([20, 255, 255])
lower_red_2 = np.array([160, 80, 50])
upper_red_2 = np.array([180, 255, 255])
global areaY_sum ,areaX_sum
areaY_sum = 0
areaX_sum = 0
CENTER_THRESHOLD = 100
Prevrect = Prevrect(0,0,0)

class TargetPosition(Enum):
    LEFT = -1
    CENTER = 0
    RIGHT = 1
    NOT_DETECTED = 2
#TODO when no line detected go up
def detect_y_lines(frame,position):
    global areaY_sum ,areaX_sum
    global center_frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Lower red range (H: 0-20) - Increased from 10
    mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)

        # Upper red range (H: 160-180) - Increased lower boundary
    mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)

    red_mask = mask1 + mask2
    _, red_mask = cv2.threshold(red_mask, 150, 255, cv2.THRESH_BINARY)


    kernel_height = 20
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_height, 1))


    red_mask = cv2.erode(red_mask, horizontal_kernel, iterations=2)  # More iterations
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, horizontal_kernel)  # Morphological closing
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, horizontal_kernel)  # Morphological opening
    cv2.imshow("red_horizontal",red_mask)
    min_line_area = center_frame[0]*2  * center_frame[0]* 2 * 0.003 # bigger then half a % 
        
    # 4. Find contours
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt_locations = []

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        print(f"y area {area}")
        if area > min_line_area:
            
            x, y, w, h = cv2.boundingRect(contour)
            cnt_locations.append([x,y,w,h])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if len(cnt_locations) == 1:
        if center_frame[0] > cnt_locations[0][1]:
            cv2.putText(frame,f"go up", (10, 60),cv2.FONT_HERSHEY_SIMPLEX, 1,(150, 200, 210), 2)
        elif center_frame[0] < cnt_locations[0][1] and position == TargetPosition.NOT_DETECTED:
            cv2.putText(frame,f"go up", (10, 60),cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 200, 210), 2)
        else:
            cv2.putText(frame,f"go down", (10, 60),cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 200, 210), 2)

    elif len(cnt_locations) == 2:
        cv2.putText(frame,f"go up", (10, 60),cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 200, 210), 2)
            
    return None


def detect_x_lines(frame) ->TargetPosition:                
    """
    Detects red lines in an image using a wider HSV range and robust filtering.
    Saves the processed mask and the final detected image.
    """
    global center_frame
    global areaY_sum ,areaX_sum
    try:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
        mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
        red_mask = mask1 + mask2
        _, red_mask = cv2.threshold(red_mask, 150, 255, cv2.THRESH_BINARY)
        kernel_height = 20
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_height))


        red_mask = cv2.erode(red_mask, horizontal_kernel, iterations=2)  # More iterations
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, horizontal_kernel)  # Morphological closing
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, horizontal_kernel)  # Morphological opening
        cv2.imshow("red_vertical",red_mask)
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_height, image_width, _ = frame.shape
        min_line_area = image_height * image_width * 0.004 #bigger then half a % of the screen 
        x_locations = []
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        # Process the two largest contours, if they meet the area threshold
        for i, contour in enumerate(contours): # Look at most 2 largest
            area = cv2.contourArea(contour)
            areaX_sum += area
            print(f"x area  {area}")
            if area > min_line_area:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                #print(x, y, w, h)
                if h > 30: #bigger then 1% of the screen                       
                    x_locations.append(x)
                    x_locations.append(x+w)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,255), 2)

        #if len(x_locations)<4:
        #    cv2.putText(frame, f"ERROR: Less than two lines detected in frame. Detected X locations: {x_locations}", (10, 110),
        #         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            #cv2.imshow("frame", frame)
            return TargetPosition.NOT_DETECTED
        x_locations.sort()
        right_w = abs(image_width-x_locations[3]-center_frame[1])
        left_w = abs(center_frame[1] - x_locations[1]) 
        if abs(left_w-right_w)<CENTER_THRESHOLD: 
            print("center")
            return TargetPosition.CENTER
        elif left_w>right_w:                
            print("left")
            return TargetPosition.LEFT
        elif right_w > left_w:
            print("right")
            return TargetPosition.RIGHT  
        else:
            return TargetPosition.NOT_DETECTED

    except Exception as e:
        print(f"An error occurred while processing frame: {e}")
        return []
def detect_white(frame):
    # 2. Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 3. Define the range for 'White'
    # White has low saturation (0-50) and high value (200-255)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])

    # 4. Create a mask and find contours
    mask = cv2.inRange(hsv, lower_white, upper_white)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
        # Process the two largest contours, if they meet the area threshold
    for i, contour in enumerate(contours): # Look at most 2 largest
        area = cv2.contourArea(contour)
        if area > 0.1:
            x, y, w, h = cv2.boundingRect(contour)
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if h > 50: #bigger then 1% of the screen                       
                cv2.rectangle(frame, (x, y), (x + w, y + h), (128, 0, 0), 2)
def show_image(frame,fps):

    cv2.putText(frame, f"FPS: {fps:.2f}", (10,150),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    #TODO make this the only detect that is needed
    position = detect_x_lines(frame)
    detect_y_lines(frame,position)
    detect_white(frame)
    cv2.putText(frame, f"lines amount:{Prevrect.horizontal_lines}", (10,170),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if position == TargetPosition.LEFT:
        put_text = f"go left"
    if position == TargetPosition.CENTER:
        print("in center")
        if Prevrect.horizontal_lines == 1:
            put_text = f"fire"
        else:
            put_text = f"in center"
    if position == TargetPosition.RIGHT:
        put_text =  f"go right"
    if position == TargetPosition.NOT_DETECTED:
            put_text = f"not detected"
    cv2.putText(frame, put_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    radius_circle=20
    thickness_circle=2
    color_circle=(0, 0, 255)
    circle_x = center_frame[1]
    circle_y = center_frame[0]
    center_circle=(circle_x,circle_y)
    cv2.line(frame, (center_frame[1]-radius_circle, center_frame[0]), (circle_x+radius_circle, circle_y), color_circle, 2)
    cv2.line(frame, (circle_x, circle_y-radius_circle), (circle_x, circle_y+radius_circle), color_circle, 2)
    cv2.imshow("final",frame)
def main():
    # --- Main Execution ---
    # Path to your video file
    video_path = "C:/Users/user/Downloads/epsteinFiles.mp4"
    cap = cv2.VideoCapture(video_path)
    prev_time = 0
    fps_count=0
    fps_sum=0

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        current_time = time.time()

        fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
        prev_time = current_time
        fps_count +=1
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA) 
        height,width = frame.shape[:2]
        global center_frame
        center_frame = (height//2,width//2)
        show_image(frame,fps)

        # Wait for ANY key to go to next frame
        key = cv2.waitKey(0)  # 0 = wait forever
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print(F"avg fps: {fps_sum/fps_count}")
    print(F"horizontal avarage: {(areaY_sum/fps_count) / (height * width) * 100}, percent"              )
    print(F"vertical average is : {(areaX_sum /fps_count) /(height * width) * 100            }, percent")
    print(f"{(height * width)}")
    cap.release()
    cv2.destroyAllWindows()
main()