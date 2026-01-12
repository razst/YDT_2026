import cv2
import numpy as np
from enum import Enum
import time

#BUG vertical -> horizontal
#BUG what happens if we see two H lines?

class Prevpos:#TODO change this to prev rectangle pos
    def __init__(self,x_pos,y_pos,vertical_lines):
        self.vertical_lines = vertical_lines
        self.x_pos = x_pos
        self.y_pos = y_pos
    
        

FRAME_WIDTH = 800 #BUG remove, use image_height, image_width
FRAME_HEIGHT = 600
DEBUG = True #BUG remove of not used or use

# RED color mask
# Define a WIDER red color range in HSV
# This accounts for lighting variations and different shades of red tape.
lower_red_1 = np.array([0, 80, 50])    # Lowered Saturation and Value minimums
upper_red_1 = np.array([20, 255, 255])
lower_red_2 = np.array([160, 80, 50])
upper_red_2 = np.array([180, 255, 255])


class TargetPosition(Enum):
    LEFT = -1
    CENTER = 0
    RIGHT = 1
    NOT_DETECTED = 2

#BUG test in case of 2 V lines
def detect_y_lines(frame):
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
    min_line_area = FRAME_HEIGHT * 1 #BUG get more realiable number, set FRAME_W / H after rezise
        
    # 4. Find contours
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x_locations = []

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > min_line_area:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if h > FRAME_HEIGHT * 0.01 : 
                

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,255), 2)
                cv2.putText(frame,f"first conditional: {prevPos.y_pos > center_frame[1]}", (200, 100),
                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame,f"second conditional: {center_frame[1] < y}", (200, 120),
                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame,f"prevPos: {prevPos.y_pos}", (300, 200),
                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame,f"curr Pos: {y}", (300, 220),
                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if prevPos.y_pos < center_frame[0] and center_frame[0] < y:
                    prevPos.vertical_lines +=1
                prevPos.y_pos = y
            
    return None


def detect_x_lines(frame) ->TargetPosition:                
    """
    Detects red lines in an image using a wider HSV range and robust filtering.
    Saves the processed mask and the final detected image.
    """
    global center_frame
    try:
        # Make a copy for drawing
        
        # 2. Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # 3. Define a WIDER red color range in HSV
        # This accounts for lighting variations and different shades of red tape.
        
        # Lower red range (H: 0-20) - Increased from 10
        mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)

        # Upper red range (H: 160-180) - Increased lower boundary
        mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)

        # Combine the masks
        red_mask = mask1 + mask2
        _, red_mask = cv2.threshold(red_mask, 150, 255, cv2.THRESH_BINARY)
        
        # Save the mask for visual inspection - CRITICAL FOR DEBUGGING!
        # Check the generated mask_f1.jpg and mask_f2.jpg. If the lines aren't white, 
        # you MUST adjust the HSV ranges above.

        kernel_height = 20
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_height))

        # 2. Apply the Erosion operation (Recommended for filtering thin lines)
        #filtered_mask_erosion = cv2.erode(red_mask, horizontal_kernel, iterations=1)

        # OR

        # 2. Apply the Opening operation
        #filtered_mask_opening = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, horizontal_kernel)
        red_mask = cv2.erode(red_mask, horizontal_kernel, iterations=2)  # More iterations
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, horizontal_kernel)  # Morphological closing
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, horizontal_kernel)  # Morphological opening
        cv2.imshow("red_vertical",red_mask)

        
        # 4. Find contours
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 5. Filter and calculate average X location
        image_height, image_width, _ = frame.shape
        # Estimate a minimum area based on the image size
        min_line_area = image_height * 1    # A simple heuristic (e.g., must be 30px wide * full height)
        #print("min_line_area",min_line_area)
        x_locations = []
        
        # Sort contours by area in descending order
        # We only care about the largest objects, which should be the two lines
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        #print(len(contours))
        # Process the two largest contours, if they meet the area threshold
        for i, contour in enumerate(contours): # Look at most 2 largest
            area = cv2.contourArea(contour)
            #print("area=",area)
            if area > min_line_area:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                #print(x, y, w, h)
                # Further ensure it is a tall object, filtering out small red labels
                if h > image_height * 0.3: 
                    
                    x_locations.append(x)
                    x_locations.append(x+w)
                    # Draw visualization
                    #print(f"Detected line {i+1}: Area={area}, X={x}, Y={y}, W={w}, H={h}")
                    #random_color_tuple = tuple(random.randint(0, 255) for _ in range(3))


                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,255), 2)

        if len(x_locations)<4:
            # print(f"ERROR: Less than two lines detected in frame. Detected X locations: {x_locations}")
            cv2.putText(frame, f"ERROR: Less than two lines detected in frame. Detected X locations: {x_locations}", (20, 40),
                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            #cv2.imshow("frame", frame)
            return TargetPosition.NOT_DETECTED
        # 6. Save the result image
        x_locations.sort()
        # print(len(x_locations))
        # print(x_locations)
        # print(center_frame[0])
        right_w = abs(image_width-x_locations[3]-center_frame[1])
        left_w = abs(center_frame[1] - x_locations[1]) 
        # print(f"right_w: {right_w}")
        # print(f"left_w: {left_w}")
        if abs(left_w-right_w)<100: #BUG - make this 100 a CONST param
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

# --- Main Execution ---
# Path to your video file
video_path = "photos/red_target_vid.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# For FPS calculation
prev_time = 0
target_delay = 1.0 / 25  # 25 FPS → 0.04 seconds per frame

fps_count=0
fps_sum=0
prevPos = Prevpos(0,0,0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA) 
    height,width = frame.shape[:2]
    global center_frame
    center_frame = (height//2,width//2)
    current_time = time.time()

    # FPS = 1 / time between frames
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time
    fps_count +=1
    fps_sum+=fps
    # Put FPS text on frame
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    #BUG put these lines of code in thier own function
    location = detect_x_lines(frame)
    detect_y_lines(frame)
    cv2.putText(frame, f"lines amount:{prevPos.vertical_lines}", (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if location == TargetPosition.LEFT:
        cv2.putText(frame, f"go left", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if location == TargetPosition.CENTER:
        print("in center")
        if prevPos.vertical_lines == 1:
            cv2.putText(frame, f"fire", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, f"in center", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if location == TargetPosition.RIGHT:
        cv2.putText(frame, f"go right", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if location == TargetPosition.NOT_DETECTED:
            cv2.putText(frame, f"not detected", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)    
    if location ==  TargetPosition.NOT_IN_SQUARE:
            cv2.putText(frame, f"not in square", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # BUG use pre rezise img. don't use 800*600
    center_frame = ((FRAME_WIDTH//2),(FRAME_HEIGHT//2))
    radius_circle=20
    thickness_circle=2
    frame = cv2.resize(frame,(FRAME_WIDTH,FRAME_HEIGHT))
    color_circle=(0, 0, 255)
    circle_x = center_frame[0]#image.shape[1]//2
    circle_y = center_frame[1]#image.shape[0]//2
    center_circle=(circle_x,circle_y)
    cv2.line(frame, (center_frame[0]-radius_circle, center_frame[1]), (circle_x+radius_circle, circle_y), color_circle, 2)
    cv2.line(frame, (circle_x, circle_y-radius_circle), (circle_x, circle_y+radius_circle), color_circle, 2)
    cv2.imshow("final",frame)
    # Wait for ANY key to go to next frame
    key = cv2.waitKey(0)  # 0 = wait forever
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(F"avg fps: {fps_sum/fps_count}")
cap.release()
cv2.destroyAllWindows()