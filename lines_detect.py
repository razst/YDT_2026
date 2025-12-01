import cv2
import numpy as np
from enum import Enum
import time
import random

DEBUG = True


class TargetPosition(Enum):
    LEFT = -1
    CENTER = 0
    RIGHT = 1
    NOT_DETECTED = 2

def detect_lines_and_get_x_locations(frame) ->TargetPosition:
    """
    Detects red lines in an image using a wider HSV range and robust filtering.
    Saves the processed mask and the final detected image.
    """
    try:
        # Make a copy for drawing
        output_image = frame.copy()
        
        # 2. Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 3. Define a WIDER red color range in HSV
        # This accounts for lighting variations and different shades of red tape.
        
        # Lower red range (H: 0-20) - Increased from 10
        lower_red_1 = np.array([0, 80, 50])    # Lowered Saturation and Value minimums
        upper_red_1 = np.array([20, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)

        # Upper red range (H: 160-180) - Increased lower boundary
        lower_red_2 = np.array([160, 80, 50])
        upper_red_2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)

        # Combine the masks
        red_mask = mask1 + mask2
        _, red_mask = cv2.threshold(red_mask, 150, 255, cv2.THRESH_BINARY)
        
        # Save the mask for visual inspection - CRITICAL FOR DEBUGGING!
        # Check the generated mask_f1.jpg and mask_f2.jpg. If the lines aren't white, 
        # you MUST adjust the HSV ranges above.

        # Optional: Clean up the mask using morphological operations
        kernel = np.ones((1, 1), np.uint8) # Slightly larger kernel for better closing
        red_mask = cv2.erode(red_mask, kernel, iterations=1)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel) 
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        
        # 4. Find contours
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 5. Filter and calculate average X location
        image_height, image_width, _ = frame.shape
        # Estimate a minimum area based on the image size
        min_line_area = image_height * 30    # A simple heuristic (e.g., must be 30px wide * full height)
        print("min_line_area",min_line_area)
        x_locations = []
        
        # Sort contours by area in descending order
        # We only care about the largest objects, which should be the two lines
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        print(len(contours))
        # Process the two largest contours, if they meet the area threshold
        for i, contour in enumerate(contours): # Look at most 2 largest
            area = cv2.contourArea(contour)
            print("area=",area)
            if area > min_line_area:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                print(x, y, w, h)
                # Further ensure it is a tall object, filtering out small red labels
                if h > image_height * 0.5: 
                    
                    x_locations.append(x)
                    # Draw visualization
                    print(f"Detected line {i+1}: Area={area}, X={x}, Y={y}, W={w}, H={h}")
                    random_color_tuple = tuple(random.randint(0, 255) for _ in range(3))


                    cv2.rectangle(output_image, (x, y), (x + w, y + h), random_color_tuple, 2)

        if len(x_locations)<2:
            print(f"ERROR: Less than two lines detected in frame. Detected X locations: {x_locations}")
            cv2.resize(output_image,(800,600))
            cv2.putText(output_image, f"ERROR: Less than two lines detected in frame. Detected X locations: {x_locations}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("output_image", output_image)
            return TargetPosition.NOT_DETECTED
        # 6. Save the result image
        else:
            cv2.resize(output_image,(800,600))
            cv2.imshow("output_image", output_image)

        right_w = image_width-max(x_locations)
        left_w = min(x_locations)
        print(x_locations, left_w, right_w)
        if abs(left_w-right_w)<10:
            return TargetPosition.CENTER
        elif left_w>right_w:
            return TargetPosition.LEFT
        else:
            return TargetPosition.RIGHT  

    except Exception as e:
        print(f"An error occurred while processing frame: {e}")
        return []

# --- Main Execution ---
# Path to your video file
video_path = "red_target_vid.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# For FPS calculation
prev_time = 0
target_delay = 1.0 / 25  # 25 FPS → 0.04 seconds per frame
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)    # Time now
    current_time = time.time()

    # FPS = 1 / time between frames
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    # Put FPS text on frame
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    detect_lines_and_get_x_locations(frame)

    # Wait for ANY key to go to next frame
    key = cv2.waitKey(0)  # 0 = wait forever
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()