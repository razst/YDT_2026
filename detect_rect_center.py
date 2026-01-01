import cv2
import numpy as np

# Ensure the file path is correct for your image
frame = cv2.imread("C:/Users/pc/Downloads/WIN_20251127_19_16_19_Pro.jpg")

def detect_red_rectangle_and_check_center(frame):
    # Check if the image was loaded successfully
    if frame is None:
        print("Error: Could not load the image. Check the file path.")
        return
        
    # Resize the frame for faster processing and noise reduction
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA) 
    
    # 1. Define Frame Center and Tolerance
    W = frame.shape[1] 
    X_mid_frame = W // 2 # Center of the entire processed frame
    
    # Define tolerance (e.g., 5% of the frame width)
    tolerance = int(W * 0.05) 
    center_min_x = X_mid_frame - tolerance
    center_max_x = X_mid_frame + tolerance
    
    # --- Red Mask and Contour Detection (Robust code) ---
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Red Color Ranges
    lower_red_1 = np.array([0, 100, 100])
    upper_red_1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    
    lower_red_2 = np.array([170, 100, 100])
    upper_red_2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    
    red_mask = mask1 + mask2
    
    # Morphological Operations to connect and clean the red contour
    kernel = np.ones((5, 5), np.uint8) 
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1) 
    
    # Find external contours
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_rectangle = None
    max_area = 0 
    
    # Find the largest, rectangle-like contour
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        
        # Filter by Aspect Ratio and Area
        if 0.2 < aspect_ratio < 5.0 and area > max_area:
            best_rectangle = cnt
            max_area = area

    # --- Center Calculation and Status Determination ---
    if best_rectangle is None:
        print("Status: No rectangle detected.")
    else:
        # Calculate the Bounding Box parameters
        x, y, w, h = cv2.boundingRect(best_rectangle)
        
        # Calculate the X-center of the rectangle
        X_center = (x + (x + w)) // 2 
        
        # Check the Center Status
        if center_min_x <= X_center <= center_max_x:
            center_status = "Centered"
            color = (0, 255, 0) # Green
        elif X_center < center_min_x:
            center_status = "Too Far Left"
            color = (0, 0, 255) # Red
        else:
            center_status = "Too Far Right"
            color = (0, 0, 255) # Red
        
        # --- Visual Feedback (DRAWING ONLY ONE LINE) ---
        
        # 1. Draw the Bounding Box with the status color
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
        
        # 2. Draw the SINGLE vertical line at the Rectangle's X_center (Blue line)
        # Note: We use a different color (e.g., Cyan/Light Blue (255, 255, 0) is Yellow)
        # Let's use Bright Blue (255, 0, 0) for the center line of the rectangle itself
        cv2.line(frame, (X_center, 0), (X_center, frame.shape[0]), (255, 0, 0), 3) 

        # Print the required values and the final status
        print(f"Frame Width (W): {W}")
        print(f"Rectangle Center (X_center): {X_center}")
        print(f"Final Center Status: **{center_status}**")

        cv2.imshow('Rectangle Center Line', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Execute the function
detect_red_rectangle_and_check_center(frame)