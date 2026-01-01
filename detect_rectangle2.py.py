import cv2
import numpy as np


frame = cv2.imread("C:/Users/pc/Downloads/imgRight.png")

def detect_red_rectangle_and_check_center(frame):
    
    if frame is None:
        print("Error: Could not load the image. Check the file path.")
        return
        
    
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA) 
    

    H = frame.shape[0] # Frame height
    W = frame.shape[1] # Frame width
    X_mid_frame = W // 2 
    Y_mid_frame = H // 2 

    # Define tolerance (5% of the frame dimensions)
    x_tolerance = int(W * 0.05)
    y_tolerance = int(H * 0.05)
    
    # Center boundaries for the target to be considered 'Centered'
    center_min_x = X_mid_frame - x_tolerance
    center_max_x = X_mid_frame + x_tolerance
    center_min_y = Y_mid_frame - y_tolerance
    center_max_y = Y_mid_frame + y_tolerance
    
   
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    
    lower_red_1 = np.array([0, 100, 100])
    upper_red_1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    
    lower_red_2 = np.array([170, 100, 100])
    upper_red_2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    
    red_mask = mask1 + mask2
    
  
    kernel = np.ones((5, 5), np.uint8) 
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1) 
    
   
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
        
        if 0.2 < aspect_ratio < 5.0 and area > max_area:
            best_rectangle = cnt
            max_area = area


    
    display_text = "No target detected"
    target_center_color = (0, 0, 255) # Default color for status

    if best_rectangle is not None:
        x, y, w, h = cv2.boundingRect(best_rectangle)
        
        # Calculate the X,Y-center of the detected rectangle
        X_center = x + (w // 2) 
        Y_center = y + (h // 2) 
        
        # heck for CENTERED 
        is_centered_x = (X_center >= center_min_x) and (X_center <= center_max_x)
        is_centered_y = (Y_center >= center_min_y) and (Y_center <= center_max_y)

        if is_centered_x and is_centered_y:
            
            display_text = "Target Centered"
            target_center_color = (0, 255, 0) # Green
      
        # If not centered, the drone need to move
        else:
            command_x = ""
            command_y = ""
            
     
            if X_center < center_min_x:
                command_x = "Left"  # Target is Left of center -> Drone must move Left
            elif X_center > center_max_x:
                command_x = "Right" # Target is Right of center -> Drone must move Right
            

            if Y_center < center_min_y:
                command_y = "Up"    # Target is Up of center -> Drone must move Up
            elif Y_center > center_max_y:
                command_y = "Down"  # Target is Down of center -> Drone must move Down
            

            if command_x and command_y:
                display_text = f"{command_x} {command_y}"
            elif command_x:
                display_text = command_x
            else: 
                display_text = command_y

            target_center_color = (255, 0, 0) 


        

        cv2.rectangle(frame, (x, y), (x + w, y + h), target_center_color, 3)
        
        #the canter of the rectangle
        cv2.circle(frame, (X_center, Y_center), 5, (255, 255, 255), -1) 
        
        # Print info to console
        print(f"Drone Command: {display_text}")

    

    #for the + in the mid
    crosshair_size = int(W * 0.1)
    crosshair_color = (0, 0, 0)


    cv2.line(frame, 
             (X_mid_frame - crosshair_size // 2, Y_mid_frame), # Start X, Center Y
             (X_mid_frame + crosshair_size // 2, Y_mid_frame), # End X, Center Y
             crosshair_color, 2) 

    cv2.line(frame, 
             (X_mid_frame, Y_mid_frame - crosshair_size // 2), # Center X, Start Y
             (X_mid_frame, Y_mid_frame + crosshair_size // 2), # Center X, End Y
             crosshair_color, 2) 

    font = cv2.FONT_HERSHEY_SIMPLEX
    text_position = (10, 30) 
    font_scale = 0.8
    font_thickness = 2
    

    cv2.putText(frame, display_text, text_position, font, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)

    cv2.imshow('Drone Alignment Check', frame) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()

detect_red_rectangle_and_check_center(frame)