import cv2
import numpy as np
import time
from enum import Enum
video_path = "C:/Users/user/Downloads/epsteinFiles.mp4"

class TargetPosition(Enum):
    LEFT = -1
    CENTER = 0
    RIGHT = 1
    NOT_DETECTED = 2
    DOWN=-1
    UP=1


# gets a frame and instrcut how to get to its center
# input: cv2 frame
# output: frame, horz (-1:left,0:center,1:right), vart (-1:down,0:center,1:up)

def center_detect(frame):
    horz,vert = 0,0
    H, W, _ = frame.shape
    X_mid_frame = W // 2
    Y_mid_frame = H // 2
    x_tol, y_tol = int(W * 0.05), int(H * 0.05)

    # HSV Processing for Red
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
    red_mask = mask1 + mask2

    # Cleaning up the noise
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    # Finding Contours TODO find only Contours that are "closed"
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    display_text = "No target detected"
    cv2.imshow("red_horizontal",red_mask)
    # TODO wh ywhen these are moved inside the loop, we don't recognise all frames ???
    last_cnt = None 
    max_area = 0

    # Logic to find the largest rectangle
    print("count of contoutrs",len(contours))
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print("area:",area)
        if area > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            print(f"aspect ratio: {aspect_ratio}")
            if 0.45 < aspect_ratio < 0.57 and area > max_area: #TODO make the numbers CONSTS
                last_cnt = cnt
                max_area = area

    if last_cnt is not None:
        x, y, w, h = cv2.boundingRect(last_cnt)
        cx, cy = x + w//2, y + h//2
        # Check if centered
        if (X_mid_frame - x_tol < cx < X_mid_frame + x_tol):
            horz=TargetPosition.CENTER
            dir_x = "CENTER in X"
        if(Y_mid_frame - y_tol < cy < Y_mid_frame + y_tol):  
            vert=TargetPosition.CENTER
            dir_x = "CENTER in Y"
        if(cx < X_mid_frame - x_tol):
            dir_x = "Left"
            horz=TargetPosition.LEFT
        else:
            horz=TargetPosition.RIGHT
            dir_x= "Right"
        if(cy > Y_mid_frame + y_tol):
            vert=TargetPosition.DOWN
            dir_y="Down"
        else:
            vert=TargetPosition.UP
            dir_y="up"

        display_text = f"{dir_x} {dir_y}".strip()
        target_color = (255, 0, 0) # Blue
        cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), target_color, 3)
        cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)
    # Draw Crosshair
    cv2.line(frame, (X_mid_frame-20, Y_mid_frame), (X_mid_frame+20, Y_mid_frame), (0,0,0), 2)
    cv2.line(frame, (X_mid_frame, Y_mid_frame-20), (X_mid_frame, Y_mid_frame+20), (0,0,0), 2)

    return frame,horz,vert



def main(video_path):
    # cap = cv2.VideoCapture(path)
    cap = cv2.VideoCapture(video_path)
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

        current_time = time.time()
        # FPS = 1 / time between frames
        fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
        prev_time = current_time
        fps_count +=1
        fps_sum+=fps

        #Resize once (0.5 is usually enough)
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA) 
        f,h,v = center_detect(frame)
                # Put FPS text on frame
        cv2.putText(frame, f"FPS: {fps:.2f}", (30, 100),

                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Drone Alignment Check', frame)

        # Press 'q' to quit early.
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    print("Avg FPS:",fps_sum/fps_count)
    cap.release()
    cv2.destroyAllWindows()

# CORRECT CALL: Pass the path variable, not 'frame'

if __name__ == "__main__":

    main(video_path)

    cv2.waitKey(0)