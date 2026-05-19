import cv2
import numpy as np
import time
from enum import Enum
from collections import deque
import threading
from constants import *

class TargetPosition(Enum):
    LEFT = -1
    CENTER = 0
    RIGHT = 1
    NOT_DETECTED = 2
    DOWN = -1
    UP = 1

class TargetColor(Enum):
    RED = 0
    GREEN = 1
    BLUE = 2


class Detect:
    def __init__(self, frame_buffer, record_buffer=None, auto_start=False, target_color=TargetColor.RED):
        self.frame_queue = frame_buffer
        self.target_bbox = None 
        self.record_buffer = record_buffer
        if target_color == TargetColor.RED:
            self.lower_1 = np.array([0, 80, 50])
            self.upper_1 = np.array([20, 255, 255])
            self.lower_2 = np.array([160, 80, 50])
            self.upper_2 = np.array([180, 255, 255])
        elif target_color == TargetColor.GREEN:
            self.lower_1 = np.array([40, 80, 50])
            self.upper_1 = np.array([80, 255, 255])
            self.lower_2 = np.array([40, 80, 50])
            self.upper_2 = np.array([80, 255, 255])
        elif target_color == TargetColor.BLUE:
            self.lower_1 = np.array([100, 80, 50])
            self.upper_1 = np.array([140, 255, 255])
            self.lower_2 = np.array([100, 80, 50])
            self.upper_2 = np.array([140, 255, 255])
            
        if auto_start:
            self.start()

    def center_detect(self, frame):
        adited_frame = frame
        original_frame = frame.copy() # TODO can we remove this ???
        horz, vert = TargetPosition.NOT_DETECTED, TargetPosition.NOT_DETECTED
        H, W, _ = adited_frame.shape
        X_mid_adited_frame = W // 2
        Y_mid_adited_frame = H // 2
        x_tol, y_tol = int(W * 0.03), int(H * 0.03) # TODO: Consts ???

        # HSV Processing for Red 
        hsv = cv2.cvtColor(adited_frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.lower_1, self.upper_1)
        mask2 = cv2.inRange(hsv, self.lower_2, self.upper_2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # Cleaning up the noise
        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        
        # Finding Contours
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not IS_HEADLESS:
            cv2.imshow("red_horizontal", red_mask)
        
        max_area = 0
        max_cnt = None

        # Logic to find the largest rectangle
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h
                if 0.45 < aspect_ratio < 0.57 and area > max_area:
                    max_cnt = cnt
                    max_area = area

        dir_x, dir_y = "", "No target detected"

        if max_cnt is not None:
            x, y, w, h = cv2.boundingRect(max_cnt)
            self.target_bbox = (x, y, w, h)
            
            cx, cy = x + w // 2, y + h // 2
            
            # Check if centered horizontally
            if (X_mid_adited_frame - x_tol < cx < X_mid_adited_frame + x_tol):
                horz = TargetPosition.CENTER
                dir_x = "CENTER in X"
            elif (cx < X_mid_adited_frame - x_tol):
                dir_x = "Left"
                horz = TargetPosition.LEFT
            else:
                horz = TargetPosition.RIGHT
                dir_x = "Right"

            # Check if centered vertically
            if (Y_mid_adited_frame - y_tol < cy < Y_mid_adited_frame + y_tol):
                vert = TargetPosition.CENTER
                dir_y = "CENTER in Y"
            elif (cy > Y_mid_adited_frame + y_tol):
                vert = TargetPosition.DOWN
                dir_y = "Down"
            else:
                vert = TargetPosition.UP
                dir_y = "Up"

            display_x = dir_x.strip()
            display_y = dir_y.strip()
            target_color = (255, 0, 0)
            
            cv2.rectangle(adited_frame, (x, y), (x + w, y + h), target_color, 3)
            cv2.circle(adited_frame, (cx, cy), 5, (255, 255, 255), -1)
        else:
            self.target_bbox = None 

        display_x = dir_x.strip()
        display_y = dir_y.strip()

        cv2.putText(adited_frame, display_y, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 105, 255), 2)
        cv2.putText(adited_frame, display_x, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 105, 255), 2)
        
        # Draw Crosshair
        cv2.line(adited_frame, (X_mid_adited_frame - 20, Y_mid_adited_frame), (X_mid_adited_frame + 20, Y_mid_adited_frame), (0, 0, 0), 2)
        cv2.line(adited_frame, (X_mid_adited_frame, Y_mid_adited_frame - 20), (X_mid_adited_frame, Y_mid_adited_frame + 20), (0, 0, 0), 2)

        return original_frame, adited_frame, horz, vert

    # imshow the detected object
    def get_cropped_rectangle(self, original_frame):
        if self.target_bbox is None:
            return None
        
        x, y, w, h = self.target_bbox
        cropped_frame = original_frame[y:y+h, x:x+w]
        if not IS_HEADLESS and cropped_frame is not None and cropped_frame.size > 0:
            cv2.imshow('get_cropped_rectangle', cropped_frame)
        return cropped_frame
    

    def start(self):
        thread = threading.Thread(target=self.process_frames, daemon=True)
        thread.start()
        

    def process_frames(self):
        prev_time = time.time()
        fps_count = 0
        fps_sum = 0

        while True:
            if len(self.frame_queue) > 0:
                frame = self.frame_queue.popleft() 

                current_time = time.time()
                fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
                prev_time = current_time
                fps_count += 1
                fps_sum += fps

                frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA) 
                
                original_frame, edited_frame, h, v = self.center_detect(frame)
                # cropped_frame = self.get_cropped_rectangle(original_frame)
                
                if not IS_HEADLESS:
                    cv2.putText(edited_frame, f"FPS: {fps:.2f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.imshow('Drone Alignment Check', edited_frame)
                
                if self.record_buffer is not None:
                    self.record_buffer.append(edited_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if not IS_HEADLESS:                
            cv2.destroyAllWindows()