import cv2
import numpy as np
from enum import Enum
import time
import threading
from collections import deque

class TargetPosition(Enum):
    LEFT = 0
    RIGHT = 1
    CENTER = 2
    CENTER_FIRE = 3
    NOT_DETECTED = 4

class Detect:
    # ADDED: record_buffer parameter
    def __init__(self, frame_buffer, record_buffer=None, auto_start=False):
        self.lower_red_1 = np.array([0, 80, 50])
        self.upper_red_1 = np.array([20, 255, 255])
        self.lower_red_2 = np.array([160, 80, 50])
        self.upper_red_2 = np.array([180, 255, 255])
        self.CENTER_THRESHOLD = 0.25
        self.frame_buffer = frame_buffer
        self.record_buffer = record_buffer # Save the outgoing buffer
        
        if auto_start:
            self.start()

    def detect_x_lines(self, frame, center_frame) -> TargetPosition:                
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(hsv, self.lower_red_1, self.upper_red_1)
            mask2 = cv2.inRange(hsv, self.lower_red_2, self.upper_red_2)
            red_mask = cv2.bitwise_or(mask1, mask2)

            red_mask = cv2.erode(red_mask, None, iterations=1)

            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            image_height, image_width = frame.shape[:2]
            min_line_area = image_height * image_width * 0.001 
            
            valid_contours = []
            for c in contours:
                area = cv2.contourArea(c)
                if area > min_line_area:
                    valid_contours.append((area, c))
                    
            valid_contours.sort(key=lambda x: x[0], reverse=True)
            
            rect_locations = []
            for area, contour in valid_contours[:2]: 
                x, y, w, h = cv2.boundingRect(contour)
                if h > 15: 
                    rect_locations.append((x, y, x + w, y + h))
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)

            if len(rect_locations) < 2:
                return TargetPosition.NOT_DETECTED 
                
            rect_locations.sort(key=lambda r: r[0])
            
            right_w = abs(image_width - rect_locations[1][0] - center_frame[1])
            left_w = abs(center_frame[1] - rect_locations[0][2]) 
            w = rect_locations[1][0] - rect_locations[0][2] 
            
            if w <= 0:
                return TargetPosition.NOT_DETECTED

            if abs(left_w - right_w) < self.CENTER_THRESHOLD * w:
                if (center_frame[0] <= rect_locations[0][3]) and (rect_locations[0][1] <= center_frame[0]):
                    return TargetPosition.CENTER_FIRE
                return TargetPosition.CENTER
            elif left_w >= right_w:                
                return TargetPosition.LEFT
            elif right_w > left_w:
                return TargetPosition.RIGHT  

        except Exception as e:
            return TargetPosition.NOT_DETECTED

    def show_image(self, frame, fps, center_frame):
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        position = self.detect_x_lines(frame, center_frame)
        
        put_text = {
            TargetPosition.CENTER_FIRE: "center fire",
            TargetPosition.LEFT: "go left",
            TargetPosition.CENTER: "center",
            TargetPosition.RIGHT: "go right",
            TargetPosition.NOT_DETECTED: "not detected"
        }.get(position, "")

        cv2.putText(frame, put_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        radius_circle = 15
        color_circle = (0, 0, 255)
        circle_x, circle_y = center_frame[1], center_frame[0]
        
        cv2.line(frame, (circle_x - radius_circle, circle_y), (circle_x + radius_circle, circle_y), color_circle, 2)
        cv2.line(frame, (circle_x, circle_y - radius_circle), (circle_x, circle_y + radius_circle), color_circle, 2)
        
        cv2.imshow("final", frame)

    def start(self):
        thread = threading.Thread(target=self.update, daemon=True)
        thread.start()
        
    def update(self):
        prev_time = time.time()
        
        while True:
            try:
                frame = self.frame_buffer.pop()
            except IndexError:
                time.sleep(0.001)
                continue 

            current_time = time.time()
            elapsed = current_time - prev_time
            fps = 1 / elapsed if elapsed > 0.001 else 0
            prev_time = current_time
            
            target_width = 480
            aspect_ratio = frame.shape[0] / frame.shape[1]
            target_height = int(target_width * aspect_ratio)
            frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR) 
            
            height, width = frame.shape[:2]
            center_frame = (height // 2, width // 2) 
            
            # This draws all the boxes and text onto the frame
            self.show_image(frame, fps, center_frame)

            # ADDED: Send the fully processed frame to the recorder buffer
            if self.record_buffer is not None:
                self.record_buffer.append(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cv2.destroyAllWindows()