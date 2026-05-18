import cv2
import numpy as np
import time
import threading
from collections import deque
from enum import Enum

class TargetPositionX(Enum):
    LEFT = -1
    CENTER = 0
    RIGHT = 1
    NOT_DETECTED = 2

class TargetPositionY(Enum):
    DOWN = -1
    CENTER = 0
    UP = 1
    NOT_DETECTED = 2

class TargetDetector:
    def __init__(self, frame_buffer, record_buffer=None, auto_start=False):
        self.frame_buffer = frame_buffer
        self.record_buffer = record_buffer
        self.running = True
        
        # HSV Processing ranges for Red
        self.lower_1 = np.array([0, 100, 100])
        self.upper_1 = np.array([10, 255, 255])
        self.lower_2 = np.array([170, 100, 100])
        self.upper_2 = np.array([180, 255, 255])
        
        # Detection Constants
        self.MIN_AREA = 500
        self.MIN_ASPECT_RATIO = 0.45
        self.MAX_ASPECT_RATIO = 0.57
        self.TOLERANCE_PCT = 0.03
        
        # Target state
        self.last_bbox = None
        
        if auto_start:
            self.start()

    def process_target(self, frame):
        """
        Detects the red target, calculates aspect ratios, and determines alignment.
        Returns the annotated frame, cropped frame, and X/Y positions.
        """
        h, w, _ = frame.shape
        x_mid, y_mid = w // 2, h // 2
        x_tol, y_tol = int(w * self.TOLERANCE_PCT), int(h * self.TOLERANCE_PCT)

        # HSV Processing
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.lower_1, self.upper_1)
        mask2 = cv2.inRange(hsv, self.lower_2, self.upper_2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # Morphological operations to remove noise
        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        max_area = 0
        best_cnt = None

        # Filter contours by area and aspect ratio
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > self.MIN_AREA:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                aspect_ratio = float(bw) / bh
                if self.MIN_ASPECT_RATIO < aspect_ratio < self.MAX_ASPECT_RATIO and area > max_area:
                    best_cnt = cnt
                    max_area = area

        pos_x = TargetPositionX.NOT_DETECTED
        pos_y = TargetPositionY.NOT_DETECTED
        dir_x, dir_y = "", "No target detected"
        cropped_frame = np.zeros((100, 100, 3), dtype=np.uint8) # Default empty crop

        if best_cnt is not None:
            bx, by, bw, bh = cv2.boundingRect(best_cnt)
            self.last_bbox = (bx, by, bw, bh)
            cx, cy = bx + bw // 2, by + bh // 2
            
            # Extract Cropped Frame
            cropped_frame = frame[by:by+bh, bx:bx+bw].copy()

            # X-Axis Alignment
            if (x_mid - x_tol < cx < x_mid + x_tol):
                pos_x = TargetPositionX.CENTER
                dir_x = "CENTER in X"
            elif cx < x_mid - x_tol:
                pos_x = TargetPositionX.LEFT
                dir_x = "Left"
            else:
                pos_x = TargetPositionX.RIGHT
                dir_x = "Right"

            # Y-Axis Alignment
            if (y_mid - y_tol < cy < y_mid + y_tol):
                pos_y = TargetPositionY.CENTER
                dir_y = "CENTER in Y"
            elif cy > y_mid + y_tol:
                pos_y = TargetPositionY.DOWN
                dir_y = "Down"
            else:
                pos_y = TargetPositionY.UP
                dir_y = "Up"

            # Draw Bounding Box and Center Point
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (255, 0, 0), 3)
            cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)

        # Draw Crosshairs
        cv2.line(frame, (x_mid - 20, y_mid), (x_mid + 20, y_mid), (0, 0, 0), 2)
        cv2.line(frame, (x_mid, y_mid - 20), (x_mid, y_mid + 20), (0, 0, 0), 2)

        # Overlay Text
        cv2.putText(frame, dir_y, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, dir_x, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        return frame, cropped_frame, pos_x, pos_y

    def start(self):
        thread = threading.Thread(target=self.update, daemon=True)
        thread.start()

    def update(self):
        prev_time = time.time()
        
        while self.running:
            try:
                # Pop the latest frame from the right side of the deque
                frame = self.frame_buffer.pop()
                # Clear out older frames so we don't build up latency
                self.frame_buffer.clear() 
            except IndexError:
                time.sleep(0.001)
                continue 

            current_time = time.time()
            elapsed = current_time - prev_time
            fps = 1 / elapsed if elapsed > 0.001 else 0
            prev_time = current_time

            # Process the frame
            annotated_frame, cropped_frame, pos_x, pos_y = self.process_target(frame)

            # Draw FPS
            cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Route to outgoing buffer if provided
            if self.record_buffer is not None:
                self.record_buffer.append(annotated_frame)

            # Display Output
            cv2.imshow('Drone Alignment Check', annotated_frame)
            if cropped_frame.shape[0] > 0 and cropped_frame.shape[1] > 0:
                cv2.imshow('get_cropped_rectangle', cropped_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break
                
        cv2.destroyAllWindows()


def main(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Thread-safe buffer for frames
    frame_buffer = deque(maxlen=5) 
    
    # Initialize the detector class and auto-start the background thread
    detector = TargetDetector(frame_buffer=frame_buffer, auto_start=True)

    print("Starting video feed... Press 'q' in the video window to quit.")
    
    while detector.running:
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read frame.")
            detector.running = False
            break

        # Resize for performance (matches your fx=0.5, fy=0.5 requirement)
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        
        # Append to the deque. The background thread will pop and process it.
        frame_buffer.append(frame)
        
        # Small sleep to yield to background thread
        time.sleep(0.01)

    cap.release()

if __name__ == "__main__":
    video_path = "C:/Users/user/Downloads/epsteinFiles.mp4"
    main(video_path)