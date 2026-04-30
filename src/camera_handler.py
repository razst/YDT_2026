import cv2
import threading
import time
from logger_handler import logger
# queue import removed!

class Camera:
    # Changed frame_queue to frame_buffer to match your updated main script
    def __init__(self, cam_idx, frame_buffer, auto_start=False):
        self.cap = cv2.VideoCapture(cam_idx)
        self.frame_buffer = frame_buffer
        self.stopped = False

        if auto_start:
            self.start()

    def update(self):
        """
        Reads frames from 'cap' and appends them into 'frame_buffer'.
        """
        prev_time = 0
        while not self.stopped:
            ret, frame = self.cap.read()
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
            prev_time = current_time
            #print(f"fps{fps}")
                
            if not ret:
                print("Stream ended or failed.")
                self.stop()
                break

            # --- THE MAGIC OF DEQUE ---
            # All the "if full, try/except get_nowait()" logic is DELETED.
            # Because we initialized deque with maxlen=1 in main.py, 
            # append() automatically and instantly drops the old frame for you!
            self.frame_buffer.append(frame)

    def start(self):
        thread = threading.Thread(target=self.update, daemon=True) 
        thread.start()
        print("Camera thread started") # Fixed a small typo here (it previously said print("stop"))
        return self

    def stop(self):
        self.stopped = True
        if self.cap.isOpened():
            self.cap.release()