import cv2
import threading
import queue
from logger_handler import logger

class Camera:
    def __init__(self, cam_idx, frame_queue, auto_start = False):
        self.cap = cv2.VideoCapture(cam_idx)
        self.q = frame_queue
        self.stopped = False

        if auto_start:
            self.start()

    def update(self):
        """
        Reads frames from 'cap' and pushes them into 'q'.
        """
        while not self.stopped:
            ret, frame = self.cap.read()
            
            if not ret:
                print("Stream ended or failed.")
                self.stop()
                break

            # Logic to prevent the queue from bloating if processing is slow
            if self.q.full(): 
                try:
                    logger.info("Frame queue is full")
                    self.q.get_nowait() # Drop oldest frame
                except queue.Empty:
                    pass
            
            self.q.put(frame)

    def start(self):
        thread = threading.Thread(target=self.update, daemon=True) 
        thread.start()
        return self

    def stop(self):
        self.stopped = True
        if self.cap.isOpened():
            self.cap.release()