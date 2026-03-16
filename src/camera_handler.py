import cv2
import threading
import queue

class Camera:
    def __init__(self, source, frame_queue): #TODO auto start = false
        self.cap = cv2.VideoCapture(source)
        self.q = frame_queue
        self.stopped = False

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
            if self.q.full(): # TODO add logger
                try:
                    self.q.get_nowait() # Drop oldest frame
                except queue.Empty:
                    pass
            
            self.q.put(frame)

    def start(self):
        """Run the update method in a background thread."""
        thread = threading.Thread(target=self.update, daemon=True) #TODO check if damon is necserarry
        thread.start()
        return self

    def stop(self):
        """Stop the loop and release resources."""
        self.stopped = True
        if self.cap.isOpened():
            self.cap.release()