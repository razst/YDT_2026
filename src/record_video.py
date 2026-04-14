import cv2
import numpy as np
import time
from datetime import datetime
import signal
import sys

# Set the resolution and frame rate
width = 640
height = 480
fps = 30  # Fixed frame rate for the Pi Camera

# Duration for each video file (in seconds)
segment_duration = 15

# Flag to control the recording loop
recording = True

# Signal handler to catch termination signals
def signal_handler(sig, frame):
    global recording
    print("\nTermination signal received. Stopping recording...")
    recording = False

cv2.video
# Register signal handlers for SIGINT (Ctrl+C) and SIGTERM (kill)
# signal.signal(signal.SIGINT, signal_handler) ## use this only if running the main here. otherwise you won't be able to close the program while running 
# signal.signal(signal.SIGTERM, signal_handler)

# Function to create a new VideoWriter with a timestamped filename
def create_new_writer(filename,width,height):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{filename}{timestamp}.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use XVID codec for AVI
    return cv2.VideoWriter(output_file, fourcc, fps, (width, height)), output_file

print(f"Recording video in {segment_duration}-second segments... Press 'q' to stop or Ctrl+C to terminate.")

def main():
    global recording
    # Initialize the PiCamera
    cap = cv2.VideoCapture(0)
    try:
        # Create the first VideoWriter
        out, current_output_file = create_new_writer()
        print(f"Started recording to {current_output_file}")
        
        # Track the start time of the current segment
        segment_start_time = time.time()
        
        while recording:
            # Capture frame from the Pi Camera
            sucsess,frame = cap.read()
            # Check if 15 seconds have passed to start a new file
            elapsed_time = time.time() - segment_start_time
            if elapsed_time >= segment_duration:
                # Release the current VideoWriter
                out.release()
                print(f"Finished recording to {current_output_file}")
                
                # Create a new VideoWriter for the next segment
                out, current_output_file = create_new_writer()
                print(f"Started recording to {current_output_file}")
                
                # Reset the segment start time
                segment_start_time = time.time()
            
            # Write the frame to the current video file
            out.write(frame)
            
            # Display the frame in a window
            cv2.imshow("Pi Camera Live Feed", frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Ensure resources are released even if an error occurs
        print("Cleaning up resources...")
        out.release()
        picam2.stop()
        cv2.destroyAllWindows()
        print(f"Last video saved as {current_output_file}")

if __name__ == "__main__":
    main()

    # ai code remember to add threads

# import cv2
# import time
# from datetime import datetime
# import queue

# class RecordVideo: # Python convention: Classes use CamelCase
#     def __init__(self, frame_queue):
#         self.frame_queue = frame_queue
#         # Set the resolution and frame rate
#         self.width = 640
#         self.height = 480
#         self.fps = 30 
#         self.segment_duration = 15
#         self.recording = True

#     def create_new_writer(self, filename="segment_"):
#         # Use self. attributes so the method can see the config
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         output_file = f"{filename}{timestamp}.avi"
#         fourcc = cv2.VideoWriter_fourcc(*'XVID')
        
#         # Create the writer using class variables
#         writer = cv2.VideoWriter(output_file, fourcc, self.fps, (self.width, self.height))
#         return writer, output_file

#     def main(self):
#         out = None
#         current_output_file = ""
        
#         try:
#             # Pass a default filename prefix
#             out, current_output_file = self.create_new_writer("PiCam_")
#             print(f"Started recording to {current_output_file}")
            
#             segment_start_time = time.time()
            
#             while self.recording:
#                 # 1. Get frame from queue
#                 try:
#                     # Timeout after 1 second so it doesn't hang if the queue is empty
#                     frame = self.frame_queue.get(timeout=1.0)
#                 except queue.Empty:
#                     continue

#                 # Check for a "stop signal" in the queue (if you send None to stop)
#                 if frame is None:
#                     break

#                 # 2. Handle Segment Rotation
#                 elapsed_time = time.time() - segment_start_time
#                 if elapsed_time >= self.segment_duration:
#                     out.release()
#                     print(f"Finished recording to {current_output_file}")
                    
#                     out, current_output_file = self.create_new_writer("PiCam_")
#                     print(f"Started recording to {current_output_file}")
#                     segment_start_time = time.time()
                
#                 # 3. Write and Display
#                 out.write(frame)
                
#                 # Note: imshow in a sub-thread/class can sometimes be unstable 
#                 # depending on your OS, but it's here if you need it.
#                 cv2.imshow("Pi Camera Live Feed", frame)
                
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     self.recording = False

#         except Exception as e:
#             print(f"An error occurred: {e}")

#         finally:
#             print("Cleaning up resources...")
#             if out:
#                 out.release()
#             cv2.destroyAllWindows()
#             if current_output_file:
#                 print(f"Last video saved as {current_output_file}")

# # --- Example of how to run this ---
# # if __name__ == "__main__":
# #     q = queue.Queue()
# #     recorder = RecordVideo(q)
# #     recorder.main()