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