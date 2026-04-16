

    #ai code remember to add threads

import cv2
import time
from datetime import datetime
import queue

class RecordVideo: # Python convention: Classes use CamelCase
    def __init__(self, frame_queue):
        self.frame_queue = frame_queue
        # Set the resolution and frame rate
        self.width = 640
        self.height = 480
        self.fps = 30 
        self.segment_duration = 15
        self.recording = True

    def create_new_writer(self, filename="segment_"):
        # Use self. attributes so the method can see the config
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{filename}{timestamp}.avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        
        # Create the writer using class variables
        writer = cv2.VideoWriter(output_file, fourcc, self.fps, (self.width, self.height))
        return writer, output_file

    def record_main(self):
        out = None
        current_output_file = ""
        print("got frame from queue")
        
        try:
            # Pass a default filename prefix
            out, current_output_file = self.create_new_writer("PiCam_")
            print(f"Started recording to {current_output_file}")
            
            segment_start_time = time.time()
            
            while self.recording:
                # 1. Get frame from queue
                try:
                    # Timeout after 1 second so it doesn't hang if the queue is empty
                    frame = self.frame_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                # Check for a "stop signal" in the queue (if you send None to stop)
                if frame is None:
                    break

                # 2. Handle Segment Rotation
                elapsed_time = time.time() - segment_start_time
                if elapsed_time >= self.segment_duration:
                    out.release()
                    print(f"Finished recording to {current_output_file}")
                    
                    out, current_output_file = self.create_new_writer("PiCam_")
                    print(f"Started recording to {current_output_file}")
                    segment_start_time = time.time()
                
                # 3. Write and Display
                out.write(frame)
                
                # Note: imshow in a sub-thread/class can sometimes be unstable 
                # depending on your OS, but it's here if you need it.
                cv2.imshow("Pi Camera Live Feed", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.recording = False

        except Exception as e:
            print(f"An error occurred: {e}")

        finally:
            print("Cleaning up resources...")
            if out:
                out.release()
            cv2.destroyAllWindows()
            if current_output_file:
                print(f"Last video saved as {current_output_file}")

#--- Example of how to run this ---
if __name__ == "__main__":
    q = queue.Queue()
    recorder = RecordVideo(q)
    recorder.main()