from logger_handler import logger
from mavLink_handler import *
from constants import *
from camera_handler import *
from collections import deque # FIX 1: Import deque instead of queue
import time
from target_detect import *
from record_video import *

if __name__ == "__main__":
   
   logger.info("Starting autonumi script 2026")
   # Connect the FC
   global mavLink
   try:
      mavLink = MavLinkHandler(MAV_COM,MAV_MSG_FREQ) 
   except Exception:
      logger.error("Unable to connect to FC")
      
   # wait 4 GUIDED mode
   mavLink.check_until_guided()
   try:
      # FIX 2: Create the deque with maxlen=1. 
      frame_buffer = deque(maxlen=1)
      video_buffer = deque(maxlen=1)
      # Connect the Camera
      # Pass the new buffer to the Camera
      camera = Camera(CAMERA_IDX, frame_buffer, True) 
   except Exception:
      logger.error("Unable to connect to Camera")

   # start the mission
   target_detect = Detect(frame_buffer,video_buffer, True) 
   
   #Start movie recording to file
   recorder = RecordVideo(video_buffer)
   thread_target = threading.Thread(target=recorder.record_main)
   thread_target.start()
   
   while True:
      time.sleep(1)
   # wait for mission to end
   
   quit()