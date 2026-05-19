from logger_handler import logger
from mavLink_handler import *
from constants import *
from camera_handler import *
from collections import deque # FIX 1: Import deque instead of queue
import time
from center_detect import *
from record_video import *
from task_man import TaskManager

if __name__ == "__main__":
   
   logger.info("Starting autonumi script 2026...")
   # Connect the FC
   try:
      mavLink = MavLinkHandler(MAV_COM,MAV_MSG_FREQ) 
   except Exception:
      logger.error("Unable to connect to FC")
      
   # wait 4 GUIDED mode
   mavLink.check_until_guided()
      
   try:
      # FIX 2: Create the deque with maxlen=1. 
      cam_buffer = deque(maxlen=1)
      target_buffer = deque(maxlen=1)
      # Connect the Camera
      # Pass the new buffer to the Camera
      camera = Camera(CAMERA_IDX, cam_buffer, True) 
   except Exception:
      logger.error("Unable to connect to Camera")
   center_detect = Detect(cam_buffer,target_buffer,True,TargetColor.BLUE)

   #Start movie recording to file
   recorder = RecordVideo(target_buffer,True)

   # start the mission
   #manager = TaskManager(mavLink, cam_buffer, target_buffer, tasks=[TargetColor.RED, TargetColor.GREEN, TargetColor.BLUE], auto_start=True)   
   
   while True:
      time.sleep(1)
   # wait for mission to end
   
   quit()