from logger_handler import logger
from mavLink_handler import *
from constants import *
from camera_handler import *
import queue
from target_detect import *

if __name__ == "__main__":
   
   logger.info("Starting autonumi script 2026")
   # Connect the FC
   global mavLink
   try:
      mavLink = MavLinkHandler(MAV_COM,MAV_MSG_FREQ) 
   except Exception:
      logger.error("Unable to connect to FC")

   # Connect the Camera
   try:
      frame_queue = queue.Queue(maxsize=10)  # Buffer up to 10 frames TBD MAXSIZE in constants.py
      camera = Camera(CAMERA_IDX, frame_queue,True) 
   except Exception:
      logger.error("Unable to connect to Camera")
   # wait 4 GUIDED mode
   mode = False
   while mode != True:
      mode = mavLink._check_guided_mode()
   # start the mission
   
   # Start movie recording to file
   # wait for mission to end
   

   quit()