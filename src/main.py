from logger_handler import logger
from mavLink_handler import *
from constants import *
from camera_handler import *
import queue
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

   # Connect the Camera
   mavLink.check_until_guided()
   try:
      frame_queue = queue.Queue(maxsize=5)  # Buffer up to 10 frames TBD MAXSIZE in constants.py
      camera = Camera(CAMERA_IDX, frame_queue,True) 
   except Exception:
      logger.error("Unable to connect to Camera")
   # wait 4 GUIDED mode
   # start the mission
   target_detect = Detect(frame_queue,True) 
   # Start movie recording to file
   #recorder = RecordVideo(frame_queue)
   #thread_target = threading.Thread(target=recorder.record_main)
   #thread_target.start()
   while True:
      pass
   
   # wait for mission to end
   

   quit()