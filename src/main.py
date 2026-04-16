from logger_handler import logger
from mavLink_handler import *
from constants import *
from camera_handler import *
import queue
#from lines_detect import *
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
   try:
      frame_queue = queue.Queue(maxsize=10)  # Buffer up to 10 frames
      camera = Camera(CAMERA_IDX, frame_queue,True) 
   except Exception:
      logger.error("Unable to connect to FC")
   # wait 4 GUIDED mode
   mavLink.check_until_guided()
   # start the mission
   
   # Start movie recording to file
   recorder = RecordVideo(frame_queue)
   thread_target = threading.Thread(target=recorder.record_main)
   thread_target.start()
   
   # wait for mission to end
   

   quit()