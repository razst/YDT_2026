from center_detect import Detect
from constants import *
from logger_handler import logger

class TaskManager:
    def __init__(self, mavLink, cam_buffer, target_buffer, auto_start=False, tasks=None):
        logger.info("Initializing Task Manager...")
        self.mavLink = mavLink
        self.cam_buffer = cam_buffer
        self.target_buffer = target_buffer
        self.tasks = tasks if tasks is not None else []
        if auto_start:
            self.run_tasks()

    def run_tasks(self):
        for color in self.tasks:
            logger.info(f"Running task for target color: {color}")
            detect = Detect(self.cam_buffer, self.target_buffer, True, color)
            
            # Wait for the task to complete
            logger.info("Waiting for detection task to finish...")
            detect.is_finished.wait() 
            
            logger.info(f"Task for {color} completed. Moving to next phase.")
            
        # RTL ??
        logger.info("All tasks finished. Executing RTL/Land...")
        # Land ??