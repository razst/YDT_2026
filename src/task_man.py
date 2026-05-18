from target_detect import Detect
from constants import *
from logger_handler import logger

class TaskManager:

    # tasks is a list of targetcolors 
    def __init__(self,mavLink, cam_buffer, target_buffer,auto_start=False,tasks=None):
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
           
            # wait for the task to complete
        # RTL ??
        # Land ??