from logger_handler import logger
from mavLink_handler import *
from constants import *


if __name__ == "__main__":
    global mavLink
    logger.info("Starting autonumi script 2026")
    # get a connection to the FC
    try:
       mavLink = MavLinkHandler(MAV_COM,MAV_MSG_FREQ) 
    except Exception:
       logger.error("Unable to connect to FC")
       quit()