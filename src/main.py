from logger_handler import logger
from mavLink_handler import *
from  constants import *


if __name__ == "__main__":
    global mavLink
    logger.info("Starting autonumi script")
    # get a connection to the PIX
    try:
       mavLink = mavLink_handler(MAV_COM,MAV_MSG_FREQ) 
    except Exception:
       logger.error("Unable to connect to PIX")
       quit()