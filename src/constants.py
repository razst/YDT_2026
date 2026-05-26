import logging

# MAVLINK params
#MAV_COM="/dev/ttyACM0" 
#MAV_COM="COM5" 
MAV_COM="tcp:localhost:5763" # SITL com
MAV_MSG_FREQ = 1 # how ofthen do we want to get messages from mavlink (in Hz)
IS_HEADLESS = False
# General params
LOGGING_LEVEL = logging.DEBUG  # You can set to INFO, WARNING, ERROR etc.

# Camera params
CAMERA_IDX=0

# Flight params
HOLD_ALTITUDE = 3.0  # meters, target altitude for hold_altitude

#video out params
OUTPUT_DIR = "recordings"
SEGMENT_DURATION = -1  # Duration of each video segment in seconds. set -1 to disable video output