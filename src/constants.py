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

#video out params
OUTPUT_DIR = "recordings"
SEGMENT_DURATION = -1  # Duration of each video segment in seconds. set -1 to disable video output

# Detect params
FRAMES_CENTERED = 10 # Number of consecutive frames the target must be centered to consider the task complete

# fire params
FIRE_DURATION = 20  # Duration of the firing mechanism activation in seconds
FIRE_ALTITUDE = 3.0  # meters, target altitude for hold_altitude
VELOCITY_Z = -0.5  # m/s, vertical velocity to maintain during firing (positive for down, negative for up)
DRONE_MOVE_ANGLE = 2