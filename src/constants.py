import logging


#PIX_COM="/dev/ttyACM0" 
#PIX_COM="COM5" 
MAV_COM="tcp:localhost:5763" # SITL com
MAV_MSG_FREQ = 1 # how ofthen do we want to get messages from mavlink (in Hz)

LOGGING_LEVEL = logging.DEBUG  # You can set to INFO, WARNING, ERROR etc.
