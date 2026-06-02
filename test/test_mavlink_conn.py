from pymavlink import mavutil
import time

# Create the connection
# /dev/serial0 is the hardware UART on RPi
# Baudrate 57600 is standard for ArduPilot Telem ports
# If you are using a USB connection to the Pixhawk, it might show up as /dev/ttyACM0 or /dev/ttyUSB0 instead, and you may need to adjust the baudrate (often 115200 or 921600 for newer flight controllers).
# H7 over GPIO = '/dev/serial0', baud=921600
connection = mavutil.mavlink_connection('/dev/serial0', baud=921600)

# Wait for the first heartbeat to ensure connection is alive
print("Waiting for heartbeat...")
connection.wait_heartbeat()
print("Heartbeat from system is: (system %u component %u)" % 
      (connection.target_system, connection.target_component))

# Request data (e.g., Attitude)
while True:
    msg = connection.recv_match(type='ATTITUDE', blocking=True)
    if msg:
        print(f"Pitch: {msg.pitch}, Roll: {msg.roll}, Yaw: {msg.yaw}")
    
    time.sleep(0.1)