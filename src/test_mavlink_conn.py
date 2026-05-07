from pymavlink import mavutil
'''
If you aren't sure which port is the Pixhawk, the most foolproof way to identify it is using the dmesg command:

Unplug the USB cable from the Pi.

Run this command: dmesg -w (This follows the system log in real-time).

Plug the Pixhawk back in.

Look for the last few lines. You should see something like:
cdc_acm 1-1.2:1.0: ttyACM0: USB ACM device
'''
# Replace 'ttyACM0' with the port you found
connection_string = '/dev/ttyACM0' #/dev/serial0 or /dev/ttyAMA0 (over TX/RX GPIO) or /ttyACM0 (over USB cable)
baud_rate = 115200 # or 921600 on new FC like H7

print("Start...")
connection = mavutil.mavlink_connection(connection_string, baud=baud_rate)

print("Waiting for heartbeat...")
connection.wait_heartbeat()
print(f"Heartbeat received from system (ID {connection.target_system})")
print("End")
