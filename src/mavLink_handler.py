from pymavlink import mavutil
import math
import time
from constants import*
from logger_handler import logger

class NotGuidedException(Exception):
    def __init__(self, message="Drone not in GUIDED mode"):
        self.message = message
        super().__init__(self.message)


class mavLink_handler:
    def __init__(self, connection_string,msg_frq=1): # msg freq in Hz
        self._last_pos = 0
        self._connection = self._connect(connection_string,msg_frq)
        
    def _check_guided_mode(self):
        mode = self.get_curr_mode()
        logger.debug(f"mode={mode}")
        if mode!=4:
            raise NotGuidedException
        else:
            return True 
        
    def _connect(self,connection_string,msg_frq):

        the_connection = mavutil.mavlink_connection(connection_string,source_system=1) 
        the_connection.wait_heartbeat()
        logger.info("Heartbeat from system (system %u component %u)" % (the_connection.target_system, the_connection.target_component))

        # ask to get all messages in a rate of msg_frq Hz
        data_rate = msg_frq # Hz
        the_connection.mav.request_data_stream_send(the_connection.target_system, the_connection.target_component,
        mavutil.mavlink.MAV_DATA_STREAM_ALL, data_rate, 1)
        return the_connection


    # wait for the next message of type. msg_type can be str or list of str. passing empty string will return the next message of any type
    # for list of message type in ardupilot: https://ardupilot.org/copter/docs/ArduCopter_MAVLink_Messages.html#requestable-messages
    # some common ones: ATTITUDE, RAW_IMU, HEARTBEAT, WIND, RC_CHANNELS
    def _get_message(self, msg_type:str):
        # msg = the_connection.recv_match(type='SYS_STATUS',blocking=True)
        if msg_type != "" and msg_type is not None:
            msg = self._connection.recv_match(type=msg_type,blocking=True)
        else:
            msg = self._connection.recv_match(blocking=True)
            logger.debug(msg.get_type())
        return msg        
    
    def move_servo(self, servo_channel, servo_angle):
        try:
            # ******* Send TEXT messages ********
            # Moves a servo connected to the ardupilot
            # aux_ch - use 7 for main_out 7, 8 for main_out 8
            # pwm - 900 = 0 degree, 2000 = 90 degree  
            logger.debug("requsting to move servo in pix")
            pwm_angle = int((2000 - 900) * (servo_angle / 90) + 900) 
            logger.debug(f"sending command move servo to pix with value: {pwm_angle}")
            self._connection.mav.command_long_send(self._connection.target_system, self._connection.target_component
            ,mavutil.mavlink.MAV_CMD_DO_SET_SERVO,0,servo_channel, pwm_angle, 0,0,0,0,0)
            # self.logger.debug("waiting to recev command ack from pix")
            # msg = self._connction.recv_match(type='COMMAND_ACK', blocking=True)
            # if msg.result != 0:
            #     self.logger.critical("Unable to move servo.")
            # else:
            #     self.logger.info("servo moved")
        except Exception as error:
            self.logger.exception("Failed to send move servo to pix")

    def send_text(self,msg):
        # ******* Send TEXT messages ********
        # how to send text that will show up on mission planner
        # use MAV_SEVERITY_WARNING. if you use MAV_SEVERITY_CRITICAL than it will not arm / takeoff. and MAV_SEVERITY_INFO will not always show up
        # can also use as b"Hello"
        self._connection.mav.statustext_send(mavutil.mavlink.MAV_SEVERITY_WARNING, bytes(msg,'utf-8'))
        logger.debug(msg)

    def change_flight_mode(self,mode:int):
        # ******* Change flight mode ********
        # Change flight mode. 0-Stabilize, 4-Guided, 5-Loiter, 6-RTL, 9-Land
        # mode values: https://ardupilot.org/copter/docs/parameters.html#fltmode1
        # see also https://ardupilot.org/dev/docs/mavlink-get-set-flightmode.html & 
        self._connection.mav.command_long_send(self._connection.target_system, self._connection.target_component,
                                                mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0, 1, mode, 0, 0, 0, 0, 0)
        msg = self._connection.recv_match(type='COMMAND_ACK',blocking=True)
        # print(msg)# Once connected, use 'the_connection' to get and send messages
        if msg.result != 0:
            self.send_text(f"Unable to change to mode {mode}")
            quit()
        else:
            self.send_text(f"Mode changed to {mode}")

    def arm(self):
        # ******* ARM ********
        self._connection.mav.command_long_send(self._connection.target_system, self._connection.target_component,
                                                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
        msg = self._connection.recv_match(type='COMMAND_ACK',blocking=True)
        # print(msg)# Once connected, use 'the_connection' to get and send messages
        if msg.result != 0:
            self.send_text("Unable to ARM. Exiting !!!")
            quit()
        else:
            self.send_text("ARMED !!!!")

    def get_curr_mode(self):
        msg=self._get_message("HEARTBEAT")
        while msg.type ==6: # 6 is type GCS, we want to ignore these
            msg=self._get_message("HEARTBEAT")
        return msg.custom_mode
        
    def get_curr_height(self):
        msg = self._connection.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
        if msg:
            if msg.get_type() == 'GLOBAL_POSITION_INT':
                alt_meter = msg.relative_alt / 1000
                self.send_text(f"Current height is {alt_meter} meters")
                return alt_meter
            else:
                self.send_text(f'didnt get the right message type')

    def takeoff(self,takeoff_alt_meters): 
        # ******* TAKEOFF ******** 
        self._connection.mav.command_long_send(self._connection.target_system, self._connection.target_component,
                                                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, takeoff_alt_meters)
        msg = self._connection.recv_match(type='COMMAND_ACK',blocking=True)
        # print(msg)# Once connected, use 'the_connection' to get and send messages
        if msg.result != 0:
            self.send_text("Unable to Take off")
        else:
            self.send_text("Taking off...")


    def fly_gps_pos(self,lat,lon):
        self._check_guided_mode()
        # ******* Fly to lat long ********
        # in mission planner in full prarameter list, update WPNAV_SPEED !! to make sure we don't fly too fast
        # see https://www.youtube.com/watch?v=yyt4VjBRG_Y
        # https://ardupilot.org/dev/docs/copter-commands-in-guided-mode.html
        # https://mavlink.io/en/messages/common.html#SET_POSITION_TARGET_LOCAL_NED
        self._connection.mav.send(mavutil.mavlink.MAVLink_set_position_target_global_int_message(10, self._connection.target_system,
                                self._connection.target_component, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, int(0b110111111000), int(lat * 10 ** 7), int(lon * 10 ** 7), 10, 0, 0, 0, 0, 0, 0, 1.57, 0.5))

        self.send_text("Flying to target...")
        # we notice we get a few zero wp_dist - so wait until we get non zero wp_dist 
        msg = self._connection.recv_match(type='NAV_CONTROLLER_OUTPUT', blocking=True)
        wp_dist = msg.wp_dist
        max_tries=0
        while max_tries<20 and wp_dist==0 and self._check_guided_mode():
            max_tries+=1
            # msg = the_connection.recv_match(type='LOCAL_POSITION_NED', blocking=True)
            msg = self._connection.recv_match(type='NAV_CONTROLLER_OUTPUT', blocking=True)
            wp_dist = msg.wp_dist
            # logger.debug(msg)
            self.send_text("distance to target:"+str(msg.wp_dist))

        # wait until we reach the wp (wp_dist=0)
        while wp_dist>0 and self._check_guided_mode():
            # msg = the_connection.recv_match(type='LOCAL_POSITION_NED', blocking=True)
            msg = self._connection.recv_match(type='NAV_CONTROLLER_OUTPUT', blocking=True)
            wp_dist = msg.wp_dist
            # logger.debug(msg)
            self.send_text("distance to target:"+str(msg.wp_dist))

        self.send_text("Target reached !")


    def fly_by_angles(self,roll_angle,pitch_angle,yaw_angle,duration,send_text=True):
        self._check_guided_mode()
        if send_text:
            self.send_text(f"ATT chnage r:{roll_angle},p:{pitch_angle}")
        # ******* Fly to using set attitude - doesnt requier GPS ********
        # e.g. to fly forward, pitch_angle=-5
        # see https://ardupilot.org/dev/docs/copter-commands-in-guided-mode.html#set-attitude-target
        # see https://github.com/dronekit/dronekit-python/blob/master/examples/set_attitude_target/set_attitude_target.py
        q = to_quaternion(roll_angle, pitch_angle, yaw_angle) # Quaternion
        # print(q)
        self._connection.mav.send(mavutil.mavlink.MAVLink_set_attitude_target_message(10, self._connection.target_system,
                                self._connection.target_component, int(0b00000111), q, 0,0,0,0.5))

        start = time.time()
        while time.time() - start < duration:
            #self.send_text("attitude for "+str(int(duration - (time.time() - start)))+" sec")
            self._connection.mav.send(mavutil.mavlink.MAVLink_set_attitude_target_message(10, self._connection.target_system,
                                    self._connection.target_component, int(0b00000111), q, 0,0,0,0.5))
            time.sleep(0.1)
        # Reset attitude, or it will persist for 1s more due to the timeout
        q = to_quaternion(0,0,0) # Quaternion
        self._connection.mav.send(mavutil.mavlink.MAVLink_set_attitude_target_message(10, self._connection.target_system,
                                self._connection.target_component, int(0b00000111), q, 0,0,0,0.5))



    def land(self):
        # ******* Land ********
        self._connection.mav.command_long_send(self._connection.target_system, self._connection.target_component,
                                                mavutil.mavlink.MAV_CMD_NAV_LAND , 0, 0, 0, 0, 0, 0, 0, 0)
        msg = self._connection.recv_match(type='COMMAND_ACK',blocking=True)
        # print(msg)# Once connected, use 'the_connection' to get and send messages
        if msg.result != 0:
            self.send_text("Unable to land")
        else:
            self.send_text("Landing...")
    
    def fly_to_height(self,height_meters):
        self._check_guided_mode()
        # ******* Fly to height ********
        # in mission planner in full prarameter list, update WPNAV_SPEED_UP & WPNAV_SPEED_DN !! to make sure we don't fly too fast
        # see https://www.youtube.com/watch?v=yyt4VjBRG_Y
        # https://ardupilot.org/dev/docs/copter-commands-in-guided-mode.html
        # https://mavlink.io/en/messages/common.html#SET_POSITION_TARGET_LOCAL_NED
        self.send_text("Changing height...")
        self._connection.mav.send(mavutil.mavlink.MAVLink_set_position_target_global_int_message(10, self._connection.target_system,
                                self._connection.target_component, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, int(0b0000111111000111), 0,0, 0, 0, 0, -0.5, 0, 0, 0, 0, 0))
        while self.get_curr_height() <= height_meters:
            self._connection.mav.send(mavutil.mavlink.MAVLink_set_position_target_global_int_message(10, self._connection.target_system,
                                self._connection.target_component, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, int(0b0000111111000111 ), 0,0, 0, 0, 0, -0.5, 0, 0, 0, 0, 0))
            self.send_text(f"Current height is {self.get_curr_height()} meters")
    def send_ned_velocity(self,velocity_x, velocity_y, velocity_z, duration):
        """
        Move vehicle in direction based on specified velocity vectors.
        NED frame: +x is North, +y is East, +z is Down.
        """
        msg = self._connection.mav.set_position_target_local_ned_encode(
            0,  # time_boot_ms
            self._connection.target_system,
            self._connection.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_OFFSET_NED,
            0b0000111111000111,  # Bitmask: only velocity components enabled
            0, 0, 0,             # x, y, z positions (not used)
            velocity_x, velocity_y, velocity_z,  # velocities in m/s
            0, 0, 0,             # acceleration (not used)
            0, 0
        )
        for _ in range(duration):
            self._connection.mav.send(msg)
            time.sleep(1)

def to_quaternion(roll = 0.0, pitch = 0.0, yaw = 0.0):
    """
    Convert degrees to quaternions
    """
    t0 = math.cos(math.radians(yaw * 0.5))
    t1 = math.sin(math.radians(yaw * 0.5))
    t2 = math.cos(math.radians(roll * 0.5))
    t3 = math.sin(math.radians(roll * 0.5))
    t4 = math.cos(math.radians(pitch * 0.5))
    t5 = math.sin(math.radians(pitch * 0.5))

    w = t0 * t2 * t4 + t1 * t3 * t5
    x = t0 * t3 * t4 - t1 * t2 * t5
    y = t0 * t2 * t5 + t1 * t3 * t4
    z = t1 * t2 * t4 - t0 * t3 * t5

    return [w, x, y, z]




