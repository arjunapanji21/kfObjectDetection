'''
    File name         : arjuna.py
    Description       : Object Detection And Kalman Filter
    Author            : Arjuna Panji Prakarsa
    Date created      : 17/01/2021
    Python Version    : 2.7
'''

#Import Python Libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import datetime
from dynamixel_sdk import *
from ax12_control_table import *

def nothing(x):
    #print(x)
    pass

window_name = 'Object Detection By Arjuna Panji'
cv2.namedWindow(window_name)
cv2.createTrackbar('L_H', window_name, 0, 255, nothing)
cv2.createTrackbar('L_S', window_name, 0, 255, nothing)
cv2.createTrackbar('L_V', window_name, 0, 255, nothing)
cv2.createTrackbar('U_H', window_name, 0, 255, nothing)
cv2.createTrackbar('U_S', window_name, 0, 255, nothing)
cv2.createTrackbar('U_V', window_name, 0, 255, nothing)
cv2.createTrackbar('Threshold', window_name, 0, 255, nothing)

cv2.setTrackbarPos('L_H', window_name, 0)
cv2.setTrackbarPos('L_S', window_name, 70)
cv2.setTrackbarPos('L_V', window_name, 225)
cv2.setTrackbarPos('U_H', window_name, 35)
cv2.setTrackbarPos('U_S', window_name, 255)
cv2.setTrackbarPos('U_V', window_name, 255)
cv2.setTrackbarPos('Threshold', window_name, 127)

def detect(frame, debugMode):

    font = cv2.FONT_HERSHEY_SIMPLEX
    date = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    cv2.putText(frame, date, (10, 20,), font, 0.5, (0,240,240), 2, cv2.LINE_AA)

    l_h = cv2.getTrackbarPos('L_H', window_name)
    l_s = cv2.getTrackbarPos('L_S', window_name)
    l_v = cv2.getTrackbarPos('L_V', window_name)
    u_h = cv2.getTrackbarPos('U_H', window_name)
    u_s = cv2.getTrackbarPos('U_S', window_name)
    u_v = cv2.getTrackbarPos('U_V', window_name)
    th = cv2.getTrackbarPos('Threshold', window_name)

    lowerBall = np.array([l_h, l_s, l_v])
    upperBall = np.array([u_h, u_s, u_v])

    # Convert frame from BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Blur the frame
    blur = cv2.medianBlur(hsv, 5)

    # Create a mask from blurred frame
    mask = cv2.inRange(blur, lowerBall, upperBall)

    # Convert to black and white image
    _, thresh = cv2.threshold(mask, th, 255, 0)

    # Refine the image using morphological transformation
    kernal = np.ones((5,5), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernal, iterations=2)

    # Find contours
    _, contours, _ = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x:cv2.contourArea(x), reverse=True)

    centers=[]

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            (x,y,w,h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x,y),(x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, "Object", (x,y), font, 0.5, (0,255,0),2)
            cv2.putText(frame, "X: " + str(x) + "Y: " + str(y), (520, 20), font, 0.5, (0,0,255),2)
            cenX = (x+x+w) / 2
            cenY = (y+y+h) / 2
            centers.append(np.array([[cenX], [cenY]]))
        #print(area)

    if (debugMode):
        cv2.imshow(window_name, morph)

    return centers

def jarak(j):
    switcher = {
        0:200,1:199,2:198,3:197,4:196,5:195,6:194,7:193,8:192,9:191,10:190,
        11:189,12:188,13:187,14:186,15:185,16:184,17:183,18:182,19:181,20:180,
        21:179,22:178,23:177,24:176,25:175,26:174,27:173,28:172,29:171,30:170,
        31:169,32:168,33:167,34:166,35:165,36:164,37:163,38:162,39:161,40:160,
        41:159,42:158,43:157,44:156,45:155,46:154,47:153,48:152,49:151,50:150,
        51:149,52:148,53:147,54:146,55:145,56:144,57:143,58:142,59:141,60:140,
        61:139,62:138,63:137,64:136,65:135,66:134,67:133,68:132,69:131,70:130,
        71:129,72:128,73:127,74:126,75:125,76:124,77:123,78:122,79:121,80:120,
        81:119,82:118,83:117,84:116,85:115,86:114,87:113,88:112,89:111,90:110,
        91:109,92:108,93:107,94:106,95:105,96:104,97:103,98:102,99:101,100:100,
        101:99,102:98,103:97,104:96,105:95,106:94,107:93,108:92,109:91,110:90,
        111:89,112:88,113:87,114:86,115:85,116:84,117:83,118:82,119:81,120:80,
        121:79,122:78,123:77,124:76,125:75,126:74,127:73,128:72,129:71,130:70,
        131:69,132:68,133:67,134:66,135:65,136:64,137:63,138:62,139:61,140:60,
        141:59,142:58,143:57,144:56,145:55,146:54,147:53,148:52,149:51,150:50,
        151:49,152:48,153:47,154:46,155:45,156:44,157:43,158:42,159:41,160:40,
        161:39,162:38,163:37,164:36,165:35,166:34,167:33,168:32,169:31,170:30,
        171:29,172:28,173:27,174:26,175:25,176:24,177:23,178:22,179:21,180:20,
        181:19,182:18,183:17,184:16,185:15,186:14,187:13,188:12,189:11,190:10,
        191:9,192:8,193:7,194:6,195:5,196:4,197:3,198:2,199:1,200:0
    }
    return switcher.get(j,"---")

class KalmanFilter:

    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    def Estimate(self, coordX, coordY):
        ''' This function estimates the position of the object'''
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        return predicted

class Ax12:
    """ Class for Dynamixel AX12A motors."""
    PROTOCOL_VERSION = 1.0
    BAUDRATE = 1000000             # Dynamixel default baudrate
    DEVICENAME = '/dev/ttyUSB0'           # Default COM Port
    portHandler = PortHandler(DEVICENAME)   # Initialize Ax12.PortHandler instance
    packetHandler = PacketHandler(PROTOCOL_VERSION)  # Initialize Ax12.PacketHandler instance
    # Dynamixel will rotate between this value
    MIN_POS_VAL = 0
    MAX_POS_VAL = 1023

    @classmethod
    def open_port(cls):
        if cls.portHandler.openPort():
            print("Succeeded to open the port")
        else:
            print("Failed to open the port")
            print("Press any key to terminate...")
            quit()

    @classmethod
    def set_baudrate(cls):
        if cls.portHandler.setBaudRate(cls.BAUDRATE):
            print("Succeeded to change the baudrate")
        else:
            print("Failed to change the baudrate")
            print("Press any key to terminate...")
            quit()

    @classmethod
    def close_port(cls):
        # Close port
        cls.portHandler.closePort()
        print('Successfully closed port')

    def __init__(self, motor_id):
        """Initialize motor id"""
        self.id = motor_id

    def set_register1(self, reg_num, reg_value):
        dxl_comm_result, dxl_error = Ax12.packetHandler.write1ByteTxRx(
            Ax12.portHandler, self.id, reg_num, reg_value)
        Ax12.check_error(dxl_comm_result, dxl_error)

    def set_register2(self, reg_num, reg_value):
        dxl_comm_result, dxl_error = Ax12.packetHandler.write2ByteTxRx(
            Ax12.portHandler, self.id, reg_num, reg_value)
        Ax12.check_error(dxl_comm_result, dxl_error)

    def get_register1(self, reg_num):
        reg_data, dxl_comm_result, dxl_error = Ax12.packetHandler.read1ByteTxRx(
            Ax12.portHandler, self.id, reg_num)
        Ax12.check_error(dxl_comm_result, dxl_error)
        return reg_data

    def get_register2(self, reg_num_low):
        reg_data, dxl_comm_result, dxl_error = Ax12.packetHandler.read2ByteTxRx(
            Ax12.portHandler, self.id, reg_num_low)
        Ax12.check_error(dxl_comm_result, dxl_error)
        return reg_data

    def enable_torque(self):
        """Enable torque for motor."""
        self.set_register1(ADDR_AX_TORQUE_ENABLE, TORQUE_ENABLE)
        print(self.get_register1(ADDR_AX_TORQUE_ENABLE))
        # print("Torque has been successfully enabled for dxl ID: %d" % self.id)

    def disable_torque(self):
        """Disable torque."""
        self.set_register1(ADDR_AX_TORQUE_ENABLE, TORQUE_DISABLE)
        print(self.get_register1(ADDR_AX_TORQUE_ENABLE))
        # print("Torque has been successfully disabled for dxl ID: %d" % self.id)

    def set_position(self, dxl_goal_position):
        """Write goal position."""
        self.set_register2(ADDR_AX_GOAL_POSITION_L, dxl_goal_position)
        print("Position of dxl ID: %d set to %d " %
              (self.id, dxl_goal_position))

    def set_moving_speed(self, dxl_goal_speed):
        """Set the moving speed to goal position [0-1023]."""
        self.set_register2(ADDR_AX_GOAL_SPEED_L, dxl_goal_speed)
        print("Moving speed of dxl ID: %d set to %d " %
              (self.id, dxl_goal_speed))

    def get_position(self):
        """Read present position."""
        dxl_present_position = self.get_register2(ADDR_AX_PRESENT_POSITION_L)
        print("ID:%03d  PresPos:%03d" % (self.id, dxl_present_position))
        return dxl_present_position

    def get_present_speed(self):
        """Returns the current speed of the motor."""
        present_speed = self.get_register2(ADDR_AX_PRESENT_SPEED_L)
        return present_speed

    def get_moving_speed(self):
        """Returns moving speed to goal position [0-1023]."""
        moving_speed = self.get_register2(ADDR_AX_GOAL_SPEED_L)
        return moving_speed

    def led_on(self):
        """Turn on Motor Led."""
        self.set_register1(ADDR_AX_LED, True)

    def led_off(self):
        """Turn off Motor Led."""
        self.set_register1(ADDR_AX_LED, False)

    def get_load(self):
        """Returns current load on motor."""
        dxl_load = self.get_register2(ADDR_AX_PRESENT_LOAD_L)
        # CCW 0-1023 # CW 1024-2047
        return dxl_load

    def get_temperature(self):
        """Returns internal temperature in units of Celsius."""
        dxl_temperature = self.get_register2(ADDR_AX_PRESENT_TEMPERATURE)
        return dxl_temperature

    def get_voltage(self):
        """Returns current voltage supplied to Motor in units of Volts."""
        dxl_voltage = (self.get_register1(ADDR_AX_PRESENT_VOLTAGE))/10
        return dxl_voltage

    def set_torque_limit(self, torque_limit):
        """Sets Torque Limit of Motor."""
        self.set_register2(ADDR_AX_TORQUE_LIMIT_L, torque_limit)

    def get_torque_limit(self):
        """Returns current Torque Limit of Motor."""
        dxl_torque_limit = self.get_register2(ADDR_AX_TORQUE_LIMIT_L)
        return dxl_torque_limit

    def is_moving(self):
        """Checks to see if motor is still moving to goal position."""
        dxl_motion = self.get_register1(ADDR_AX_MOVING)
        return dxl_motion

    @staticmethod
    def check_error(comm_result, dxl_err):
        if comm_result != COMM_SUCCESS:
            print("%s" % Ax12.packetHandler.getTxRxResult(comm_result))
        elif dxl_err != 0:
            print("%s" % Ax12.packetHandler.getRxPacketError(dxl_err))