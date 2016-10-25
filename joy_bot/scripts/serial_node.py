#!/usr/bin/python2.7

import rospy
from geometry_msgs.msg import Twist
import serial
import sys

import time

# ser = serial.Serial('/dev/rfcomm1', 9600, serial.EIGHTBITS, serial.PARITY_NONE, serial.STOPBITS_ONE)
ser = serial.Serial('/dev/ttyACM0', 9600, serial.EIGHTBITS, serial.PARITY_NONE, serial.STOPBITS_ONE)
time.sleep(3)


class SerialRobot:
    def __init__(self):
        # Initialize ROS
        rospy.init_node('serial_node', anonymous=True)

        # Subscribe and Publish to Topics
        rospy.Subscriber("/robo_twist", Twist, self.array_callback)

        # Initialize global variables and constants
        self.left = 0
        self.right = 0

        # self.string = ''
        rospy.loginfo("Serial Node is loaded...")
        self.r = rospy.Rate(10)
        self.updater()

    def array_callback(self, twist_array):
        self.left = chr(int(twist_array.linear.x) + 66)
        self.right = chr(int(twist_array.angular.z) + 66)

    def updater(self):
        while not rospy.is_shutdown():
            header = 72
            values = bytearray([header, self.left, self.right])
            ser.write(values)

            self.r.sleep()


def main(args):
    try:
        SerialRobot()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down serial node.")


if __name__ == '__main__':
    main(sys.argv)
