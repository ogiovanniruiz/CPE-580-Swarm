#!/usr/bin/python2.7

import rospy
from sensor_msgs.msg import Joy
import sys

from geometry_msgs.msg import Twist
from geometry_msgs.msg import Vector3


class TeleRobot:
    def __init__(self):
        # Initialize ROS
        rospy.init_node('control_node', anonymous=True)

        # Subscribe and Publish to Topics
        rospy.Subscriber("joy", Joy, self.array_callback)

        self.pub_twist = rospy.Publisher("/motor_control", Twist, queue_size=10, latch=True)

        # Initialize global variables and constants
        self.linear = 0
        self.angular = 0

        rospy.loginfo("Control Node is loaded...")
        self.r = rospy.Rate(10)
        self.updater()

    def array_callback(self, joy_array):
        self.linear = joy_array.axes[3]
        self.angular = -joy_array.axes[1]

    def updater(self):
        while not rospy.is_shutdown():
            self.pub_twist.publish(Twist(Vector3(self.linear, 0, 0), Vector3(0, 0, self.angular)))

            self.r.sleep()


def main(args):
    try:
        TeleRobot()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down control node.")


if __name__ == '__main__':
    main(sys.argv)
