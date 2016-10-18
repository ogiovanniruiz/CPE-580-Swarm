cmake_minimum_required(VERSION 2.8.3)
project(joy_bot)

find_package(catkin REQUIRED COMPONENTS
  rospy
  joy
  sensor_msgs
  std_msgs
  message_generation
)

catkin_package(
 CATKIN_DEPENDS rospy sensor_msgs std_msgs message_runtime joy
 DEPENDS system_lib
)


install(DIRECTORY launch
DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS scripts/control.py
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

catkin_install_python(PROGRAMS scripts/serial_node.py
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
 
 
 
