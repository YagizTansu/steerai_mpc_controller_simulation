cmake_minimum_required(VERSION 3.0.2)
project(steerai_data_collector)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  ackermann_msgs
  nav_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES steerai_data_collector
#  CATKIN_DEPENDS rospy std_msgs
#  DEPENDS system_lib
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/data_collector.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
