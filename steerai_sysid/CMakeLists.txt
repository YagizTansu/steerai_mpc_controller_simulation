cmake_minimum_required(VERSION 3.0.2)
project(steerai_sysid)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES steerai_sysid
#  CATKIN_DEPENDS rospy std_msgs
#  DEPENDS system_lib
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/train_dynamics.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
