cmake_minimum_required(VERSION 3.0.2)
project(steerai_localization)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  robot_localization
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES steerai_localization
#  CATKIN_DEPENDS roscpp rospy robot_localization
#  DEPENDS system_lib
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
