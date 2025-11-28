cmake_minimum_required(VERSION 3.0.2)
project(steerai_mpc)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  ackermann_msgs
  nav_msgs
  dynamic_reconfigure
)

# Generate dynamic reconfigure options
generate_dynamic_reconfigure_options(
  cfg/MPCDynamicParams.cfg
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES steerai_mpc
#  CATKIN_DEPENDS rospy std_msgs
#  DEPENDS system_lib
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

# Install executable scripts
catkin_install_python(PROGRAMS
  scripts/mpc_controller.py
  scripts/tf_broadcaster.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install Python modules (for import)
install(FILES
  scripts/path_manager.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install config files
install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
  FILES_MATCHING PATTERN "*.yaml"
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  FILES_MATCHING PATTERN "*.launch"
)

