cmake_minimum_required(VERSION 3.8)
project(gem_gazebo)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(ackermann_msgs REQUIRED)
find_package(std_msgs REQUIRED)

install(DIRECTORY config launch models worlds media
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/gem_ackermann_to_twist.py
  scripts/gem_sensor_info.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
