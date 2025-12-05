#!/usr/bin/env python3

# ================================================================
# File name: gem_sensor_info.py                                                                  
# Description: show sensor info in Rviz                                                              
# Author: Hang Cui
# Email: hangcui3@illinois.edu                                                                     
# Date created: 06/10/2021                                                                
# Date last modified: 07/02/2021                                                          
# Version: 0.1                                                                    
# Usage: ros2 run gem_gazebo gem_sensor_info.py                                                                     
# Python version: 3.8
# ================================================================

# Python Headers
import numpy as np
import math

# ROS Headers
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix, Imu
from nav_msgs.msg import Odometry
from std_msgs.msg import ColorRGBA
import tf_transformations

# Try to import OverlayText, if not available, we will just log to console
try:
    from jsk_rviz_plugins.msg import OverlayText
    OVERLAY_AVAILABLE = True
except ImportError:
    OVERLAY_AVAILABLE = False

class GEMOverlay(Node):

    def __init__(self):
        super().__init__('gem_sensor_info')
          
        if OVERLAY_AVAILABLE:
            self.sensor_info_pub = self.create_publisher(OverlayText, "/gem/sensor_info", 1)
            self.sensor_overlaytext = self.default_overlaytext_style()
        else:
            self.get_logger().warn("jsk_rviz_plugins not found. OverlayText will not be published. Info will be logged to console.")

        self.gps_sub     = self.create_subscription(NavSatFix, "/gem/gps/fix", self.gps_callback, 1)
        self.imu_sub     = self.create_subscription(Imu, "/gem/imu", self.imu_callback, 1)
        self.odom_sub    = self.create_subscription(Odometry, "/gem/base_footprint/odom", self.odom_callback, 1)
        
        self.timer = self.create_timer(0.1, self.timer_callback)

        self.lat         = 0.0
        self.lon         = 0.0
        self.alt         = 0.0
        self.imu_yaw     = 0.0

        self.x = 0.0
        self.y = 0.0
        self.x_dot = 0.0
        self.y_dot = 0.0
        self.gazebo_yaw = 0.0

    def gps_callback(self, msg):
        self.lat = msg.latitude
        self.lon = msg.longitude
        self.alt = msg.altitude

    def imu_callback(self, msg):
        orientation_q      = msg.orientation
        orientation_list   = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = tf_transformations.euler_from_quaternion(orientation_list)
        self.imu_yaw       = yaw

    def odom_callback(self, msg: Odometry):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = tf_transformations.euler_from_quaternion(orientation_list)
        self.gazebo_yaw = yaw
        self.x_dot = msg.twist.twist.linear.x
        self.y_dot = msg.twist.twist.linear.y

    def default_overlaytext_style(self):
        if not OVERLAY_AVAILABLE:
            return None
            
        text            = OverlayText()
        text.width      = 230
        text.height     = 290
        text.left       = 10
        text.top        = 10
        text.text_size  = 12
        text.line_width = 2
        text.font       = "DejaVu Sans Mono"
        text.fg_color   = ColorRGBA(r=25 / 255.0, g=1.0, b=240.0 / 255.0, a=1.0)
        text.bg_color   = ColorRGBA(r=0.0, g=0.0, b=0.0, a=0.2)
        return text

    def timer_callback(self):
        f_vel = np.sqrt(self.x_dot**2 + self.y_dot**2)
        
        info_text = """----------------------
                         Sensor (Measurement):
                         Lat = %.6f
                         Lon = %.6f
                         Alt = %.6f
                         Yaw = %.6f
                         ----------------------
                         Gazebo (Ground Truth):
                         X_pos = %.3f
                         Y_pos = %.3f
                         X_dot = %.3f
                         Y_dot = %.3f
                         F_vel = %.3f
                         Yaw   = %.3f
                         ----------------------                               
                      """ % (self.lat, self.lon, self.alt, self.imu_yaw,
                             self.x, self.y, self.x_dot, self.y_dot, f_vel,
                             self.gazebo_yaw)

        if OVERLAY_AVAILABLE:
            self.sensor_overlaytext.text = info_text
            self.sensor_info_pub.publish(self.sensor_overlaytext)
        else:
            # Log to console periodically (maybe not every 0.1s to avoid spam)
            # self.get_logger().info(info_text)
            pass


def main(args=None):
    rclpy.init(args=args)
    gem_overlay = GEMOverlay()
    try:
        rclpy.spin(gem_overlay)
    except KeyboardInterrupt:
        pass
    finally:
        gem_overlay.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
