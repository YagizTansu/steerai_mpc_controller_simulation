#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from ackermann_msgs.msg import AckermannDrive

class AckermannToTwist(Node):
    def __init__(self):
        super().__init__('ackermann_to_twist')
        
        self.wheelbase = 1.75 # meters
        
        self.sub = self.create_subscription(
            AckermannDrive,
            '/gem/ackermann_cmd',
            self.callback,
            1)
            
        self.pub = self.create_publisher(
            Twist,
            '/cmd_vel', # Topic expected by ros_gz_bridge -> Gazebo
            1)
            
        self.get_logger().info("Ackermann to Twist Converter Started")

    def callback(self, msg):
        twist = Twist()
        twist.linear.x = msg.speed
        
        # omega = v * tan(delta) / L
        if abs(msg.speed) > 0.001:
            twist.angular.z = msg.speed * math.tan(msg.steering_angle) / self.wheelbase
        else:
            twist.angular.z = 0.0
            
        self.pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = AckermannToTwist()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
