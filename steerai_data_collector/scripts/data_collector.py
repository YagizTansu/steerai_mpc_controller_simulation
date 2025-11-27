#!/usr/bin/env python3

import rospy
import math
import csv
import time
import os
import rospkg
from ackermann_msgs.msg import AckermannDrive
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion

class DataCollector:
    def __init__(self):
        rospy.init_node('data_collector', anonymous=True)
        
        # Get package path and create data directory
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('steerai_data_collector')
        data_dir = os.path.join(package_path, 'data')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        # Parameters
        self.log_file_path = os.path.join(data_dir, 'training_data.csv')
        self.log_frequency = 100.0 # Hz
        self.max_speed = 5.5 # m/s (~20 km/h)
        self.max_steering = 0.6 # rad
        self.duration = 240.0 # seconds
        
        # State variables
        self.curr_x = 0.0
        self.curr_y = 0.0
        self.curr_yaw = 0.0
        self.curr_speed = 0.0
        
        # Control variables
        self.cmd_speed = 0.0
        self.cmd_steering = 0.0
        self.start_time = rospy.Time.now().to_sec()
        
        # Setup Publishers and Subscribers
        self.pub_cmd = rospy.Publisher('/gem/ackermann_cmd', AckermannDrive, queue_size=1)
        self.sub_odom = rospy.Subscriber('/gem/base_footprint/odom', Odometry, self.odom_callback)
        
        # Setup CSV logging
        self.csv_file = open(self.log_file_path, 'w')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['timestamp', 'cmd_speed', 'cmd_steering_angle', 'curr_x', 'curr_y', 'curr_yaw', 'curr_speed'])
        
        # Safety shutdown
        rospy.on_shutdown(self.shutdown_hook)
        
        rospy.loginfo("Data Collector Node Started. Logging to %s", self.log_file_path)

    def odom_callback(self, msg):
        self.curr_x = msg.pose.pose.position.x
        self.curr_y = msg.pose.pose.position.y
        
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        self.curr_yaw = yaw
        
        # Calculate speed from twist
        self.curr_speed = math.sqrt(msg.twist.twist.linear.x**2 + msg.twist.twist.linear.y**2)

    def get_excitation_commands(self, t):
        # Persistent Excitation Strategy
        
        # Varying sinusoidal steering: A * sin(omega * t)
        # Vary frequency slightly over time
        omega = 0.5 + 0.2 * math.sin(0.1 * t)
        amplitude = 0.5 # rad
        steering = amplitude * math.sin(omega * t)
        
        # Velocity: Step inputs and ramps
        # Simple logic: Switch between constant speed and ramping every 10 seconds
        period = 20.0
        phase = t % period
        
        if phase < 5.0:
            # Constant low speed
            speed = 2.0
        elif phase < 10.0:
            # Ramp up
            speed = 2.0 + (phase - 5.0) * 0.5 # Reach 4.5 m/s
        elif phase < 15.0:
            # Constant high speed
            speed = 4.5
        else:
            # Ramp down
            speed = 4.5 - (phase - 15.0) * 0.5 # Back to 2.0 m/s
            
        return speed, steering

    def run(self):
        rate = rospy.Rate(self.log_frequency)
        
        while not rospy.is_shutdown():
            current_time = rospy.Time.now().to_sec()
            elapsed_time = current_time - self.start_time
            
            if elapsed_time > self.duration:
                rospy.loginfo("Duration limit reached. Stopping...")
                break
            
            remaining_time = self.duration - elapsed_time
            rospy.loginfo_throttle(1.0, "Time remaining: %.1f s", remaining_time)
            
            # Calculate commands
            target_speed, target_steering = self.get_excitation_commands(elapsed_time)
            
            # Apply constraints
            self.cmd_speed = max(min(target_speed, self.max_speed), -self.max_speed)
            self.cmd_steering = max(min(target_steering, self.max_steering), -self.max_steering)
            
            # Publish command
            ack_msg = AckermannDrive()
            ack_msg.speed = self.cmd_speed
            ack_msg.steering_angle = self.cmd_steering
            self.pub_cmd.publish(ack_msg)
            
            # Log data
            self.csv_writer.writerow([current_time, self.cmd_speed, self.cmd_steering, self.curr_x, self.curr_y, self.curr_yaw, self.curr_speed])
            
            rate.sleep()

    def shutdown_hook(self):
        rospy.loginfo("Shutting down Data Collector. Stopping vehicle...")
        ack_msg = AckermannDrive()
        ack_msg.speed = 0.0
        ack_msg.steering_angle = 0.0
        self.pub_cmd.publish(ack_msg)
        self.csv_file.close()

if __name__ == '__main__':
    try:
        node = DataCollector()
        node.run()
    except rospy.ROSInterruptException:
        pass
