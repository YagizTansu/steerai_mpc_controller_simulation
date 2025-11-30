#!/usr/bin/env python3

import rospy
import math
import csv
import time
import os
import rospkg
import numpy as np
import random
from ackermann_msgs.msg import AckermannDrive
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion

class ManeuverManager:
    def __init__(self):
        self.maneuvers = []
        self.current_maneuver_idx = 0
        self.start_time = 0.0
        
    def add_maneuver(self, name, duration, func):
        self.maneuvers.append({
            'name': name,
            'duration': duration,
            'func': func
        })
        
    def get_command(self, t):
        if self.current_maneuver_idx >= len(self.maneuvers):
            return 0.0, 0.0, "DONE"
            
        maneuver = self.maneuvers[self.current_maneuver_idx]
        
        # Check if maneuver is finished
        if t - self.start_time > maneuver['duration']:
            self.current_maneuver_idx += 1
            self.start_time = t
            if self.current_maneuver_idx >= len(self.maneuvers):
                return 0.0, 0.0, "DONE"
            maneuver = self.maneuvers[self.current_maneuver_idx]
            
        # Execute maneuver
        local_t = t - self.start_time
        speed, steering = maneuver['func'](local_t, maneuver['duration'])
        return speed, steering, maneuver['name']

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
        timestamp_str = time.strftime("%Y%m%d-%H%M%S")
        self.log_file_path = os.path.join(data_dir, f'training_data_{timestamp_str}.csv')
        self.log_frequency = 50.0 # Hz
        self.max_speed = 5.5 # m/s
        self.max_steering = 0.6 # rad
        
        # State variables
        self.curr_x = 0.0
        self.curr_y = 0.0
        self.curr_yaw = 0.0
        self.curr_speed = 0.0
        self.curr_yaw_rate = 0.0
        
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
        self.csv_writer.writerow(['timestamp', 'dt', 'maneuver_type', 'cmd_speed', 'cmd_steering_angle', 'curr_x', 'curr_y', 'curr_yaw', 'curr_speed', 'curr_yaw_rate'])
        
        # Maneuver Manager
        self.maneuver_manager = ManeuverManager()
        self.setup_maneuvers()
        
        # Safety shutdown
        rospy.on_shutdown(self.shutdown_hook)
        
        rospy.loginfo("Data Collector Node Started. Logging to %s", self.log_file_path)

    def setup_maneuvers(self):
        # 1. Warmup (Constant Speed)
        self.maneuver_manager.add_maneuver("WARMUP", 5.0, lambda t, d: (2.0, 0.0))
        
        # 2. Chirp Signal (Steering) - Low Speed
        # Frequency sweeps from 0.1 Hz to 1.0 Hz
        def chirp_steering_low(t, d):
            f0 = 0.1
            f1 = 1.0
            k = (f1 - f0) / d
            freq = f0 + k * t
            steering = 0.4 * math.sin(2 * math.pi * freq * t)
            return 2.5, steering
        self.maneuver_manager.add_maneuver("CHIRP_LOW_SPEED", 20.0, chirp_steering_low)
        
        # 3. Ramp Speed (Acceleration)
        def ramp_speed_up(t, d):
            speed = 1.0 + (4.5 / d) * t
            return speed, 0.0
        self.maneuver_manager.add_maneuver("RAMP_UP", 10.0, ramp_speed_up)
        
        # 3.5. Ramp Speed Down (Braking)
        def ramp_speed_down(t, d):
            speed = 5.5 - (4.5 / d) * t
            return speed, 0.0
        self.maneuver_manager.add_maneuver("RAMP_DOWN", 10.0, ramp_speed_down)
        
        # 4. Chirp Signal (Steering) - High Speed
        def chirp_steering_high(t, d):
            f0 = 0.1
            f1 = 0.8 # Lower max freq at high speed for safety
            k = (f1 - f0) / d
            freq = f0 + k * t
            steering = 0.2 * math.sin(2 * math.pi * freq * t) # Lower amplitude
            return 4.5, steering
        self.maneuver_manager.add_maneuver("CHIRP_HIGH_SPEED", 15.0, chirp_steering_high)
        
        # 5. Step Inputs (Steering)
        def step_steering(t, d):
            # Switch every 2 seconds
            stage = int(t / 2.0)
            val = 0.3 if stage % 2 == 0 else -0.3
            return 3.0, val
        self.maneuver_manager.add_maneuver("STEP_STEERING", 10.0, step_steering)
        
        # 6. Random Walk (Exploration)
        def random_walk(t, d):
            # Randomly perturb speed and steering
            # Note: This is pseudo-random, deterministic for a given run if seeded, 
            # but here we use time-based noise
            noise_steer = random.uniform(-0.5, 0.5)
            noise_speed = random.uniform(1.5, 4.5)
            return noise_speed, noise_steer
        self.maneuver_manager.add_maneuver("RANDOM_WALK", 20.0, random_walk)
        
        # 7. Cooldown
        self.maneuver_manager.add_maneuver("COOLDOWN", 5.0, lambda t, d: (0.0, 0.0))

    def odom_callback(self, msg):
        self.curr_x = msg.pose.pose.position.x
        self.curr_y = msg.pose.pose.position.y
        
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        self.curr_yaw = yaw
        
        self.curr_speed = math.sqrt(msg.twist.twist.linear.x**2 + msg.twist.twist.linear.y**2)
        self.curr_yaw_rate = msg.twist.twist.angular.z

    def run(self):
        rate = rospy.Rate(self.log_frequency)
        last_time = rospy.Time.now().to_sec()
        
        while not rospy.is_shutdown():
            current_time = rospy.Time.now().to_sec()
            dt = current_time - last_time
            elapsed_time = current_time - self.start_time
            
            # Get Command
            target_speed, target_steering, maneuver_name = self.maneuver_manager.get_command(elapsed_time)
            
            if maneuver_name == "DONE":
                rospy.loginfo("All maneuvers completed. Stopping...")
                break
                
            rospy.loginfo_throttle(1.0, f"Maneuver: {maneuver_name} | T: {elapsed_time:.1f}s")
            
            # Safety Constraints
            # 1. Lateral Acceleration Limit (approx a_lat = v^2 * tan(delta) / L)
            # If a_lat > 2.0 m/s^2, reduce speed
            L = 1.75
            a_lat = (target_speed**2) * math.tan(abs(target_steering)) / L
            if a_lat > 2.5:
                target_speed *= 0.8 # Reduce speed by 20%
                rospy.logwarn_throttle(0.5, "Safety Trigger: High Lateral Accel! Reducing speed.")
            
            # 2. Hard Limits
            self.cmd_speed = max(min(target_speed, self.max_speed), -self.max_speed)
            self.cmd_steering = max(min(target_steering, self.max_steering), -self.max_steering)
            
            # Publish command
            ack_msg = AckermannDrive()
            ack_msg.speed = self.cmd_speed
            ack_msg.steering_angle = self.cmd_steering
            self.pub_cmd.publish(ack_msg)
            
            # Log data
            self.csv_writer.writerow([
                f"{current_time:.4f}", 
                f"{dt:.4f}",
                maneuver_name,
                f"{self.cmd_speed:.4f}", 
                f"{self.cmd_steering:.4f}", 
                f"{self.curr_x:.4f}", 
                f"{self.curr_y:.4f}", 
                f"{self.curr_yaw:.4f}", 
                f"{self.curr_speed:.4f}",
                f"{self.curr_yaw_rate:.4f}"
            ])
            
            last_time = current_time
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
