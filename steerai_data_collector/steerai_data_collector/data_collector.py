#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
import math
import csv
import time
import os
from ament_index_python.packages import get_package_share_directory
import numpy as np
import random
from ackermann_msgs.msg import AckermannDrive
from nav_msgs.msg import Odometry
try:
    from tf_transformations import euler_from_quaternion
except ImportError:
    # Fallback
    def euler_from_quaternion(quat):
        x = quat[0]
        y = quat[1]
        z = quat[2]
        w = quat[3]
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        sinp = 2 * (w * y - z * x)
        if np.abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw

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

class DataCollector(Node):
    def __init__(self):
        super().__init__('data_collector')
        
        self.logger = self.get_logger()
        
        # Get package path and create data directory
        try:
            package_path = get_package_share_directory('steerai_data_collector')
            # For development, we might want to save to the source directory or a specific location
            # But for now, let's save to the share directory or user home to avoid permission issues
            # Or better, use the current working directory if running from source, but that's unreliable
            # Let's use a safe path in home
            data_dir = os.path.expanduser('~/steerai_data/data')
        except Exception as e:
            self.logger.warn(f"Could not find package share directory: {e}")
            data_dir = os.path.expanduser('~/steerai_data/data')

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
        self.start_time = self.get_clock().now().nanoseconds / 1e9
        self.last_time = self.start_time
        
        # Setup Publishers and Subscribers
        qos = QoSProfile(depth=1)
        self.pub_cmd = self.create_publisher(AckermannDrive, '/gem/ackermann_cmd', qos)
        self.sub_odom = self.create_subscription(Odometry, '/gem/base_footprint/odom', self.odom_callback, 10)
        
        # Setup CSV logging
        self.csv_file = open(self.log_file_path, 'w')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['timestamp', 'dt', 'maneuver_type', 'cmd_speed', 'cmd_steering_angle', 'curr_x', 'curr_y', 'curr_yaw', 'curr_speed', 'curr_yaw_rate'])
        
        # Maneuver Manager
        self.maneuver_manager = ManeuverManager()
        self.setup_maneuvers()
        
        self.logger.info(f"Data Collector Node Started. Logging to {self.log_file_path}")
        
        # Timer for control loop
        self.timer = self.create_timer(1.0 / self.log_frequency, self.control_loop)

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
        
        # 6. Random Walk (Exploration) - AGGRESSIVE
        # Increased duration and amplitude to cover more state space
        def random_walk(t, d):
            # Random speed between 1.5 and 5.0 m/s
            # Change target every 2 seconds
            seed_v = int(t / 2.0)
            np.random.seed(seed_v)
            target_v = 1.5 + np.random.rand() * 3.5
            
            # Random steering between -0.5 and 0.5 rad
            # Change target every 0.5 seconds (more frequent)
            seed_s = int(t / 0.5)
            np.random.seed(seed_s)
            target_s = -0.5 + np.random.rand() * 1.0
            
            # Add high frequency noise
            noise_s = np.random.normal(0, 0.05)
            
            return target_v, target_s + noise_s
            
        self.maneuver_manager.add_maneuver("RANDOM_WALK", 40.0, random_walk)
        
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

    def control_loop(self):
        current_time = self.get_clock().now().nanoseconds / 1e9
        dt = current_time - self.last_time
        elapsed_time = current_time - self.start_time
        
        # Get Command
        target_speed, target_steering, maneuver_name = self.maneuver_manager.get_command(elapsed_time)
        
        if maneuver_name == "DONE":
            self.logger.info("All maneuvers completed. Stopping...")
            self.shutdown_hook()
            self.destroy_node()
            rclpy.shutdown()
            return
            
        # Log throttle manually since loginfo_throttle is not directly available in same way
        # Or use logging filter, but simple counter or time check is easier
        # self.logger.info(f"Maneuver: {maneuver_name} | T: {elapsed_time:.1f}s")
        
        # Safety Constraints
        # 1. Lateral Acceleration Limit (approx a_lat = v^2 * tan(delta) / L)
        # If a_lat > 2.0 m/s^2, reduce speed
        L = 1.75
        a_lat = (target_speed**2) * math.tan(abs(target_steering)) / L
        if a_lat > 2.5:
            target_speed *= 0.8 # Reduce speed by 20%
            self.logger.warn("Safety Trigger: High Lateral Accel! Reducing speed.", throttle_duration_sec=0.5)
        
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
        
        self.last_time = current_time

    def shutdown_hook(self):
        self.logger.info("Shutting down Data Collector. Stopping vehicle...")
        ack_msg = AckermannDrive()
        ack_msg.speed = 0.0
        ack_msg.steering_angle = 0.0
        self.pub_cmd.publish(ack_msg)
        self.csv_file.close()

def main(args=None):
    rclpy.init(args=args)
    node = DataCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown_hook()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
