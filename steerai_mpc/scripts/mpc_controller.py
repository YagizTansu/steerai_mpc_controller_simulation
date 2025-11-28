#!/usr/bin/env python3

import rospy
import numpy as np
import os
import sys
from ackermann_msgs.msg import AckermannDrive
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from tf.transformations import euler_from_quaternion
from dynamic_reconfigure.server import Server
from steerai_mpc.cfg import MPCDynamicParamsConfig

# Add current directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vehicle_model import VehicleModel
from mpc_solver import MPCSolver
from path_manager import PathManager

class MPCController:
    def __init__(self):
        rospy.init_node('mpc_controller')
        
        # Load parameters
        self.load_parameters()
        
        # Initialize Modules
        self.vehicle_model = VehicleModel(
            dt=self.dt, 
            L=self.L, 
            v_low=self.v_low, 
            v_high=self.v_high
        )
        
        self.path_manager = PathManager(param_namespace='~path_manager')
        # Set initial target speed
        self.path_manager.set_target_speed(self.target_speed)
        
        # Prepare solver parameters
        solver_params = {
            'T': self.T,
            'dt': self.dt,
            'constraints': {
                'v_max': self.v_max,
                'delta_max': self.delta_max
            },
            'weights': {
                'position': self.weight_position,
                'heading': self.weight_heading,
                'velocity': self.weight_velocity,
                'steering_smooth': self.weight_steering_smooth,
                'acceleration_smooth': self.weight_acceleration_smooth
            },
            'target_speed': self.target_speed,
            'solver_opts': {
                'max_iter': self.solver_max_iter,
                'print_level': self.solver_print_level,
                'tol': self.solver_tol,
                'acceptable_tol': self.solver_acceptable_tol,
                'acceptable_iter': self.solver_acceptable_iter,
                'max_cpu_time': self.solver_max_cpu_time,
                'hessian_approximation': 'limited-memory',
                'limited_memory_max_history': 6,
            }
        }
        
        self.solver = MPCSolver(self.vehicle_model, solver_params)
        
        # ROS Setup
        self.pub_cmd = rospy.Publisher('/gem/ackermann_cmd', AckermannDrive, queue_size=1)
        self.sub_odom = rospy.Subscriber('/gem/base_footprint/odom', Odometry, self.odom_callback)
        self.stats_marker_pub = rospy.Publisher('/gem/stats_text', Marker, queue_size=1)
        
        # State
        self.current_state = None # [x, y, yaw, v]
        self.current_yaw_rate = 0.0
        self.total_distance_traveled = 0.0
        self.prev_position = None
        
        # Dynamic Reconfigure
        self.dyn_reconfig_srv = Server(MPCDynamicParamsConfig, self.dynamic_reconfigure_callback)
        
        rospy.loginfo("MPC Controller Initialized (Modular Version)")

    def load_parameters(self):
        """Load parameters from ROS parameter server."""
        param_ns = '~'
        
        # Vehicle
        self.v_max = rospy.get_param(param_ns + 'vehicle/v_max', 5.5)
        self.delta_max = rospy.get_param(param_ns + 'vehicle/delta_max', 0.6)
        self.L = rospy.get_param(param_ns + 'vehicle/wheelbase', 1.75)
        
        # MPC
        self.T = rospy.get_param(param_ns + 'mpc/horizon', 20)
        self.dt = rospy.get_param(param_ns + 'mpc/dt', 0.1)
        
        # Weights
        self.weight_position = rospy.get_param(param_ns + 'weights/position', 4.0)
        self.weight_heading = rospy.get_param(param_ns + 'weights/heading', 15.0)
        self.weight_velocity = rospy.get_param(param_ns + 'weights/velocity', 1.0)
        self.weight_steering_smooth = rospy.get_param(param_ns + 'weights/steering_smooth', 5.0)
        self.weight_acceleration_smooth = rospy.get_param(param_ns + 'weights/acceleration_smooth', 0.1)
        
        # Solver
        self.solver_max_iter = rospy.get_param(param_ns + 'solver/max_iter', 900)
        self.solver_print_level = rospy.get_param(param_ns + 'solver/print_level', 0)
        self.solver_tol = rospy.get_param(param_ns + 'solver/tol', 1e-2)
        self.solver_acceptable_tol = rospy.get_param(param_ns + 'solver/acceptable_tol', 2e-1)
        self.solver_acceptable_iter = rospy.get_param(param_ns + 'solver/acceptable_iter', 5)
        self.solver_max_cpu_time = rospy.get_param(param_ns + 'solver/max_cpu_time', 0.09)
        
        # Hybrid
        self.v_low = rospy.get_param(param_ns + 'hybrid/v_low', 0.5)
        self.v_high = rospy.get_param(param_ns + 'hybrid/v_high', 2.0)
        
        # Control
        self.target_speed = rospy.get_param(param_ns + 'control/target_speed', 5.556)
        self.goal_tolerance = rospy.get_param(param_ns + 'control/goal_tolerance', 0.5)
        self.loop_rate = rospy.get_param(param_ns + 'control/loop_rate', 10)

    def dynamic_reconfigure_callback(self, config, level):
        """Callback for dynamic reconfigure."""
        rospy.loginfo("Dynamic reconfigure request received")
        
        # Update weights in solver
        new_weights = {
            'position': config.weight_position,
            'heading': config.weight_heading,
            'velocity': config.weight_velocity,
            'steering_smooth': config.weight_steering_smooth,
            'acceleration_smooth': config.weight_acceleration_smooth
        }
        
        if hasattr(self, 'solver'):
            self.solver.update_weights(new_weights)
        
        # Update target speed
        old_speed = self.target_speed
        self.target_speed = config.target_speed
        if abs(old_speed - config.target_speed) > 0.01:
            rospy.loginfo(f"Target speed updated: {old_speed:.2f} -> {config.target_speed:.2f} m/s")
            # Update PathManager
            if hasattr(self, 'path_manager'):
                self.path_manager.set_target_speed(self.target_speed)
        
        self.goal_tolerance = config.goal_tolerance
        
        return config

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        
        v = np.sqrt(msg.twist.twist.linear.x**2 + msg.twist.twist.linear.y**2)
        self.current_yaw_rate = msg.twist.twist.angular.z
        
        self.current_state = np.array([x, y, yaw, v])
        
        # Update total distance
        if self.prev_position is not None:
            dx = x - self.prev_position[0]
            dy = y - self.prev_position[1]
            self.total_distance_traveled += np.sqrt(dx**2 + dy**2)
        
        self.prev_position = np.array([x, y])

    def publish_stats_marker(self, robot_x, robot_y, robot_v):
        """Publishes text markers showing current speed and distance."""
        speed_ms = robot_v
        speed_kmh = robot_v * 3.6
        
        remaining_dist = self.path_manager.calculate_remaining_distance(robot_x, robot_y)
        completed_dist = self.total_distance_traveled
        
        text_content = (
            f"Speed: {speed_ms:.2f} m/s ({speed_kmh:.2f} km/h)\n"
            f"Remaining Distance: {remaining_dist:.2f} m\n"
            f"Completed Distance: {completed_dist:.2f} m"
        )
        
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "robot_stats"
        marker.id = 1
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = robot_x
        marker.pose.position.y = robot_y + 0.5
        marker.pose.position.z = 2.0
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.5
        marker.color = ColorRGBA(1.0, 1.0, 1.0, 1.0)
        marker.text = text_content
        
        self.stats_marker_pub.publish(marker)

    def run(self):
        rate = rospy.Rate(self.loop_rate)
        
        rospy.loginfo("MPC Controller: Waiting for state...")
        while not rospy.is_shutdown():
            if self.current_state is None:
                rospy.logwarn_throttle(2, "MPC Controller: No Odom received yet.")
                rate.sleep()
                continue
            
            # Check if path is available
            if self.path_manager.path_data is None:
                rospy.logwarn_throttle(5, "MPC Controller: Waiting for path on /gem/raw_path...")
                rate.sleep()
                continue

            # Check goal
            if self.path_manager.is_goal_reached(self.current_state[0], self.current_state[1], self.goal_tolerance):
                rospy.loginfo_throttle(1, "🎯 Goal Reached! Stopping vehicle.")
                msg = AckermannDrive()
                msg.speed = 0.0
                msg.steering_angle = 0.0
                self.pub_cmd.publish(msg)
                rate.sleep()
                continue
            
            # Delay Compensation
            delay_dt = self.dt
            x0, y0, yaw0, v0 = self.current_state
            yaw_rate = self.current_yaw_rate
            
            x_pred = x0 + v0 * np.cos(yaw0) * delay_dt
            y_pred = y0 + v0 * np.sin(yaw0) * delay_dt
            yaw_pred = yaw0 + yaw_rate * delay_dt
            v_pred = v0
            
            predicted_state = np.array([x_pred, y_pred, yaw_pred, v_pred])
            
            # Get Reference Trajectory (T+1 points)
            ref_traj = self.path_manager.get_reference(
                predicted_state[0], 
                predicted_state[1], 
                self.T + 1, 
                self.dt
            )
            
            # Transpose to match shape (4, T+1) for solver
            ref_traj = ref_traj.T
            
            # Solve MPC
            cmd_v, cmd_steer, success = self.solver.solve(predicted_state, ref_traj)
            
            # Publish Command
            msg = AckermannDrive()
            msg.speed = cmd_v
            msg.steering_angle = cmd_steer
            self.pub_cmd.publish(msg)
            
            # Debug Info
            cte = self.path_manager.get_cross_track_error(self.current_state[0], self.current_state[1])
            rospy.loginfo_throttle(1, f"MPC: v={cmd_v:.2f}, steer={cmd_steer:.2f}, CTE={cte:.3f}")
            
            # Publish stats
            self.publish_stats_marker(self.current_state[0], self.current_state[1], self.current_state[3])
            
            rate.sleep()

if __name__ == '__main__':
    try:
        controller = MPCController()
        controller.run()
    except rospy.ROSInterruptException:
        pass
