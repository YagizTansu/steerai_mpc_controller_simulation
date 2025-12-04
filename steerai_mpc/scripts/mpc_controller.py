#!/usr/bin/env python3

import rospy
import numpy as np
import os
import sys
import matplotlib
matplotlib.use('Agg') # Set backend to non-interactive to prevent threading issues
import matplotlib.pyplot as plt
import datetime
from ackermann_msgs.msg import AckermannDrive
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from tf.transformations import euler_from_quaternion

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
        self.vehicle_model = VehicleModel(dt=self.dt)
        
        self.path_manager = PathManager(param_namespace='~path_manager')
        # Set initial target speed
        self.path_manager.set_target_speed(self.target_speed)
        
        # Prepare solver parameters
        solver_params = {
            'T': self.T,
            'dt': self.dt,
            'constraints': {
                'v_max': self.v_max,
                'delta_max': self.delta_max,
                'cte_max': self.cte_max
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
                'hessian_approximation': 'exact',
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
        self.last_completed_path_seq = -1
        self.last_cmd = np.array([0.0, 0.0])  # [cmd_v, cmd_steer]
        
        # Path tracking for goal detection
        self.current_path_seq = -1
        self.dist_traveled_on_path = 0.0
        self.current_path_length = 0.0
        
        # Data recording
        self.cte_history = []
        self.time_history = []
        self.start_time = None
        
        # CTE monitoring
        self.cte_violations = 0
        self.max_cte = 0.0
        
        rospy.loginfo("MPC Controller Initialized (Modular Version)")

    def load_parameters(self):
        """Load parameters from ROS parameter server."""
        param_ns = '~'
        
        # Vehicle
        self.v_max = rospy.get_param(param_ns + 'vehicle/v_max', 5.5)
        self.delta_max = rospy.get_param(param_ns + 'vehicle/delta_max', 0.6)
        self.cte_max = rospy.get_param(param_ns + 'vehicle/cte_max', 1.0)
                
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
        
        # Control
        self.target_speed = rospy.get_param(param_ns + 'control/target_speed', 5.556)
        self.goal_tolerance = rospy.get_param(param_ns + 'control/goal_tolerance', 0.5)
        self.loop_rate = rospy.get_param(param_ns + 'control/loop_rate', 10)

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
            dist = np.sqrt(dx**2 + dy**2)
            self.total_distance_traveled += dist
            self.dist_traveled_on_path += dist
        
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
        marker.header.frame_id = "base_footprint"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "robot_stats"
        marker.id = 1
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        # Fixed position in top-left of the map view
        marker.pose.position.x = -10.0
        marker.pose.position.y = 10.0
        marker.pose.position.z = 5.0
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.8
        marker.color = ColorRGBA(1.0, 1.0, 0.0, 1.0)
        marker.text = text_content
        
        self.stats_marker_pub.publish(marker)

    def plot_cte(self, seq_id):
        """Plots and saves the Cross Track Error over time."""
        if not self.cte_history:
            rospy.logwarn("No CTE data to plot.")
            return

        try:
            plt.figure(figsize=(10, 6))
            plt.plot(self.time_history, self.cte_history, label='Cross Track Error')
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            plt.axhline(y=self.cte_max, color='orange', linestyle='--', alpha=0.7, label=f'CTE Limit ({self.cte_max}m)')
            plt.axhline(y=-self.cte_max, color='orange', linestyle='--', alpha=0.7)
            plt.xlabel('Time (s)')
            plt.ylabel('CTE (m)')
            plt.title(f'Cross Track Error over Time (Path Seq: {seq_id})')
            plt.grid(True)
            plt.legend()

            # Save to steerai_mpc/plot_cte directory
            # Get the path to the steerai_mpc package (parent of scripts dir)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            package_dir = os.path.dirname(script_dir)
            save_dir = os.path.join(package_dir, 'plot_cte')

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"cte_plot_seq_{seq_id}_{timestamp}.png"
            save_path = os.path.join(save_dir, filename)
            
            plt.savefig(save_path)
            plt.close()
            
            # Log CTE statistics
            cte_array = np.array(self.cte_history)
            mean_cte = np.mean(np.abs(cte_array))
            std_cte = np.std(cte_array)
            max_cte = np.max(np.abs(cte_array))
            
            rospy.loginfo(f"CTE plot saved to {save_path}")
            rospy.loginfo(f"CTE Statistics - Mean: {mean_cte:.3f}m, Std: {std_cte:.3f}m, Max: {max_cte:.3f}m")
            rospy.loginfo(f"CTE Violations (>{self.cte_max}m): {self.cte_violations}")
        except Exception as e:
            rospy.logerr(f"Failed to plot CTE: {e}")

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

            # Detect new path
            current_path_seq = self.path_manager.get_path_seq()
            if current_path_seq != self.current_path_seq:
                self.current_path_seq = current_path_seq
                self.dist_traveled_on_path = 0.0
                self.current_path_length = self.path_manager.get_path_length()
                rospy.loginfo(f"New path received (Seq: {current_path_seq}), Length: {self.current_path_length:.2f}m")
            
            # Check if goal is reached            
            if self.path_manager.is_goal_reached(self.current_state[0], self.current_state[1], self.goal_tolerance):
                rospy.loginfo_throttle(1, f"🎯 Goal Reached! Stopping vehicle. (Seq: {current_path_seq})")
                
                # Plot CTE before resetting
                self.plot_cte(current_path_seq)
                self.cte_history = []
                self.time_history = []
                self.start_time = None
                self.cte_violations = 0
                self.max_cte = 0.0
                
                self.last_completed_path_seq = current_path_seq
                msg = AckermannDrive()
                msg.speed = 0.0
                msg.steering_angle = 0.0
                self.pub_cmd.publish(msg)
                rate.sleep()
                continue
            
            # Delay Compensation using Neural Network model for consistency
            # Use the last command sent to predict current state
            predicted_state = self.vehicle_model.predict_next_state_numpy(
                self.current_state, 
                self.current_yaw_rate,
                self.last_cmd
            )
            
            # Get Reference Trajectory (T+1 points)
            ref_traj = self.path_manager.get_reference(
                predicted_state[0], 
                predicted_state[1], 
                self.T + 1, 
                self.dt
            )
            
            # Solve MPC
            cmd_v, cmd_steer = self.solver.solve(predicted_state, ref_traj.T)
            
            # Store command for next delay compensation
            self.last_cmd = np.array([cmd_v, cmd_steer])
            
            # Publish Command
            msg = AckermannDrive()
            msg.speed = cmd_v
            msg.steering_angle = cmd_steer
            self.pub_cmd.publish(msg)
            
            # Debug Info
            cte = self.path_manager.get_cross_track_error(self.current_state[0], self.current_state[1])
            
            # Monitor CTE violations
            abs_cte = abs(cte)
            if abs_cte > self.cte_max:
                self.cte_violations += 1
                rospy.logwarn_throttle(2, f"⚠️  CTE VIOLATION: {abs_cte:.3f}m > {self.cte_max}m (Total: {self.cte_violations})")
            
            if abs_cte > self.max_cte:
                self.max_cte = abs_cte
            
            rospy.loginfo_throttle(1, f"MPC: v={cmd_v:.2f}, steer={cmd_steer:.2f}, CTE={cte:.3f}, MaxCTE={self.max_cte:.3f}")
            
            # Record Data
            if self.start_time is None:
                self.start_time = rospy.Time.now().to_sec()
            
            self.time_history.append(rospy.Time.now().to_sec() - self.start_time)
            self.cte_history.append(cte)
            
            # Publish stats
            self.publish_stats_marker(self.current_state[0], self.current_state[1], self.current_state[3])
            
            rate.sleep()

if __name__ == '__main__':
    try:
        controller = MPCController()
        controller.run()
    except rospy.ROSInterruptException:
        pass
