#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
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

# Add current directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vehicle_model import VehicleModel
from mpc_solver import MPCSolver
from path_manager import PathManager

class MPCController(Node):
    def __init__(self):
        super().__init__('mpc_controller')
        
        self.logger = self.get_logger()
        
        # Load parameters
        self.load_parameters()
        
        # Initialize Modules
        self.vehicle_model = VehicleModel(dt=self.dt)
        
        self.path_manager = PathManager(self, param_namespace='path_manager')
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
        qos = QoSProfile(depth=1)
        self.pub_cmd = self.create_publisher(AckermannDrive, '/gem/ackermann_cmd', qos)
        self.sub_odom = self.create_subscription(Odometry, '/gem/base_footprint/odom', self.odom_callback, 10)
        self.stats_marker_pub = self.create_publisher(Marker, '/gem/stats_text', qos)
        
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
        
        self.logger.info("MPC Controller Initialized (Modular Version)")
        
        # Timer for control loop
        self.timer = self.create_timer(1.0 / self.loop_rate, self.control_loop)

    def load_parameters(self):
        """Load parameters from ROS parameter server."""
        def get_param(name, default):
            if not self.has_parameter(name):
                self.declare_parameter(name, default)
            return self.get_parameter(name).value

        param_ns = '' # Parameters are usually node-local or namespaced
        
        # Vehicle
        self.v_max = get_param('vehicle.v_max', 5.5)
        self.delta_max = get_param('vehicle.delta_max', 0.6)
        self.cte_max = get_param('vehicle.cte_max', 1.0)
                
        # MPC
        self.T = get_param('mpc.horizon', 20)
        self.dt = get_param('mpc.dt', 0.1)
        
        # Weights
        self.weight_position = get_param('weights.position', 4.0)
        self.weight_heading = get_param('weights.heading', 15.0)
        self.weight_velocity = get_param('weights.velocity', 1.0)
        self.weight_steering_smooth = get_param('weights.steering_smooth', 5.0)
        self.weight_acceleration_smooth = get_param('weights.acceleration_smooth', 0.1)
        
        # Solver
        self.solver_max_iter = get_param('solver.max_iter', 900)
        self.solver_print_level = get_param('solver.print_level', 0)
        self.solver_tol = get_param('solver.tol', 1e-2)
        self.solver_acceptable_tol = get_param('solver.acceptable_tol', 2e-1)
        self.solver_acceptable_iter = get_param('solver.acceptable_iter', 5)
        self.solver_max_cpu_time = get_param('solver.max_cpu_time', 0.09)
        
        # Control
        self.target_speed = get_param('control.target_speed', 5.556)
        self.goal_tolerance = get_param('control.goal_tolerance', 0.5)
        self.loop_rate = get_param('control.loop_rate', 10)

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
        marker.header.stamp = self.get_clock().now().to_msg()
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
        marker.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)
        marker.text = text_content
        
        self.stats_marker_pub.publish(marker)

    def plot_cte(self, seq_id):
        """Plots and saves the Cross Track Error over time."""
        if not self.cte_history:
            self.logger.warn("No CTE data to plot.")
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
            
            self.logger.info(f"CTE plot saved to {save_path}")
            self.logger.info(f"CTE Statistics - Mean: {mean_cte:.3f}m, Std: {std_cte:.3f}m, Max: {max_cte:.3f}m")
            self.logger.info(f"CTE Violations (>{self.cte_max}m): {self.cte_violations}")
        except Exception as e:
            self.logger.error(f"Failed to plot CTE: {e}")

    def control_loop(self):
        if self.current_state is None:
            # Throttle warning
            # self.logger.warn("MPC Controller: No Odom received yet.") 
            # Manual throttle
            return
        
        # Check if path is available
        if self.path_manager.path_data is None:
            # self.logger.warn("MPC Controller: Waiting for path on /gem/raw_path...")
            return

        # Detect new path
        current_path_seq = self.path_manager.get_path_seq()
        if current_path_seq != self.current_path_seq:
            self.current_path_seq = current_path_seq
            self.dist_traveled_on_path = 0.0
            self.current_path_length = self.path_manager.get_path_length()
            self.logger.info(f"New path received (Seq: {current_path_seq}), Length: {self.current_path_length:.2f}m")
        
        # Check if goal is reached            
        if self.path_manager.is_goal_reached(self.current_state[0], self.current_state[1], self.goal_tolerance):
            # Throttle info
            # self.logger.info(f"🎯 Goal Reached! Stopping vehicle. (Seq: {current_path_seq})")
            
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
            return
        
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
            # self.logger.warn(f"⚠️  CTE VIOLATION: {abs_cte:.3f}m > {self.cte_max}m (Total: {self.cte_violations})")
        
        if abs_cte > self.max_cte:
            self.max_cte = abs_cte
        
        # self.logger.info(f"MPC: v={cmd_v:.2f}, steer={cmd_steer:.2f}, CTE={cte:.3f}, MaxCTE={self.max_cte:.3f}")
        
        # Record Data
        if self.start_time is None:
            self.start_time = self.get_clock().now().nanoseconds / 1e9
        
        self.time_history.append((self.get_clock().now().nanoseconds / 1e9) - self.start_time)
        self.cte_history.append(cte)
        
        # Publish stats
        self.publish_stats_marker(self.current_state[0], self.current_state[1], self.current_state[3])

def main(args=None):
    rclpy.init(args=args)
    controller = MPCController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
