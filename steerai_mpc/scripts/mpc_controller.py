#!/usr/bin/env python3

import rospy
import casadi as ca
import torch
import numpy as np
import os
import rospkg
import joblib
from scipy.spatial import KDTree
from ackermann_msgs.msg import AckermannDrive
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from tf.transformations import euler_from_quaternion
import sys

# Add current directory to path so we can import path_manager
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from path_manager import PathManager

class MPCController:
    def __init__(self):
        rospy.init_node('mpc_controller')
        
        # Load Model and Scalers
        self.load_model()
        
        # Constraints
        self.v_max = 5.5
        self.delta_max = 0.6
        self.L = 1.75 # Wheelbase

        # MPC Parameters
        self.T = 20 # Horizon (2 seconds lookahead - balance between turn prediction and solver speed)
        self.dt = 0.1 # Time step
        
        # MPC Cost Function Weights (Ağırlıklar)
        # Bu ağırlıklar MPC'nin hangi hatalara ne kadar önem verdiğini belirler
        
        # Pozisyon Hatası Ağırlığı: Aracın referans yoldan ne kadar saptığını cezalandırır
        # Yüksek değer = Yola daha sıkı yapışır, ama dönüşlerde ani manevralara yol açabilir
        # AZALTILDI: 10.0 → 5.0 (Dönüşlerde biraz sapmasına tolerans göster)
        self.weight_position = 4.0
        
        # Yönelim (Yaw) Hatası Ağırlığı: Aracın başının referans yöne ne kadar baktığını kontrol eder
        # Yüksek değer = Araç her zaman yol yönüne bakar, ama düz yolda salınıma (oscillation) yol açabilir
        # ARTIRILDI: 10.0 → 15.0 (Dönüşlerde doğru yöne bakması pozisyondan daha önemli)
        self.weight_heading = 15.0
        
        # Hız Hatası Ağırlığı: Aracın hedef hıza ne kadar uyduğunu kontrol eder
        # Düşük değer = Hız kontrolü esnek, ama hedef hıza ulaşması zor olabilir
        self.weight_velocity = 1.0
        
        # Direksiyon Yumuşatma Ağırlığı: Ardışık direksiyon komutları arasındaki farkı cezalandırır
        # Yüksek değer = Daha yumuşak direksiyon, ama dönüşlerde yavaş tepki verebilir
        # ARTIRILDI: 1.0 → 10.0 (Ani direksiyon değişimlerini ÇOK DAHA FAZLA cezalandır!)
        self.weight_steering_smooth = 5.0
        
        # Gaz/Fren Yumuşatma Ağırlığı: Ardışık hız komutları arasındaki farkı cezalandırır
        # Yüksek değer = Daha yumuşak ivmelenme/yavaşlama
        self.weight_acceleration_smooth = 0.1
        
        # Solver Parameters (IPOPT)
        # Bu parametreler solver'ın hızını ve kalitesini kontrol eder
        self.solver_max_iter = 900              # Maksimum iterasyon sayısı
        self.solver_print_level = 0             # Log seviyesi (0 = sessiz)
        self.solver_tol = 1e-2                  # Optimality toleransı (daha gevşek = daha hızlı)
        self.solver_acceptable_tol = 2e-1       # Kabul edilebilir tolerans (daha gevşek)
        self.solver_acceptable_iter = 5         # Kabul edilebilir iterasyon sayısı
        self.solver_max_cpu_time = 0.09         # Maksimum CPU süresi (saniye)
        
        # Initialize Path Manager
        # Assuming the CSV file is in the data folder of steerai_data_collector
        rospack = rospkg.RosPack()
        path_file = os.path.join(rospack.get_path('steerai_data_collector'), 'data', 'reference_path.csv')
        self.path_manager = PathManager(path_file)
        
        # Get path data and build KDTree for fast nearest neighbor search
        self.path_data = self.path_manager.get_path_data()  # [x, y, yaw, v]
        if self.path_data is not None:
            self.tree = KDTree(self.path_data[:, :2])  # Build KDTree on x,y coordinates
        else:
            self.tree = None
            rospy.logerr("Failed to load path data!")
        
        # Setup CasADi Solver
        self.setup_solver()
        
        # ROS Setup
        self.pub_cmd = rospy.Publisher('/gem/ackermann_cmd', AckermannDrive, queue_size=1)
        self.sub_odom = rospy.Subscriber('/gem/base_footprint/odom', Odometry, self.odom_callback)
        self.target_marker_pub = rospy.Publisher('/gem/target_point', Marker, queue_size=1)
        
        # State
        self.current_state = None # [x, y, yaw, v]
        
        rospy.loginfo("MPC Controller Initialized")

    def load_model(self):
        rospack = rospkg.RosPack()
        sysid_path = rospack.get_path('steerai_sysid')
        
        # Load Scalers
        self.scaler_X = joblib.load(os.path.join(sysid_path, 'scaler_X.pkl'))
        self.scaler_y = joblib.load(os.path.join(sysid_path, 'scaler_y.pkl'))
        
        # Load PyTorch Model Weights
        model_path = os.path.join(sysid_path, 'dynamics_model.pth')
        state_dict = torch.load(model_path, weights_only=True)
        
        # Extract weights as numpy arrays
        self.W1 = state_dict['fc1.weight'].cpu().numpy()
        self.b1 = state_dict['fc1.bias'].cpu().numpy()
        self.W2 = state_dict['fc2.weight'].cpu().numpy()
        self.b2 = state_dict['fc2.bias'].cpu().numpy()
        self.W3 = state_dict['fc3.weight'].cpu().numpy()
        self.b3 = state_dict['fc3.bias'].cpu().numpy()
        
        # Scaler parameters for CasADi
        self.mean_X = self.scaler_X.mean_
        self.scale_X = self.scaler_X.scale_
        self.mean_y = self.scaler_y.mean_
        self.scale_y = self.scaler_y.scale_

    def neural_net_dynamics(self, v, cmd_v, cmd_steer):
        # Input: [v, cmd_v, cmd_steer]
        # Normalize Input
        inp = ca.vertcat(v, cmd_v, cmd_steer)
        inp_norm = (inp - self.mean_X) / self.scale_X
        
        # Forward Pass (ReLU activation)
        h1 = ca.mtimes(self.W1, inp_norm) + self.b1
        h1 = ca.fmax(0, h1) # ReLU
        
        h2 = ca.mtimes(self.W2, h1) + self.b2
        h2 = ca.fmax(0, h2) # ReLU
        
        out_norm = ca.mtimes(self.W3, h2) + self.b3
        
        # Denormalize Output: [next_v, delta_yaw]
        out = out_norm * self.scale_y + self.mean_y
        
        return out[0], out[1] # next_v, delta_yaw

    def setup_solver(self):
        # Optimization Variables
        # State: [x, y, yaw, v]
        # Control: [cmd_v, cmd_steer]
        
        self.opti = ca.Opti()
        
        # Decision variables for horizon T
        self.X = self.opti.variable(4, self.T + 1) # State trajectory
        self.U = self.opti.variable(2, self.T)     # Control trajectory
        
        # Parameters
        self.P = self.opti.parameter(4) # Initial State
        self.Ref = self.opti.parameter(4, self.T + 1) # Reference Trajectory [x, y, yaw, v]
        
        # Cost Function
        obj = 0
        for k in range(self.T):
            # State Error
            # Minimize distance to reference point
            x_err = self.X[0, k] - self.Ref[0, k]
            y_err = self.X[1, k] - self.Ref[1, k]
            yaw_err = self.X[2, k] - self.Ref[2, k]
            v_err = self.X[3, k] - self.Ref[3, k]
            
            # Normalize yaw error to [-pi, pi]
            # Note: CasADi doesn't have a direct atan2 for difference, but for small errors it's fine.
            # Or we can just penalize sin/cos differences if needed.
            # For now, simple difference is usually okay if the reference is close.
            
            obj += self.weight_position * (x_err**2 + y_err**2) # Position Error
            # Yaw Error handling wrapping
            # Minimize 1 - cos(yaw_err) which behaves like yaw_err^2/2 for small errors
            # but handles wrapping correctly.
            # Original weight 5.0 * err^2 is roughly 10.0 * (1 - cos(err))
            obj += self.weight_heading * (1 - ca.cos(yaw_err)) # Heading Error
            obj += self.weight_velocity * v_err**2               # Speed Error
            
            # Control Effort
            if k > 0:
                obj += self.weight_steering_smooth * (self.U[1, k] - self.U[1, k-1])**2 # Smooth steering
                obj += self.weight_acceleration_smooth * (self.U[0, k] - self.U[0, k-1])**2 # Smooth acceleration
                
        self.opti.minimize(obj)
        
        # Constraints
        for k in range(self.T):
            # Dynamics Constraints
            curr_x = self.X[0, k]
            curr_y = self.X[1, k]
            curr_yaw = self.X[2, k]
            curr_v = self.X[3, k]
            
            cmd_v = self.U[0, k]
            cmd_steer = self.U[1, k]
            
            # Neural Net Prediction
            next_v_nn, delta_yaw_nn = self.neural_net_dynamics(curr_v, cmd_v, cmd_steer)
            
            # Kinematic Prediction
            # Simple kinematic model for low speeds
            next_v_kin = cmd_v # Assume we track command at low speed
            delta_yaw_kin = (curr_v / self.L * ca.tan(cmd_steer)) * self.dt
            
            # Hybrid Blending
            # alpha = 0 for v < 0.5 (Pure Kinematic)
            # alpha = 1 for v > 2.0 (Pure NN)
            v_low = 0.5
            v_high = 2.0
            alpha = ca.fmin(1.0, ca.fmax(0.0, (curr_v - v_low) / (v_high - v_low)))
            
            next_v_pred = (1 - alpha) * next_v_kin + alpha * next_v_nn
            delta_yaw_pred = (1 - alpha) * delta_yaw_kin + alpha * delta_yaw_nn
            
            # Kinematic Update
            next_x = curr_x + curr_v * ca.cos(curr_yaw) * self.dt
            next_y = curr_y + curr_v * ca.sin(curr_yaw) * self.dt
            next_yaw = curr_yaw + delta_yaw_pred
            next_v = next_v_pred
            
            self.opti.subject_to(self.X[0, k+1] == next_x)
            self.opti.subject_to(self.X[1, k+1] == next_y)
            self.opti.subject_to(self.X[2, k+1] == next_yaw)
            self.opti.subject_to(self.X[3, k+1] == next_v)
            
            # Input Constraints
            self.opti.subject_to(self.opti.bounded(-self.v_max, self.U[0, k], self.v_max)) # -v_max <= cmd_v <= v_max
            self.opti.subject_to(self.opti.bounded(-self.delta_max, self.U[1, k], self.delta_max))
            
        # Initial Condition Constraint
        self.opti.subject_to(self.X[:, 0] == self.P)
        
        # Solver Options
        p_opts = {'expand': True}
        s_opts = {
            'max_iter': self.solver_max_iter,
            'print_level': self.solver_print_level,
            'tol': self.solver_tol,
            'acceptable_tol': self.solver_acceptable_tol,
            'acceptable_iter': self.solver_acceptable_iter,
            'max_cpu_time': self.solver_max_cpu_time
        }
        self.opti.solver('ipopt', p_opts, s_opts)
        
        # Warm start variables
        self.prev_X = np.zeros((4, self.T + 1))
        self.prev_U = np.zeros((2, self.T))

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        
        v = np.sqrt(msg.twist.twist.linear.x**2 + msg.twist.twist.linear.y**2)
        
        self.current_state = np.array([x, y, yaw, v])
    
    def get_reference(self, robot_x, robot_y, horizon_size):
        """
        Finds the nearest point and returns the next N points.
        :param robot_x: Robot X position
        :param robot_y: Robot Y position
        :param horizon_size: Number of points to return
        :return: Numpy array of shape (horizon_size, 4) -> [x, y, yaw, v]
        """
        if self.tree is None:
            return np.zeros((horizon_size, 4))

        # Query KDTree for nearest neighbor
        dist, idx = self.tree.query([robot_x, robot_y])
        
        # Get slice
        end_idx = idx + horizon_size
        
        if end_idx < len(self.path_data):
            ref_path = self.path_data[idx:end_idx]
        else:
            # We are near the end, take what's left and pad
            ref_path = self.path_data[idx:]
            rows_missing = horizon_size - len(ref_path)
            if rows_missing > 0:
                # Pad with the last point
                last_point = ref_path[-1]
                padding = np.tile(last_point, (rows_missing, 1))
                ref_path = np.vstack((ref_path, padding))
        
        # Visualize the target point (the first point in the reference)
        self.publish_target_marker(ref_path[0])
        
        return ref_path

    def get_cross_track_error(self, robot_x, robot_y):
        """
        Calculates perpendicular distance to the nearest path point.
        """
        if self.tree is None:
            return 0.0
            
        dist, idx = self.tree.query([robot_x, robot_y])
        
        # To determine sign (left or right), we can use the cross product
        # Vector from path point to robot
        path_pt = self.path_data[idx]
        dx = robot_x - path_pt[0]
        dy = robot_y - path_pt[1]
        
        # Path tangent (yaw)
        yaw = path_pt[2]
        
        cross_prod = np.cos(yaw) * dy - np.sin(yaw) * dx

        return cross_prod # Signed distance
    
    def is_goal_reached(self, robot_x, robot_y, tolerance=0.5):
        """
        Check if robot has reached the final goal.
        :param robot_x: Robot X position
        :param robot_y: Robot Y position
        :param tolerance: Distance threshold in meters (default: 0.5m)
        :return: True if goal reached, False otherwise
        """
        if self.path_data is None or len(self.path_data) == 0:
            return False
        
        # Get final point
        final_point = self.path_data[-1, :2]
        
        # Calculate distance to final point
        dist = np.sqrt((robot_x - final_point[0])**2 + (robot_y - final_point[1])**2)
        
        return dist < tolerance
    
    def publish_target_marker(self, target_pt):
        """
        Publishes a marker for the current target point in RViz.
        """
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "target_point"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = target_pt[0]
        marker.pose.position.y = target_pt[1]
        marker.pose.position.z = 0.5  # Slightly elevated
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        marker.color = ColorRGBA(1.0, 0.0, 0.0, 1.0)  # Red
        
        self.target_marker_pub.publish(marker)

    def run(self):
        rate = rospy.Rate(10) # 10 Hz
        
        rospy.loginfo("MPC Controller: Waiting for state...")
        while not rospy.is_shutdown():
            if self.current_state is None:
                rospy.logwarn_throttle(2, "MPC Controller: No Odom received yet.")
                rate.sleep()
                continue
            
            # Check if goal is reached
            if self.is_goal_reached(self.current_state[0], self.current_state[1]):
                rospy.loginfo_throttle(1, "🎯 Goal Reached! Stopping vehicle.")
                # Send stop command
                msg = AckermannDrive()
                msg.speed = 0.0
                msg.steering_angle = 0.0
                self.pub_cmd.publish(msg)
                rate.sleep()
                continue
            
            # Get Reference Trajectory
            # We need T+1 points
            ref_traj = self.get_reference(self.current_state[0], self.current_state[1], self.T + 1)
            
            # Transpose to match shape (4, T+1)
            ref_traj = ref_traj.T
            
            try:
                # Set Parameters
                self.opti.set_value(self.P, self.current_state)
                self.opti.set_value(self.Ref, ref_traj)
                
                # Warm Start
                self.opti.set_initial(self.X, self.prev_X)
                self.opti.set_initial(self.U, self.prev_U)
                
                # Solve
                sol = self.opti.solve()
                
                # Get Optimal Control
                u_opt = sol.value(self.U[:, 0])
                cmd_v = u_opt[0]
                cmd_steer = u_opt[1]
                
                # Store solution for next warm start
                self.prev_X[:, :-1] = sol.value(self.X)[:, 1:]
                self.prev_X[:, -1] = sol.value(self.X)[:, -1]
                
                self.prev_U[:, :-1] = sol.value(self.U)[:, 1:]
                self.prev_U[:, -1] = sol.value(self.U)[:, -1]
                
                # Publish Command
                msg = AckermannDrive()
                msg.speed = cmd_v
                msg.steering_angle = cmd_steer
                self.pub_cmd.publish(msg)
                
                # Debug Info
                cte = self.get_cross_track_error(self.current_state[0], self.current_state[1])
                rospy.loginfo_throttle(1, f"MPC: v={cmd_v:.2f}, steer={cmd_steer:.2f}, CTE={cte:.3f}")
                
            except Exception as e:
                rospy.logwarn_throttle(1, f"MPC Solver Timeout (Recovering...)")                                                
                x0, y0, th0, v0 = self.current_state
                
                new_prev_X = np.zeros((4, self.T + 1))
                new_prev_U = np.zeros((2, self.T))
                
                # Gelecekteki T adımı basitçe tahmin et (x = x + v*t)
                for k in range(self.T + 1):
                    new_prev_X[0, k] = x0 + v0 * k * self.dt * np.cos(th0)
                    new_prev_X[1, k] = y0 + v0 * k * self.dt * np.sin(th0)
                    new_prev_X[2, k] = th0
                    new_prev_X[3, k] = v0
                
                self.prev_X = new_prev_X
                self.prev_U = new_prev_U # Komutları sıfırla
            
            rate.sleep()

if __name__ == '__main__':
    try:
        controller = MPCController()
        controller.run()
    except rospy.ROSInterruptException:
        pass
