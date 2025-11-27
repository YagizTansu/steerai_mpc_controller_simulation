#!/usr/bin/env python3

import rospy
import casadi as ca
import torch
import numpy as np
import os
import rospkg
import joblib
from ackermann_msgs.msg import AckermannDrive
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion

class MPCController:
    def __init__(self):
        rospy.init_node('mpc_controller')
        
        # Load Model and Scalers
        self.load_model()
        
        # MPC Parameters
        self.T = 10 # Horizon
        self.dt = 0.1 # Time step
        self.v_ref = 2.0 # Reference speed (m/s)
        
        # Constraints
        self.v_max = 5.5
        self.delta_max = 0.6
        
        # Setup CasADi Solver
        self.setup_solver()
        
        # ROS Setup
        self.pub_cmd = rospy.Publisher('/gem/ackermann_cmd', AckermannDrive, queue_size=1)
        self.sub_odom = rospy.Subscriber('/gem/base_footprint/odom', Odometry, self.odom_callback)
        
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
        
        # Parameters (Initial State)
        self.P = self.opti.parameter(4)
        
        # Cost Function
        obj = 0
        for k in range(self.T):
            # State Error
            # Simple reference: y = 0 (Lane centering)
            obj += 10.0 * self.X[1, k]**2 # Minimize y (Cross Track Error)
            obj += 5.0 * self.X[2, k]**2  # Minimize yaw (Heading Error)
            obj += 1.0 * (self.X[3, k] - self.v_ref)**2 # Maintain reference speed
            
            # Control Effort
            if k > 0:
                obj += 1.0 * (self.U[1, k] - self.U[1, k-1])**2 # Smooth steering
                
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
            next_v_pred, delta_yaw_pred = self.neural_net_dynamics(curr_v, cmd_v, cmd_steer)
            
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
            self.opti.subject_to(self.opti.bounded(0, self.U[0, k], self.v_max)) # 0 <= cmd_v <= v_max
            self.opti.subject_to(self.opti.bounded(-self.delta_max, self.U[1, k], self.delta_max))
            
        # Initial Condition Constraint
        self.opti.subject_to(self.X[:, 0] == self.P)
        
        # Solver Options
        p_opts = {'expand': True}
        # Solver Options
        p_opts = {'expand': True}
        s_opts = {'max_iter': 500, 'print_level': 0, 'tol': 1e-3}
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

    def run(self):
        rate = rospy.Rate(10) # 10 Hz
        
        while not rospy.is_shutdown():
            if self.current_state is None:
                rate.sleep()
                continue
            
            try:
                # Set Initial State Parameter
                self.opti.set_value(self.P, self.current_state)
                
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
                # Shift X: [x1, x2, ..., xT, xT]
                self.prev_X[:, :-1] = sol.value(self.X)[:, 1:]
                self.prev_X[:, -1] = sol.value(self.X)[:, -1]
                
                # Shift U: [u1, u2, ..., uT, uT]
                self.prev_U[:, :-1] = sol.value(self.U)[:, 1:]
                self.prev_U[:, -1] = sol.value(self.U)[:, -1]
                
                # Publish Command
                msg = AckermannDrive()
                msg.speed = cmd_v
                msg.steering_angle = cmd_steer
                self.pub_cmd.publish(msg)
                
                # rospy.loginfo(f"MPC Solved: v={cmd_v:.2f}, steer={cmd_steer:.2f}")
                
            except Exception as e:
                rospy.logwarn(f"MPC Solver Failed: {e}")
                # Stop on failure
                msg = AckermannDrive()
                msg.speed = 0.0
                msg.steering_angle = 0.0
                self.pub_cmd.publish(msg)
                
                # Reset warm start on failure
                self.prev_X = np.zeros((4, self.T + 1))
                self.prev_U = np.zeros((2, self.T))
            
            rate.sleep()

if __name__ == '__main__':
    try:
        controller = MPCController()
        controller.run()
    except rospy.ROSInterruptException:
        pass
