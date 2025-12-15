#!/usr/bin/env python3

import torch
import numpy as np
import os
import rospkg
import joblib
import casadi as ca
import rospy

class VehicleModel:
    def __init__(self, dt=0.1, wheelbase=1.75):
        """
        Initialize Vehicle Model.
        Handles loading of Neural Network dynamics and provides symbolic expressions.
        
        :param dt: Time step
        :param wheelbase: Vehicle wheelbase (meters)
        """
        self.dt = dt
        self.wheelbase = wheelbase
        
        # Load Model and Scalers
        self.load_model()

    def load_model(self):
        """Load PyTorch model and scalers from steerai_sysid package."""
        try:
            rospack = rospkg.RosPack()
            sysid_path = rospack.get_path('steerai_sysid')
            
            # Load Scalers
            self.scaler_X = joblib.load(os.path.join(sysid_path, 'scaler_X.pkl'))
            self.scaler_y = joblib.load(os.path.join(sysid_path, 'scaler_y.pkl'))
            
            # Load PyTorch Model Weights
            model_path = os.path.join(sysid_path, 'dynamics_model.pth')
            state_dict = torch.load(model_path, map_location='cpu') # Ensure CPU load
            
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
            
            rospy.loginfo("VehicleModel: Neural network dynamics loaded successfully.")
            
        except Exception as e:
            rospy.logerr(f"VehicleModel: Failed to load model: {e}")
            raise e

    def _neural_net_dynamics(self, v, yaw_rate, cmd_v, cmd_steer):
        """
        Symbolic Neural Network forward pass using CasADi.
        Input: [v, yaw_rate, cmd_v, cmd_steer]
        Output: [delta_v, delta_yaw]
        """
        # Normalize Input
        inp = ca.vertcat(v, yaw_rate, cmd_v, cmd_steer)
        inp_norm = (inp - self.mean_X) / self.scale_X
        
        # Forward Pass (Softplus activation for smooth gradients)
        h1 = ca.mtimes(self.W1, inp_norm) + self.b1
        h1 = ca.log(1 + ca.exp(h1)) # Softplus
        
        h2 = ca.mtimes(self.W2, h1) + self.b2
        h2 = ca.log(1 + ca.exp(h2)) # Softplus
        
        out_norm = ca.mtimes(self.W3, h2) + self.b3
        
        # Denormalize Output: [delta_v, delta_yaw]
        out = out_norm * self.scale_y + self.mean_y
        
        return out[0], out[1] # delta_v, delta_yaw

    def get_next_state(self, curr_state, control_input):
        """
        Returns the symbolic expression for the next state using Neural Network dynamics.
        
        :param curr_state: CasADi variable [x, y, yaw, v]
        :param control_input: CasADi variable [cmd_v, cmd_steer]
        :return: Next state symbolic expression [x_next, y_next, yaw_next, v_next]
        """
        curr_x = curr_state[0]
        curr_y = curr_state[1]
        curr_yaw = curr_state[2]
        curr_v = curr_state[3]
        
        cmd_v = control_input[0]
        cmd_steer = control_input[1]
        
        # Calculate Kinematic Yaw Rate for Model Input
        # yaw_rate = v * tan(delta) / L
        current_yaw_rate = curr_v * ca.tan(cmd_steer) / self.wheelbase
        
        # Neural Network prediction
        delta_v_pred, delta_yaw_pred = self._neural_net_dynamics(curr_v, current_yaw_rate, cmd_v, cmd_steer)
            
        # State Update
        next_x = curr_x + curr_v * ca.cos(curr_yaw) * self.dt
        next_y = curr_y + curr_v * ca.sin(curr_yaw) * self.dt
        next_yaw = curr_yaw + delta_yaw_pred
        next_v = curr_v + delta_v_pred # Residual update
        
        return ca.vertcat(next_x, next_y, next_yaw, next_v)

    def _neural_net_dynamics_numpy(self, v, yaw_rate, cmd_v, cmd_steer):
        """
        Numpy Neural Network forward pass for fast prediction (no CasADi overhead).
        Input: scalars
        Output: delta_v, delta_yaw
        """
        # Normalize Input
        inp = np.array([v, yaw_rate, cmd_v, cmd_steer])
        inp_norm = (inp - self.mean_X) / self.scale_X
        
        # Forward Pass (Softplus: log(1 + exp(x)))
        h1 = np.dot(self.W1, inp_norm) + self.b1
        h1 = np.log(1 + np.exp(h1)) # Softplus
        
        h2 = np.dot(self.W2, h1) + self.b2
        h2 = np.log(1 + np.exp(h2)) # Softplus
        
        out_norm = np.dot(self.W3, h2) + self.b3
        
        # Denormalize Output
        out = out_norm * self.scale_y + self.mean_y
        
        return out[0], out[1]

    def predict_next_state_numpy(self, current_state, control_input):
        """
        Predict next state using NumPy (for delay compensation).
        
        :param current_state: [x, y, yaw, v]
        :param control_input: [cmd_v, cmd_steer]
        :return: [x_next, y_next, yaw_next, v_next]
        """
        curr_x = current_state[0]
        curr_y = current_state[1]
        curr_yaw = current_state[2]
        curr_v = current_state[3]
        
        cmd_v = control_input[0]
        cmd_steer = control_input[1]
        
        # Kinematic Yaw Rate
        current_yaw_rate = curr_v * np.tan(cmd_steer) / self.wheelbase
        
        # NN Prediction
        delta_v_pred, delta_yaw_pred = self._neural_net_dynamics_numpy(curr_v, current_yaw_rate, cmd_v, cmd_steer)
        
        # State Update
        next_x = curr_x + curr_v * np.cos(curr_yaw) * self.dt
        next_y = curr_y + curr_v * np.sin(curr_yaw) * self.dt
        next_yaw = curr_yaw + delta_yaw_pred
        next_v = curr_v + delta_v_pred
        
        return np.array([next_x, next_y, next_yaw, next_v])
