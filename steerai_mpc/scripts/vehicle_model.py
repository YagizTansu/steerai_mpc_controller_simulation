#!/usr/bin/env python3

import torch
import numpy as np
import os
import rospkg
import joblib
import casadi as ca
import rospy

class VehicleModel:
    def __init__(self, dt=0.1):
        """
        Initialize Vehicle Model.
        Handles loading of Neural Network dynamics and provides symbolic expressions.
        
        :param dt: Time step
        """
        self.dt = dt
        
        # Load Model and Scalers
        self.load_model()
        
    def update_params(self, dt=None):
        """Update parameters dynamically."""
        if dt is not None: self.dt = dt

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

    def _neural_net_dynamics(self, v, cmd_v, cmd_steer):
        """
        Symbolic Neural Network forward pass using CasADi.
        Input: [v, cmd_v, cmd_steer]
        Output: [next_v, delta_yaw]
        """
        # Normalize Input
        inp = ca.vertcat(v, cmd_v, cmd_steer)
        inp_norm = (inp - self.mean_X) / self.scale_X
        
        # Forward Pass (ReLU activation)
        # Forward Pass (Softplus activation for smooth gradients)
        h1 = ca.mtimes(self.W1, inp_norm) + self.b1
        h1 = ca.log(1 + ca.exp(h1)) # Softplus
        
        h2 = ca.mtimes(self.W2, h1) + self.b2
        h2 = ca.log(1 + ca.exp(h2)) # Softplus
        
        out_norm = ca.mtimes(self.W3, h2) + self.b3
        
        # Denormalize Output: [next_v, delta_yaw]
        out = out_norm * self.scale_y + self.mean_y
        
        return out[0], out[1] # next_v, delta_yaw

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
        
        # Neural Network prediction
        next_v_pred, delta_yaw_pred = self._neural_net_dynamics(curr_v, cmd_v, cmd_steer)
            
        # State Update
        next_x = curr_x + curr_v * ca.cos(curr_yaw) * self.dt
        next_y = curr_y + curr_v * ca.sin(curr_yaw) * self.dt
        next_yaw = curr_yaw + delta_yaw_pred
        next_v = next_v_pred
        
        return ca.vertcat(next_x, next_y, next_yaw, next_v)
