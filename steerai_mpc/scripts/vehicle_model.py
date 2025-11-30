#!/usr/bin/env python3

import torch
import numpy as np
import os
import rospkg
import joblib
import casadi as ca
import rospy

class VehicleModel:
    def __init__(self, dt=0.1, L=1.75, v_low=0.5, v_high=2.0):
        """
        Initialize Vehicle Model.
        Handles loading of Neural Network dynamics and provides symbolic expressions.
        
        :param dt: Time step
        :param L: Wheelbase
        :param v_low: Low speed threshold for hybrid blending
        :param v_high: High speed threshold for hybrid blending
        """
        self.dt = dt
        self.L = L
        self.v_low = v_low
        self.v_high = v_high
        
        # Load Model and Scalers
        self.load_model()
        
    def update_params(self, dt=None, L=None, v_low=None, v_high=None):
        """Update parameters dynamically."""
        if dt is not None: self.dt = dt
        if L is not None: self.L = L
        if v_low is not None: self.v_low = v_low
        if v_high is not None: self.v_high = v_high

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
        h1 = ca.mtimes(self.W1, inp_norm) + self.b1
        h1 = ca.fmax(0, h1) # ReLU
        
        h2 = ca.mtimes(self.W2, h1) + self.b2
        h2 = ca.fmax(0, h2) # ReLU
        
        out_norm = ca.mtimes(self.W3, h2) + self.b3
        
        # Denormalize Output: [next_v, delta_yaw]
        out = out_norm * self.scale_y + self.mean_y
        
        return out[0], out[1] # next_v, delta_yaw

    def get_next_state(self, curr_state, control_input, use_hybrid=True):
        """
        Returns the symbolic expression for the next state.
        
        :param curr_state: CasADi variable [x, y, yaw, v]
        :param control_input: CasADi variable [cmd_v, cmd_steer]
        :param use_hybrid: Boolean, if True uses NN/Kinematic blending, else pure Kinematic
        :return: Next state symbolic expression [x_next, y_next, yaw_next, v_next]
        """
        curr_x = curr_state[0]
        curr_y = curr_state[1]
        curr_yaw = curr_state[2]
        curr_v = curr_state[3]
        
        cmd_v = control_input[0]
        cmd_steer = control_input[1]
        
        # Kinematic prediction (Baseline)
        next_v_kin = cmd_v
        delta_yaw_kin = (curr_v / self.L * ca.tan(cmd_steer)) * self.dt
        
        if use_hybrid:
            # Neural Network prediction
            next_v_nn, delta_yaw_nn = self._neural_net_dynamics(curr_v, cmd_v, cmd_steer)
            
            # Hybrid blending based on vehicle speed
            # alpha=0: pure kinematic (low speed), alpha=1: pure NN (high speed)
            # alpha = ca.fmin(1.0, ca.fmax(0.0, (curr_v - self.v_low) / (self.v_high - self.v_low)))
            alpha = 1
            next_v_pred = (1 - alpha) * next_v_kin + alpha * next_v_nn
            delta_yaw_pred = (1 - alpha) * delta_yaw_kin + alpha * delta_yaw_nn
        else:
            # Pure Kinematic
            next_v_pred = next_v_kin
            delta_yaw_pred = delta_yaw_kin
            
        # State Update
        next_x = curr_x + curr_v * ca.cos(curr_yaw) * self.dt
        next_y = curr_y + curr_v * ca.sin(curr_yaw) * self.dt
        next_yaw = curr_yaw + delta_yaw_pred
        next_v = next_v_pred
        
        return ca.vertcat(next_x, next_y, next_yaw, next_v)
