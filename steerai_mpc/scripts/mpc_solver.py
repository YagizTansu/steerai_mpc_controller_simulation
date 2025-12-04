#!/usr/bin/env python3

import casadi as ca
import numpy as np
import rospy

class MPCSolver:
    def __init__(self, vehicle_model, params):
        """
        Initialize MPC Solver.
        
        :param vehicle_model: Instance of VehicleModel class
        :param params: Dictionary of parameters (T, dt, constraints, weights, solver_opts)
        """
        self.vehicle_model = vehicle_model
        self.params = params
        
        # Unpack commonly used parameters
        self.T = params['T']
        self.dt = params['dt']
        
        # Initialize weights
        self.weights = params['weights']
        
        # Setup Solver
        self.setup_solver()
        
        # Warm start variables
        self.prev_X = np.zeros((4, self.T + 1))
        self.prev_U = np.zeros((2, self.T))
        
        # Initialize warm start with target speed to avoid infeasibility
        target_speed = params.get('target_speed', 5.0)
        self.prev_X[3, :] = target_speed
        self.prev_U[0, :] = target_speed

    def setup_solver(self):
        """Setup CasADi Opti stack."""
        self.opti = ca.Opti()
        
        # Decision variables
        self.X = self.opti.variable(4, self.T + 1) # State [x, y, yaw, v]
        self.U = self.opti.variable(2, self.T)     # Control [cmd_v, cmd_steer]
        
        # Parameters (Inputs to the solver)
        self.P = self.opti.parameter(4) # Initial State
        self.Ref = self.opti.parameter(4, self.T + 1) # Reference Trajectory
        
        # Weight Parameters (allows dynamic reconfigure without rebuilding graph)
        self.W_pos = self.opti.parameter()
        self.W_head = self.opti.parameter()
        self.W_vel = self.opti.parameter()
        self.W_steer = self.opti.parameter()
        self.W_acc = self.opti.parameter()
        
        # Initialize weight parameters
        self.opti.set_value(self.W_pos, self.weights['position'])
        self.opti.set_value(self.W_head, self.weights['heading'])
        self.opti.set_value(self.W_vel, self.weights['velocity'])
        self.opti.set_value(self.W_steer, self.weights['steering_smooth'])
        self.opti.set_value(self.W_acc, self.weights['acceleration_smooth'])
        
        # Cost Function
        obj = 0
        for k in range(self.T):
            # State Error
            x_err = self.X[0, k] - self.Ref[0, k]
            y_err = self.X[1, k] - self.Ref[1, k]
            yaw_err = self.X[2, k] - self.Ref[2, k]
            v_err = self.X[3, k] - self.Ref[3, k]
            
            obj += self.W_pos * (x_err**2 + y_err**2)
            obj += self.W_head * (1 - ca.cos(yaw_err)) # Robust heading error
            obj += self.W_vel * v_err**2
            
            # Soft CTE Constraint - Exponential Penalty for violations
            if 'cte_max' in self.params['constraints']:
                cte_max = self.params['constraints']['cte_max']
                position_error = ca.sqrt(x_err**2 + y_err**2)
                # Smooth penalty that increases exponentially beyond limit
                cte_penalty = ca.fmax(0, position_error - cte_max)**2
                obj += 1000.0 * cte_penalty  # Very high penalty weight
            
            # Control Effort / Smoothness
            if k > 0:
                obj += self.W_steer * (self.U[1, k] - self.U[1, k-1])**2
                obj += self.W_acc * (self.U[0, k] - self.U[0, k-1])**2
                
        self.opti.minimize(obj)
        
        # Constraints
        for k in range(self.T):
            # Dynamics
            curr_state = self.X[:, k]
            control_input = self.U[:, k]
            
            # Estimate yaw_rate kinematically for the model input
            # yaw_rate = v * tan(delta) / L
            L = 1.75
            v = curr_state[3]
            delta = control_input[1]
            yaw_rate = v * ca.tan(delta) / L
            
            # Use Neural Network model for all steps
            next_state_expr = self.vehicle_model.get_next_state(curr_state, yaw_rate, control_input)
            
            self.opti.subject_to(self.X[:, k+1] == next_state_expr)
            
            # Input Constraints
            v_max = self.params['constraints']['v_max']
            delta_max = self.params['constraints']['delta_max']
            
            self.opti.subject_to(self.opti.bounded(-v_max, self.U[0, k], v_max))
            self.opti.subject_to(self.opti.bounded(-delta_max, self.U[1, k], delta_max))
            
        # Initial Condition
        self.opti.subject_to(self.X[:, 0] == self.P)
        
        # Solver Options
        p_opts = {'expand': True}
        s_opts = self.params.get('solver_opts', {})
            
        self.opti.solver('ipopt', p_opts, s_opts)

    def solve(self, current_state, reference_trajectory):
        """
        Solve the MPC optimization problem.
        
        :param current_state: [x, y, yaw, v]
        :param reference_trajectory: Shape (4, T+1) -> [x, y, yaw, v]
        :return: (cmd_v, cmd_steer, solved_successfully)
        """
        try:
            # Set Parameters
            self.opti.set_value(self.P, current_state)
            self.opti.set_value(self.Ref, reference_trajectory)
            
            # Warm Start
            self.opti.set_initial(self.X, self.prev_X)
            self.opti.set_initial(self.U, self.prev_U)
            
            # Solve
            sol = self.opti.solve()
            
            # Extract Solution
            u_opt = sol.value(self.U[:, 0])
            cmd_v = u_opt[0]
            cmd_steer = u_opt[1]
            
            # Update Warm Start
            self.prev_X[:, :-1] = sol.value(self.X)[:, 1:]
            self.prev_X[:, -1] = sol.value(self.X)[:, -1]
            
            self.prev_U[:, :-1] = sol.value(self.U)[:, 1:]
            self.prev_U[:, -1] = sol.value(self.U)[:, -1]
            
            return cmd_v, cmd_steer
            
        except Exception as e:
            rospy.logwarn_throttle(1, f"MPC Solver Failed: {str(e)}")
            
            # Reset Warm Start on failure (simple prediction)
            x0, y0, th0, v0 = current_state
            target_speed = self.params.get('target_speed', 5.0)
            
            for k in range(self.T + 1):
                self.prev_X[0, k] = x0 + v0 * k * self.dt * np.cos(th0)
                self.prev_X[1, k] = y0 + v0 * k * self.dt * np.sin(th0)
                self.prev_X[2, k] = th0
                self.prev_X[3, k] = target_speed
                
            self.prev_U[0, :] = target_speed
            self.prev_U[1, :] = 0.0
            
            return 0.0, 0.0
