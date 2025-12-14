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
        
        # Initialize warm start with zeros
        self.prev_X[3, :] = 0.0
        self.prev_U[0, :] = 0.0

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
        self.W_cte = self.opti.parameter()

        # Last Command (for smoothness at k=0)
        self.LastCmd = self.opti.parameter(2) # [cmd_v, cmd_steer]
        
        # Initialize weight parameters
        self.opti.set_value(self.W_pos, self.weights['position'])
        self.opti.set_value(self.W_head, self.weights['heading'])
        self.opti.set_value(self.W_vel, self.weights['velocity'])
        self.opti.set_value(self.W_steer, self.weights['steering_smooth'])
        self.opti.set_value(self.W_acc, self.weights['acceleration_smooth'])
        self.opti.set_value(self.W_cte, self.weights['cte'])
        
        # Cost Function
        obj = 0
        
        # 1. Smoothness for the very first step (k=0 transition)
        # Penalize difference between LastCmd (executed) and U[:, 0] (planned)
        obj += self.W_steer * (self.U[1, 0] - self.LastCmd[1])**2
        obj += self.W_acc * (self.U[0, 0] - self.LastCmd[0])**2

        for k in range(1, self.T + 1):
            # State Error
            x_err = self.X[0, k] - self.Ref[0, k]
            y_err = self.X[1, k] - self.Ref[1, k]
            yaw_err = self.X[2, k] - self.Ref[2, k]
            v_err = self.X[3, k] - self.Ref[3, k]
            
            obj += self.W_pos * (x_err**2 + y_err**2)
            obj += self.W_head * (1 - ca.cos(yaw_err)) # Robust heading error
            obj += self.W_vel * v_err**2
            
            # Soft CTE Constraint - Exponential Penalty for violations
            cte_max = self.params['constraints']['cte_max']
            ref_yaw = self.Ref[2, k]
            
            # Calculate Lateral Error (Cross Track Error)
            lat_err = -ca.sin(ref_yaw) * x_err + ca.cos(ref_yaw) * y_err
            
            # Use absolute lateral error for constraint
            cte_penalty = ca.fmax(0, ca.fabs(lat_err) - cte_max)**2
            obj += self.W_cte * cte_penalty
            
            # Control Effort / Smoothness
            if k < self.T:
                obj += self.W_steer * (self.U[1, k] - self.U[1, k-1])**2
                obj += self.W_acc * (self.U[0, k] - self.U[0, k-1])**2
                
        self.opti.minimize(obj)
        
        # Constraints
        for k in range(self.T):
            # Dynamics
            curr_state = self.X[:, k]
            control_input = self.U[:, k]

            v_max = self.params['constraints']['v_max']
            delta_max = self.params['constraints']['delta_max']
            
            # Use Neural Network model for all steps
            # Yaw rate is calculated internally by the model (model encapsulated)
            next_state_expr = self.vehicle_model.get_next_state(curr_state, control_input)
            
            self.opti.subject_to(self.X[:, k+1] == next_state_expr) # State Constraint
            self.opti.subject_to(self.opti.bounded(-v_max, self.U[0, k], v_max)) # Input Constraint
            self.opti.subject_to(self.opti.bounded(-delta_max, self.U[1, k], delta_max)) # Input Constraint
                
        # Initial Condition
        self.opti.subject_to(self.X[:, 0] == self.P)
        
        # Solver Options
        p_opts = {'expand': True}
        s_opts = self.params.get('solver_opts', {})
        # Default solver options if not provided
        if not s_opts:
            s_opts = {
                'max_iter': 100,
                'print_level': 0,
                'tol': 1e-2,
                'acceptable_tol': 1e-1,
            }
            
        self.opti.solver('ipopt', p_opts, s_opts)

    def solve(self, current_state, reference_trajectory, last_cmd):
        """
        Solve the MPC optimization problem.
        
        :param current_state: [x, y, yaw, v]
        :param reference_trajectory: Shape (4, T+1) -> [x, y, yaw, v]
        :param last_cmd: [cmd_v, cmd_steer] - Last applied command
        :return: (cmd_v, cmd_steer, solved_successfully)
        """
        try:
            # Set Parameters
            self.opti.set_value(self.P, current_state)
            self.opti.set_value(self.Ref, reference_trajectory)
            self.opti.set_value(self.LastCmd, last_cmd)
            
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
            
            return cmd_v, cmd_steer, True
            
        except Exception as e:
            rospy.logwarn_throttle(1, f"MPC Solver Failed: {str(e)}")
            
            # Reset Warm Start on failure using Reference Trajectory
            x0, y0, th0, v0 = current_state
            
            for k in range(self.T + 1):
                # Use Reference Velocity for prediction
                ref_v = reference_trajectory[3, k] if k < reference_trajectory.shape[1] else 0.0
                
                self.prev_X[0, k] = x0 + ref_v * k * self.dt * np.cos(th0)
                self.prev_X[1, k] = y0 + ref_v * k * self.dt * np.sin(th0)
                self.prev_X[2, k] = th0
                self.prev_X[3, k] = ref_v
                
            # Set initial control guess to the first reference velocity
            first_ref_v = reference_trajectory[3, 0] if reference_trajectory.shape[1] > 0 else 0.0
            self.prev_U[0, :] = first_ref_v
            self.prev_U[1, :] = 0.0
            
            return 0.0, 0.0, False
