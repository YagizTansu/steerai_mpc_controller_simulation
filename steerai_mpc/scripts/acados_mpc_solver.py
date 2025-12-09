#!/usr/bin/env python3

"""
Acados-based MPC Solver for Vehicle Path Tracking

This module implements a high-performance Model Predictive Controller using
the acados optimization framework. It provides 10-100x faster solve times
compared to traditional IPOPT-based solvers.

Key Features:
- Real-Time Iteration (RTI) SQP algorithm for guaranteed real-time performance
- Neural network-based vehicle dynamics model
- Discrete-time formulation
- Warm starting for improved convergence
"""

import numpy as np
import rospy
import os
import shutil
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import casadi as ca


class AcadosMPCSolver:
    """
    MPC Solver using acados optimization framework.
    
    Provides high-performance optimal control for autonomous vehicle path tracking
    using neural network-based dynamics.
    """
    
    def __init__(self, vehicle_model, params):
        """
        Initialize Acados MPC Solver.
        
        :param vehicle_model: Instance of VehicleModel class with export_acados_model() method
        :param params: Dictionary of parameters (T, dt, constraints, weights, solver_opts, acados)
        """
        self.vehicle_model = vehicle_model
        self.params = params
        
        # Unpack commonly used parameters
        self.T = params['T']  # Prediction horizon
        self.dt = params['dt']  # Time step
        self.nx = 4  # State dimension: [x, y, yaw, v]
        self.nu = 2  # Control dimension: [cmd_v, cmd_steer]
        
        # Initialize weights
        self.weights = params['weights']
        
        # Acados-specific parameters
        self.acados_params = params.get('acados', {})
        
        # Setup and build the solver
        self.ocp = self.setup_ocp()
        self.solver = self.build_solver()
        
        # Warm start variables
        self.prev_X = np.zeros((self.nx, self.T + 1))
        self.prev_U = np.zeros((self.nu, self.T))
        
        # Initialize warm start with target speed to avoid infeasibility
        target_speed = params.get('target_speed', 5.0)
        self.prev_X[3, :] = target_speed
        self.prev_U[0, :] = target_speed
        
        rospy.loginfo("AcadosMPCSolver: Initialized successfully with RTI-SQP algorithm")
    
    def setup_ocp(self):
        """
        Setup the Optimal Control Problem (OCP) for acados.
        
        :return: AcadosOcp object
        """
        ocp = AcadosOcp()
        
        # === Model Setup ===
        model_name, f_discrete = self.vehicle_model.export_acados_model()
        
        model = AcadosModel()
        model.name = model_name
        
        # Define symbolic variables for acados
        x = ca.MX.sym('x', self.nx)
        u = ca.MX.sym('u', self.nu)
        
        # Define parameter for reference (this will be updated at each solve)
        # p = [x_ref, y_ref, yaw_ref, v_ref, u_prev_0, u_prev_1]
        p = ca.MX.sym('p', self.nx + self.nu)  # Reference state + previous control
        
        model.x = x
        model.u = u
        model.p = p  # Set parameter
        
        # Discrete-time dynamics
        model.disc_dyn_expr = f_discrete(x, u)
        
        ocp.model = model
        
        # === Dimensions ===
        ocp.dims.N = self.T
        
        # === Cost Function ===
        # Extract reference and previous control from parameter
        x_ref = p[:self.nx]  # First 4 elements: [x_ref, y_ref, yaw_ref, v_ref]
        u_prev = p[self.nx:]  # Last 2 elements: [cmd_v_prev, cmd_steer_prev]
        
        # State errors
        x_err = x[0] - x_ref[0]
        y_err = x[1] - x_ref[1]
        yaw_err = x[2] - x_ref[2]
        v_err = x[3] - x_ref[3]
        
        # Position error (cross-track error)
        position_cost = self.weights['position'] * (x_err**2 + y_err**2)
        
        # Heading error (robust formulation)
        heading_cost = self.weights['heading'] * (1 - ca.cos(yaw_err))
        
        # Velocity error
        velocity_cost = self.weights['velocity'] * v_err**2
        
        # Control smoothness (rate penalties)
        steering_rate = u[1] - u_prev[1]  # Change in steering
        accel_rate = u[0] - u_prev[0]     # Change in velocity command
        
        steering_smooth_cost = self.weights['steering_smooth'] * steering_rate**2
        accel_smooth_cost = self.weights['acceleration_smooth'] * accel_rate**2
        
        # Total stage cost
        stage_cost = (position_cost + heading_cost + velocity_cost + 
                     steering_smooth_cost + accel_smooth_cost)
        
        ocp.cost.cost_type = 'EXTERNAL'
        ocp.cost.cost_type_e = 'EXTERNAL'
        
        ocp.model.cost_expr_ext_cost = stage_cost
        ocp.model.cost_expr_ext_cost_e = position_cost + heading_cost + velocity_cost  # Terminal cost (no control)


        # === Constraints ===
        # Control input bounds
        v_max = self.params['constraints']['v_max']
        delta_max = self.params['constraints']['delta_max']
        
        ocp.constraints.lbu = np.array([-v_max, -delta_max])
        ocp.constraints.ubu = np.array([v_max, delta_max])
        ocp.constraints.idxbu = np.array([0, 1])
        
        # Initial state constraint (will be set at runtime)
        ocp.constraints.x0 = np.zeros(self.nx)
        
        # Soft CTE (Cross-Track Error) constraint
        # NOTE: Temporarily disabled - acados requires additional slack variable setup
        # CTE is still heavily penalized in the cost function which is sufficient
        # TODO: Properly configure soft constraints with idxsh in future version
        # if 'cte_max' in self.params['constraints']:
        #     cte_max = self.params['constraints']['cte_max']
        #     h_expr = ca.sqrt(x_err**2 + y_err**2)
        #     ocp.model.con_h_expr = h_expr
        #     ocp.constraints.lh = np.array([0.0])
        #     ocp.constraints.uh = np.array([cte_max])
        #     ocp.constraints.idxsh = np.array([0])  # This is needed!
        #     ocp.cost.zl = np.array([0.0])
        #     ocp.cost.zu = np.array([0.0])
        #     ocp.cost.Zl = np.array([1000.0])
        #     ocp.cost.Zu = np.array([1000.0])


        # === Parameter Initialization ===
        # Set default parameter values (will be updated at runtime in solve())
        # Acados requires initial parameter values even if they'll be overwritten
        # p = [x_ref, y_ref, yaw_ref, v_ref, u_prev_0, u_prev_1]
        ocp.parameter_values = np.zeros(self.nx + self.nu)  # Reference + previous control

        
        # === Solver Options ===
        ocp.solver_options.tf = self.T * self.dt  # Prediction horizon time
        
        # QP solver
        ocp.solver_options.qp_solver = self.acados_params.get('qp_solver', 'PARTIAL_CONDENSING_HPIPM')
        ocp.solver_options.qp_solver_iter_max = self.acados_params.get('qp_solver_iter_max', 50)
        
        # Hessian approximation
        ocp.solver_options.hessian_approx = self.acados_params.get('hessian_approx', 'GAUSS_NEWTON')
        
        # Integrator (discrete for our model)
        ocp.solver_options.integrator_type = 'DISCRETE'
        
        # SQP iterations
        ocp.solver_options.nlp_solver_type = self.acados_params.get('solver_type', 'SQP_RTI')
        ocp.solver_options.nlp_solver_max_iter = self.acados_params.get('nlp_solver_max_iter', 1)
        
        # Globalization
        globalization = self.acados_params.get('globalization', 'MERIT_BACKTRACKING')
        if globalization and ocp.solver_options.nlp_solver_type == 'SQP':
            ocp.solver_options.globalization = globalization
        
        # Tolerance
        ocp.solver_options.qp_solver_tol_stat = 1e-2
        ocp.solver_options.qp_solver_tol_eq = 1e-2
        ocp.solver_options.qp_solver_tol_ineq = 1e-2
        ocp.solver_options.qp_solver_tol_comp = 1e-2
        
        return ocp
    
    def build_solver(self):
        """
        Build the acados solver from the OCP.
        
        :return: AcadosOcpSolver object
        """
        # Set output directory for generated code
        code_export_dir = self.acados_params.get('code_export_dir', 'c_generated_code')
        
        # Clean up old generated code if exists
        if os.path.exists(code_export_dir):
            try:
                shutil.rmtree(code_export_dir)
                rospy.loginfo(f"AcadosMPCSolver: Cleaned up old generated code in {code_export_dir}")
            except Exception as e:
                rospy.logwarn(f"AcadosMPCSolver: Could not clean up old code: {e}")
        
        # Build solver
        try:
            solver = AcadosOcpSolver(self.ocp, json_file=self.ocp.model.name + '_ocp.json')
            rospy.loginfo("AcadosMPCSolver: Solver built successfully")
            return solver
        except Exception as e:
            rospy.logerr(f"AcadosMPCSolver: Failed to build solver: {e}")
            raise e
    
    def update_weights(self, new_weights):
        """
        Update cost function weights dynamically.
        
        Note: For acados, weight updates require rebuilding the solver.
        For real-time applications, consider using parameter-based weights.
        
        :param new_weights: Dictionary of new weight values
        """
        self.weights.update(new_weights)
        rospy.logwarn("AcadosMPCSolver: Weight update requires solver rebuild. "
                     "Rebuilding solver (this may take a moment)...")
        
        # Rebuild OCP and solver with new weights
        self.ocp = self.setup_ocp()
        self.solver = self.build_solver()
        
        rospy.loginfo("AcadosMPCSolver: Solver rebuilt with new weights")
    
    def solve(self, current_state, reference_trajectory):
        """
        Solve the MPC optimization problem.
        
        :param current_state: numpy array [x, y, yaw, v]
        :param reference_trajectory: numpy array of shape (4, T+1) -> [x, y, yaw, v]
        :return: Tuple (cmd_v, cmd_steer, solved_successfully)
        """
        try:
            # Set initial state constraint
            self.solver.set(0, 'lbx', current_state)
            self.solver.set(0, 'ubx', current_state)
            
            # Set reference trajectory and previous control as parameters for all stages
            for k in range(self.T):
                # Set reference for this stage
                y_ref = reference_trajectory[:, k]
                
                # Get previous control (from k-1, or initial if k==0)
                if k == 0:
                    u_prev = self.prev_U[:, 0]  # Use warm start for first stage
                else:
                    u_prev = self.prev_U[:, k-1]
                
                # Combine reference and previous control into parameter vector
                # p = [x_ref, y_ref, yaw_ref, v_ref, u_prev_0, u_prev_1]
                param = np.concatenate([y_ref, u_prev])
                self.solver.set(k, 'p', param)
                
                # Set warm start
                self.solver.set(k, 'x', self.prev_X[:, k])
                self.solver.set(k, 'u', self.prev_U[:, k])
            
            # Terminal stage warm start and reference (no previous control needed for terminal)
            self.solver.set(self.T, 'x', self.prev_X[:, self.T])
            param_terminal = np.concatenate([reference_trajectory[:, self.T], np.zeros(self.nu)])
            self.solver.set(self.T, 'p', param_terminal)
            
            # Solve OCP
            status = self.solver.solve()
            
            if status == 0:
                # Solution found
                # Extract first control input
                u_opt = self.solver.get(0, 'u')
                cmd_v = u_opt[0]
                cmd_steer = u_opt[1]
                
                # Update warm start for next iteration
                for k in range(self.T):
                    self.prev_X[:, k] = self.solver.get(k, 'x')
                    self.prev_U[:, k] = self.solver.get(k, 'u')
                self.prev_X[:, self.T] = self.solver.get(self.T, 'x')
                
                return cmd_v, cmd_steer, True
            
            elif status == 2:
                # Maximum iterations reached (acceptable for RTI mode)
                u_opt = self.solver.get(0, 'u')
                cmd_v = u_opt[0]
                cmd_steer = u_opt[1]
                
                # Update warm start
                for k in range(self.T):
                    self.prev_X[:, k] = self.solver.get(k, 'x')
                    self.prev_U[:, k] = self.solver.get(k, 'u')
                self.prev_X[:, self.T] = self.solver.get(self.T, 'x')
                
                rospy.logdebug("AcadosMPCSolver: Max iterations reached (RTI mode - acceptable)")
                return cmd_v, cmd_steer, True
            
            else:
                # Solver failed
                rospy.logwarn_throttle(1, f"AcadosMPCSolver: Solver failed with status {status}")
                return self._get_fallback_control(current_state)
                
        except Exception as e:
            rospy.logwarn_throttle(1, f"AcadosMPCSolver: Exception during solve: {e}")
            return self._get_fallback_control(current_state)
    
    def _get_fallback_control(self, current_state):
        """
        Generate fallback control when solver fails.
        
        :param current_state: Current vehicle state [x, y, yaw, v]
        :return: Tuple (cmd_v, cmd_steer, False)
        """
        # Reset warm start with simple prediction
        x0, y0, th0, v0 = current_state
        target_speed = self.params.get('target_speed', 5.0)
        
        for k in range(self.T + 1):
            self.prev_X[0, k] = x0 + v0 * k * self.dt * np.cos(th0)
            self.prev_X[1, k] = y0 + v0 * k * self.dt * np.sin(th0)
            self.prev_X[2, k] = th0
            self.prev_X[3, k] = target_speed
        
        self.prev_U[0, :] = target_speed
        self.prev_U[1, :] = 0.0
        
        return 0.0, 0.0, False
    
    def __del__(self):
        """Cleanup generated code on deletion."""
        code_export_dir = self.acados_params.get('code_export_dir', 'c_generated_code')
        if os.path.exists(code_export_dir):
            try:
                shutil.rmtree(code_export_dir)
            except:
                pass
