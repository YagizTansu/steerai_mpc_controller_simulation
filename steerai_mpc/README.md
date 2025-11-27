# SteerAI MPC Controller

This package implements a Model Predictive Controller (MPC) for the POLARIS GEM e2 vehicle using CasADi and a learned neural network dynamics model.

## Overview

The `mpc_controller.py` node solves a nonlinear optimization problem at each time step to compute the optimal steering and velocity commands. It uses a neural network trained by `steerai_sysid` as the prediction model.

## Dependencies

- ROS Noetic
- CasADi (`pip install casadi`)
- PyTorch
- `steerai_sysid` (for the trained model)

## Usage

1. **Train the Model**
   Ensure you have a trained model in `steerai_sysid`.
   ```bash
   rosrun steerai_sysid train_dynamics.py
   ```

2. **Run the MPC Controller**
   ```bash
   rosrun steerai_mpc mpc_controller.py
   ```

## Configuration

The controller parameters are defined in `mpc_controller.py`:
- `T`: Prediction horizon (default: 10 steps)
- `dt`: Time step (default: 0.1s)
- `v_ref`: Reference speed (default: 2.0 m/s)
- `v_max`: Maximum speed (5.5 m/s)
- `delta_max`: Maximum steering angle (0.6 rad)

## Objective

The controller minimizes a cost function comprising:
- Cross-track error (deviation from y=0)
- Heading error (deviation from yaw=0)
- Speed error (deviation from v_ref)
- Control effort (change in steering)
