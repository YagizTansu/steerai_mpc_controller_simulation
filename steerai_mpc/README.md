# SteerAI MPC Controller

This package implements a Model Predictive Controller (MPC) for the POLARIS GEM e2 vehicle using CasADi and a learned neural network dynamics model. It features a modular architecture separating the control logic, optimization solver, vehicle dynamics, and path management.

## Overview

The package is designed with the following modular components:

- **MPC Controller**: The main node that orchestrates the control loop, handling state updates and sending control commands.
- **MPC Solver**: Encapsulates the CasADi optimization problem, constraints, and cost function.
- **Vehicle Model**: A neural network-based model (trained with `steerai_sysid`) that predicts vehicle dynamics.
- **Path Manager**: Handles loading, processing, and querying of the reference path.
- **Path Publisher**: A utility to publish raw path data from CSV files for visualization and processing.

## Dependencies

- ROS Noetic
- CasADi (`pip install casadi`)
- PyTorch
- `steerai_sysid` (for the trained model)

## Usage

The recommended way to run the controller is using the provided launch file, which handles parameter loading and node configuration.

1. **Train the Model**
   Ensure you have a trained model in `steerai_sysid`.
   ```bash
   rosrun steerai_sysid train_dynamics.py
   ```

2. **Run the MPC Controller**
   ```bash
   roslaunch steerai_mpc mpc_controller.launch
   ```

## Nodes

### `mpc_controller.py`
The central node that subscribes to odometry and path data, invokes the solver, and publishes drive commands.
- **Subscribes**: `/gem/base_footprint/odom`, `/steerai_mpc/reference_path`
- **Publishes**: `/gem/ackermann_cmd`

### `path_manager.py`
Manages the reference path. It processes the raw path (interpolation, smoothing, yaw calculation) and provides a service or topic for the controller to query the target state.

### `path_publisher.py`
Reads a CSV file containing waypoints and publishes it as a `nav_msgs/Path`. This allows for easy visualization in RViz and decoupling of path source from path processing.

### `vehicle_model.py`
Contains the `VehicleModel` class which loads the PyTorch model and provides a forward pass method compatible with the MPC solver's requirements.

### `mpc_solver.py`
Defines the optimization problem using CasADi. It sets up the decision variables, cost function (tracking error, control effort), and constraints (actuation limits, dynamic feasibility).

### `tf_broadcaster.py`
A helper node to broadcast necessary TF frames (e.g., `world` to `base_footprint`) if they are not provided by the simulation or localization system.

## Configuration

Configuration is managed via YAML files in the `config/` directory:

- **`config/mpc_params.yaml`**: Parameters for the MPC controller and solver.
    - `T`: Prediction horizon steps (default: 15)
    - `dt`: Time step in seconds (default: 0.1)
    - `v_ref`: Reference speed (default: 2.0 m/s)
    - `weights`: Cost function weights for cross-track, heading, speed, and steering errors.
    - `constraints`: Limits for speed (`v_max`) and steering (`delta_max`).

- **`config/path_params.yaml`**: Parameters for path loading and processing.
    - `csv_path`: Path to the reference CSV file.
    - `frame_id`: Reference frame for the path (e.g., `world`).

## Objective

The controller minimizes a cost function comprising:
- **Cross-track error**: Deviation distance from the reference path.
- **Heading error**: Deviation from the reference path's orientation.
- **Speed error**: Deviation from the reference speed (`v_ref`).
- **Control effort**: Penalties on steering magnitude and rate of change to ensure smooth driving.
