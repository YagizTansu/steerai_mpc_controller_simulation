# SteerAI Data Collector

This package contains the `data_collector.py` node, designed to record vehicle dynamics data for System Identification of the POLARIS GEM e2 platform.

## Overview

The `data_collector` node drives the vehicle using a persistent excitation strategy (sinusoidal steering and variable velocity) and logs the state and control inputs to a CSV file.

## Dependencies

- ROS Noetic
- `ackermann_msgs`
- `geometry_msgs`
- `nav_msgs`
- `rospy`

## Usage

1. **Launch the Simulation**
   ```bash
   roslaunch gem_gazebo gem_gazebo_rviz.launch
   ```

2. **Run the Data Collector**
   ```bash
   rosrun steerai_data_collector data_collector.py
   ```

## Output

The node generates a `training_data.csv` file in the directory where it is executed.

### CSV Columns

- `timestamp`: ROS timestamp (seconds)
- `cmd_speed`: Commanded linear velocity (m/s)
- `cmd_steering_angle`: Commanded steering angle (rad)
- `curr_x`: Current X position (m)
- `curr_y`: Current Y position (m)
- `curr_yaw`: Current Yaw angle (rad)
- `curr_speed`: Current linear velocity (m/s)

## Safety

The node includes a safety shutdown hook that stops the vehicle (publishes 0 speed/steering) when the node is terminated (e.g., via `Ctrl+C`).
