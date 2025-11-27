# SteerAI System Identification

This package performs System Identification for the POLARIS GEM e2 vehicle using PyTorch. It trains a neural network to model the vehicle's dynamics based on collected data.

## Overview

The `train_dynamics.py` script reads data collected by `steerai_data_collector`, processes it, and trains a Feed-Forward Neural Network (MLP) to predict the next state of the vehicle.

## Dependencies

- ROS Noetic
- PyTorch
- Pandas
- Scikit-learn
- Matplotlib
- `steerai_data_collector` (for data access)

## Installation

Install the required Python packages:

```bash
pip install torch pandas scikit-learn matplotlib joblib rospkg
```

## Usage

1. **Collect Data**
   Ensure you have run the data collector and generated `training_data.csv`.
   ```bash
   rosrun steerai_data_collector data_collector.py
   ```

2. **Train the Model**
   ```bash
   rosrun steerai_sysid train_dynamics.py
   ```

## Outputs

The script saves the following files in the `steerai_sysid` package directory:

- `dynamics_model.pth`: The trained PyTorch model state dictionary.
- `scaler_X.pkl`: Scikit-learn StandardScaler for input features.
- `scaler_y.pkl`: Scikit-learn StandardScaler for target features.
- `training_results.png`: Plots showing training/validation loss and prediction performance.

## Model Details

- **Inputs**: `[current_speed, cmd_speed, cmd_steering_angle]`
- **Outputs**: `[next_speed, delta_yaw]`
- **Architecture**: MLP (3 -> 64 -> 64 -> 2) with ReLU activations.
