# SteerAI: Learning-Based MPC for Autonomous Racing

This project implements a learning-based Model Predictive Controller (MPC) for the POLARIS GEM e2 vehicle in a ROS Noetic simulation environment. It consists of three main stages: Data Collection, System Identification, and MPC Control.

## 1. Build and Run Instructions

### Prerequisites
- **ROS Noetic**
- **Python 3**
- **ROS Packages**:
  - `ackermann_msgs`
  - `geometry_msgs`
  - `nav_msgs`
  - `visualization_msgs`
  - `jsk_rviz_plugins`
  - `tf`
  - `dynamic_reconfigure`
  - `gem_gazebo` (Simulation environment)
- **Python Libraries**:
  - `torch` (PyTorch)
  - `casadi` (Optimization)
  - `pandas` (Data manipulation)
  - `numpy` (Numerical operations)
  - `scikit-learn` (Preprocessing)
  - `matplotlib` (Plotting)
  - `scipy` (Interpolation/KDTree)
  - `joblib` (Model persistence)
  - `rospkg` (ROS package path handling)

### Build
Clone the repository into your catkin workspace and build:
```bash
cd ~/catkin_ws/src
# Clone this repository
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

### Run the Simulation
1. **Launch the Gazebo Simulation:**
   ```bash
   roslaunch gem_gazebo gem_gazebo_rviz.launch
   ```

2. **Run the MPC Controller:**
   ```bash
   roslaunch steerai_mpc mpc_controller.launch
   ```
   *Note: The controller will start and wait for a path to be published on `/gem/raw_path`.*

3. **Publish the Reference Path:**
   In a new terminal, publish the desired path (e.g., the modified reference path):
   ```bash
   rosrun steerai_mpc path_publisher.py _path_file:=paths/reference_path_modified.csv
   ```
   *Note: You can specify any CSV file relative to the package or an absolute path.*

---

---

## 2. Data Collection

To train the neural network dynamics model, we first need to collect a diverse dataset covering the vehicle's operating range. The `steerai_data_collector` package provides a script to automatically drive the vehicle through various maneuvers.

### Maneuvers
The `data_collector.py` script executes a sequence of maneuvers:
1. **Warmup**: Constant speed driving.
2. **Chirp Signal (Low Speed)**: Sinusoidal steering with increasing frequency at low speed.
3. **Ramp Speed**: Linear acceleration and deceleration.
4. **Chirp Signal (High Speed)**: Sinusoidal steering at high speed.
5. **Step Inputs**: Step changes in steering angle.
6. **Random Walk**: Randomized speed and steering commands to explore the state space.

### Running the Data Collector
1. **Launch the Simulation:**
   ```bash
   roslaunch gem_gazebo gem_gazebo_rviz.launch
   ```

2. **Run the Data Collector:**
   ```bash
   rosrun steerai_data_collector data_collector.py
   ```

The data will be saved as a CSV file in `steerai_data_collector/data/` with a timestamped filename (e.g., `training_data_20231027-120000.csv`).

---

## 3. System Identification

We use a data-driven approach to model the vehicle's dynamics, specifically capturing the complex relationship between control inputs and the vehicle's state updates.

### Method
1. **Data Collection**: (See Section 2)
2. **Model Architecture**: A Feed-Forward Neural Network (MLP) is trained using PyTorch.
   - **Inputs**: `[current_speed, cmd_speed, cmd_steering_angle]`
   - **Outputs**: `[next_speed, delta_yaw]` (Change in yaw)
   - **Structure**: Input Layer (3) -> Hidden Layer (64, ReLU) -> Hidden Layer (64, ReLU) -> Output Layer (2)

### Validation & Accuracy
The model is validated on a hold-out test set (20% of collected data).

- **Metric**: Root Mean Squared Error (RMSE)
- **Typical Performance**:
  - Speed RMSE: 0.0442 m/s
  - Delta Yaw RMSE: 0.0194 rad

**Training Results:**
The training script generates plots showing the loss curve and prediction performance against ground truth.
*(See `steerai_sysid/training_results.png` after training)*

![Training Results](steerai_sysid/training_results.png)

To retrain the model:
```bash
rosrun steerai_sysid train_dynamics.py
```

---

## 4. MPC Controller Implementation

The `steerai_mpc` package implements a nonlinear MPC using **CasADi**.

### Hybrid Dynamics Model
To ensure stability at low speeds and accuracy at high speeds, the controller uses a **Hybrid Model** controlled by a blending parameter $\alpha$:

- **$\alpha = 0$ (Kinematic)**: Uses a Kinematic Bicycle Model. Ideal for low speeds.
- **$\alpha = 1$ (Neural Network)**: Uses the learned Neural Network Model. Ideal for high speeds where dynamics are complex.
- **Hybrid (Blended)**: A smooth linear blending is applied between the two models based on current velocity ($v_{low}$ to $v_{high}$).

$$
f_{hybrid}(x, u) = (1 - \alpha) \cdot f_{kin}(x, u) + \alpha \cdot f_{nn}(x, u)
$$

### Optimization Problem
The MPC solves the following optimization problem at 10 Hz with a prediction horizon of **20 steps** (2.0 seconds):

**Cost Function:**
Minimize $J = \sum_{k=0}^{T} (w_{pos} \cdot e_{pos}^2 + w_{head} \cdot (1 - \cos(e_{head})) + w_{vel} \cdot e_{vel}^2 + w_{steer} \cdot \Delta \delta^2 + w_{acc} \cdot \Delta v^2)$

- **Cross-Track Error ($e_{pos}$)**: Deviation from the reference path.
- **Heading Error ($e_{head}$)**: Deviation from the path's tangent (using robust cosine loss).
- **Speed Error ($e_{vel}$)**: Deviation from the target reference speed.
- **Control Effort**: Penalties on rapid changes in steering ($\Delta \delta$) and acceleration ($\Delta v$) for smoothness.

**Constraints:**
- **Actuator Limits**:
  - Speed: $[-5.5, 5.5]$ m/s
  - Steering Angle: $[-0.6, 0.6]$ rad
- **Dynamics**: The vehicle state must evolve according to the Hybrid Dynamics Model.

### Path Following
The controller receives a global path (waypoints) and uses a **KDTree** for efficient nearest-neighbor search to find the local reference trajectory at each time step.

---

## 4. Docker Support

You can run the entire simulation and control stack using Docker.

### Build the Image
Navigate to the project directory:
```bash
cd ~/catkin_ws/src/POLARIS_GEM_e2
```

Build the Docker image:
```bash
docker build -t steerai .
```

### Run the Container
To run with GUI support (Gazebo/RViz):

```bash
xhost +local:root # Allow docker to access X server
docker run -it --rm \
    --net=host \
    --privileged \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    steerai
```

### Optional: Enable GPU Support
If you have an NVIDIA GPU and the NVIDIA Container Toolkit installed, you can enable GPU acceleration for better performance:

```bash
xhost +local:root
docker run -it --rm \
    --net=host \
    --gpus all \
    --privileged \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    steerai
```

Inside the container, you can run the standard launch commands (in separate terminals):
```bash
# Terminal 1: Simulation
roslaunch gem_gazebo gem_gazebo_rviz.launch

# Terminal 2: MPC Controller
roslaunch steerai_mpc mpc_controller.launch

# Terminal 3: Path Publisher
rosrun steerai_mpc path_publisher.py _path_file:=paths/reference_path_modified.csv
```