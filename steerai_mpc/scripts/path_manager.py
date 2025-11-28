#!/usr/bin/env python3

import rospy
import numpy as np
import pandas as pd
import os
from scipy.interpolate import splprep, splev
from scipy.spatial import KDTree
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA

class PathManager:
    def __init__(self, path_file=None, param_namespace='~path_manager'):
        """
        Initialize PathManager.
        :param path_file: Absolute path to the CSV file (overrides parameter server)
        :param param_namespace: ROS parameter namespace (default: '~path_manager')
        """
        # Load parameters from parameter server
        self.param_namespace = param_namespace
        self.load_parameters(path_file)
        
        self.path_data = None # [x, y, yaw]
        self.path_velocities = None # [v]
        self.tree = None
        
        # Publisher for global path visualization
        if self.publish_path:
            self.global_path_pub = rospy.Publisher('/gem/global_path', Path, queue_size=1, latch=True)
            self.target_marker_pub = rospy.Publisher('/gem/target_point', Marker, queue_size=1)
        else:
            self.global_path_pub = None
            self.target_marker_pub = None
        
        # Load and process path
        self.load_and_process_path()
        
        # Publish global path once ready
        if self.path_data is not None and self.global_path_pub is not None:
            self.publish_global_path()
    
    def load_parameters(self, path_file_override=None):
        """
        Load parameters from ROS parameter server with fallback defaults.
        :param path_file_override: If provided, overrides parameter server path
        """
        # Path file
        if path_file_override is not None:
            self.path_file = path_file_override
        else:
            # Try to get from parameter server
            param_name = self.param_namespace + '/path/file'
            if rospy.has_param(param_name):
                self.path_file = rospy.get_param(param_name)
                # Expand ROS package paths like $(find package_name)
                import re
                match = re.search(r'\$\(find ([^\)]+)\)', self.path_file)
                if match:
                    package_name = match.group(1)
                    import rospkg
                    rospack = rospkg.RosPack()
                    try:
                        package_path = rospack.get_path(package_name)
                        self.path_file = self.path_file.replace(
                            f'$(find {package_name})', package_path
                        )
                    except rospkg.ResourceNotFound:
                        rospy.logerr(f"Package {package_name} not found!")
            else:
                rospy.logwarn(f"Parameter {param_name} not found, no path file specified!")
                self.path_file = None
        
        # Interpolation parameters
        self.interpolation_smoothness = rospy.get_param(
            self.param_namespace + '/interpolation/smoothness', 0.0)
        self.interpolation_degree = rospy.get_param(
            self.param_namespace + '/interpolation/degree', 3)
        self.interpolation_resolution = rospy.get_param(
            self.param_namespace + '/interpolation/resolution', 0.1)
        
        # Processing parameters
        self.duplicate_threshold = rospy.get_param(
            self.param_namespace + '/processing/duplicate_threshold', 0.01)
        
        # Visualization parameters
        self.frame_id = rospy.get_param(
            self.param_namespace + '/visualization/frame_id', 'world')
        self.publish_path = rospy.get_param(
            self.param_namespace + '/visualization/publish_path', True)
        
        # Validate parameters
        self._validate_parameters()
        
        rospy.loginfo(f"PathManager parameters loaded from {self.param_namespace}")
    
    def _validate_parameters(self):
        """Validate loaded parameters and warn if out of range."""
        # Validate interpolation smoothness
        if not (0.0 <= self.interpolation_smoothness <= 10.0):
            rospy.logwarn(f"interpolation_smoothness={self.interpolation_smoothness} "
                         f"out of range [0.0, 10.0], clamping")
            self.interpolation_smoothness = max(0.0, min(10.0, self.interpolation_smoothness))
        
        # Validate interpolation degree
        if not (1 <= self.interpolation_degree <= 5):
            rospy.logwarn(f"interpolation_degree={self.interpolation_degree} "
                         f"out of range [1, 5], clamping")
            self.interpolation_degree = max(1, min(5, self.interpolation_degree))
        
        # Validate interpolation resolution
        if not (0.01 <= self.interpolation_resolution <= 1.0):
            rospy.logwarn(f"interpolation_resolution={self.interpolation_resolution} "
                         f"out of range [0.01, 1.0], clamping")
            self.interpolation_resolution = max(0.01, min(1.0, self.interpolation_resolution))
        
        # Validate duplicate threshold
        if not (0.0 <= self.duplicate_threshold <= 0.5):
            rospy.logwarn(f"duplicate_threshold={self.duplicate_threshold} "
                         f"out of range [0.0, 0.5], clamping")
            self.duplicate_threshold = max(0.0, min(0.5, self.duplicate_threshold))


    def load_and_process_path(self):
        """
        Loads CSV, removes duplicates, interpolates, and builds KDTree.
        """
        if not os.path.exists(self.path_file):
            rospy.logerr(f"Path file not found: {self.path_file}")
            return

        try:
            # 1. Load Data
            # Handle headers or no headers. Assume if 'x' is in columns, it has headers.
            # Otherwise assume first two columns are x, y.
            try:
                df = pd.read_csv(self.path_file)
                
                # Check for various column names
                if 'x' in df.columns and 'y' in df.columns:
                    points = df[['x', 'y']].values
                elif 'curr_x' in df.columns and 'curr_y' in df.columns:
                    points = df[['curr_x', 'curr_y']].values
                else:
                    # If headers don't match known names, try reading without header
                    # But first check if the first row looks like strings
                    df_no_header = pd.read_csv(self.path_file, header=None)
                    # Try to convert to float
                    try:
                        points = df_no_header.iloc[:, 0:2].astype(float).values
                    except ValueError:
                        # If conversion fails, maybe the first row was a header we didn't recognize
                        # Try skipping the first row
                        points = df_no_header.iloc[1:, 0:2].astype(float).values
                        
                # Ensure points are float
                points = np.array(points, dtype=float)
                
            except Exception as e:
                rospy.logerr(f"Error reading CSV: {e}")
                return

            if len(points) < 2:
                rospy.logerr("Path must have at least 2 points.")
                return

            # 2. Cleaning: Remove duplicate points
            # Calculate distance between consecutive points
            diffs = np.diff(points, axis=0)
            dists = np.linalg.norm(diffs, axis=1)
            # Keep first point, and any point where distance to previous > duplicate_threshold
            # We reconstruct the array.
            clean_points = [points[0]]
            for i in range(len(dists)):
                if dists[i] > self.duplicate_threshold:
                    clean_points.append(points[i+1])
            points = np.array(clean_points)

            if len(points) < 2:
                rospy.logerr("Path has too few points after cleaning.")
                return

            # 3. Interpolation (B-spline)
            # splprep requires a list of arrays [x_coords, y_coords]
            # Use loaded interpolation parameters
            tck, u = splprep(points.T, s=self.interpolation_smoothness, k=self.interpolation_degree)
            
            # Generate dense path
            # Calculate total length to estimate number of points based on resolution
            # Simple Euclidean sum
            total_dist = np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
            num_points = int(total_dist / self.interpolation_resolution)
            u_new = np.linspace(0, 1, num_points)
            
            x_new, y_new = splev(u_new, tck)
            
            # 4. Calculate Yaw
            # np.arctan2(dy, dx)
            dx = np.gradient(x_new)
            dy = np.gradient(y_new)
            yaw_new = np.arctan2(dy, dx)
            
            # Store as [x, y, yaw]
            self.path_data = np.column_stack((x_new, y_new, yaw_new))
            
            # Build KDTree
            self.tree = KDTree(self.path_data[:, :2])
            
            rospy.loginfo(f"Path loaded and smoothed successfully.")
            rospy.loginfo(f"Original points: {len(points)}, Interpolated points: {len(self.path_data)}")

        except Exception as e:
            rospy.logerr(f"Failed to process path: {e}")

    def get_path_data(self):
        """
        Returns the processed path data.
        :return: Numpy array of shape (N, 3) -> [x, y, yaw] or None if not loaded
        """
        return self.path_data

    def publish_global_path(self):
        msg = Path()
        msg.header.frame_id = self.frame_id
        msg.header.stamp = rospy.Time.now()
        
        for pt in self.path_data:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = pt[0]
            pose.pose.position.y = pt[1]
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
            
        self.global_path_pub.publish(msg)

    def calculate_velocity_profile(self, target_speed):
        """
        Calculates a velocity profile for the path based on curvature.
        v = sqrt(a_lat_max / curvature)
        """
        if self.path_data is None or len(self.path_data) < 3:
            return
            
        # Extract path components
        x = self.path_data[:, 0]
        y = self.path_data[:, 1]
        yaw = self.path_data[:, 2]
        
        # Calculate distances between points
        dx = np.diff(x)
        dy = np.diff(y)
        dists = np.sqrt(dx**2 + dy**2)
        dists = np.maximum(dists, 1e-6) # Avoid division by zero
        
        # Calculate curvature (change in yaw per meter)
        # Note: yaw is already in radians
        dyaw = np.diff(yaw)
        
        # Handle yaw wrapping (-pi to pi)
        dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi
        
        curvature = np.abs(dyaw / dists)
        
        # Pad curvature to match length
        curvature = np.append(curvature, curvature[-1])
        
        # Calculate max velocity based on lateral acceleration limit
        # a_lat = v^2 * k  ->  v = sqrt(a_lat / k)
        a_lat_max = 1.0  # m/s^2 (Conservative limit for stability)
        
        v_profile = np.sqrt(a_lat_max / (curvature + 1e-6))
        
        # Clamp velocities
        v_min = 1.0 # Minimum speed to prevent stopping
        v_max = target_speed
        
        v_profile = np.clip(v_profile, v_min, v_max)
        
        # Smooth the velocity profile (Moving average)
        window_size = 5
        kernel = np.ones(window_size) / window_size
        v_profile = np.convolve(v_profile, kernel, mode='same')
        
        self.path_velocities = v_profile
        rospy.loginfo("Velocity profile generated based on curvature.")
        
    def get_reference(self, robot_x, robot_y, horizon_size, dt, target_speed):
        """
        Finds the nearest point and returns the next N points with MPC's target velocity.
        """
        if self.tree is None:
            return np.zeros((horizon_size, 4))

        # Query KDTree for nearest neighbor
        dist, idx = self.tree.query([robot_x, robot_y])
        
        # Initialize reference trajectory list
        ref_traj_list = []
        
        # Start from the nearest point
        curr_idx = idx
        
        # Add the first point (k=0)
        first_pt = self.path_data[curr_idx]
        first_v = self.path_velocities[curr_idx] if self.path_velocities is not None else target_speed
        ref_traj_list.append(np.append(first_pt, first_v))
        
        # Loop to find subsequent points based on distance
        for _ in range(horizon_size - 1):
            # Determine distance to travel for this step: dt * v
            current_ref_v = self.path_velocities[curr_idx] if self.path_velocities is not None else target_speed
            target_dist = dt * current_ref_v
            
            accumulated_dist = 0.0
            
            # Advance along the path until we cover the target distance
            while accumulated_dist < target_dist and curr_idx < len(self.path_data) - 1:
                p1 = self.path_data[curr_idx]
                p2 = self.path_data[curr_idx + 1]
                seg_dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                
                accumulated_dist += seg_dist
                curr_idx += 1
            
            # Add the point we landed on
            pt = self.path_data[curr_idx]
            v = self.path_velocities[curr_idx] if self.path_velocities is not None else target_speed
            ref_traj_list.append(np.append(pt, v))
            
        # Convert to numpy array
        ref_path = np.array(ref_traj_list) # Shape: (horizon_size, 4)
        
        # Visualize the target point
        self.publish_target_marker(ref_path[0])
        
        return ref_path

    def publish_target_marker(self, target_pt):
        if self.target_marker_pub is None:
            return
            
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "target_point"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = target_pt[0]
        marker.pose.position.y = target_pt[1]
        marker.pose.position.z = 0.5
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        marker.color = ColorRGBA(1.0, 0.0, 0.0, 1.0)
        
        self.target_marker_pub.publish(marker)

    def get_cross_track_error(self, robot_x, robot_y):
        if self.tree is None:
            return 0.0
            
        dist, idx = self.tree.query([robot_x, robot_y])
        
        path_pt = self.path_data[idx]
        dx = robot_x - path_pt[0]
        dy = robot_y - path_pt[1]
        yaw = path_pt[2]
        
        cross_prod = np.cos(yaw) * dy - np.sin(yaw) * dx
        return cross_prod

    def is_goal_reached(self, robot_x, robot_y, tolerance):
        if self.path_data is None or len(self.path_data) == 0:
            return False
        
        final_point = self.path_data[-1, :2]
        dist = np.sqrt((robot_x - final_point[0])**2 + (robot_y - final_point[1])**2)
        
        return dist < tolerance

    def calculate_remaining_distance(self, robot_x, robot_y):
        if self.tree is None or self.path_data is None:
            return 0.0
        
        dist, idx = self.tree.query([robot_x, robot_y])
        
        remaining = 0.0
        for i in range(idx, len(self.path_data) - 1):
            dx = self.path_data[i+1, 0] - self.path_data[i, 0]
            dy = self.path_data[i+1, 1] - self.path_data[i, 1]
            remaining += np.sqrt(dx**2 + dy**2)
        
        remaining += dist
        return remaining