#!/usr/bin/env python3

import rospy
import numpy as np
import pandas as pd
import os
from scipy.interpolate import splprep, splev
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

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
        
        self.path_data = None # [x, y, yaw, v]
        
        # Publisher for global path visualization
        if self.publish_path:
            self.global_path_pub = rospy.Publisher('/gem/global_path', Path, queue_size=1, latch=True)
        else:
            self.global_path_pub = None
        
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
        
        # NOTE: target_speed removed - now controlled by MPC Controller
        
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
        
        # NOTE: target_speed validation removed - now in MPC Controller


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
            
            # Store as [x, y, yaw] - velocity will be added by MPC Controller
            self.path_data = np.column_stack((x_new, y_new, yaw_new))
            
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