#!/usr/bin/env python3

import rospy
import numpy as np
import pandas as pd
import os
from scipy.interpolate import splprep, splev
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

class PathManager:
    def __init__(self, path_file):
        """
        Initialize PathManager.
        :param path_file: Absolute path to the CSV file.
        """
        self.path_file = path_file
        self.path_data = None # [x, y, yaw, v]
        
        # Publisher for global path visualization
        self.global_path_pub = rospy.Publisher('/gem/global_path', Path, queue_size=1, latch=True)
        
        # Load and process path
        self.load_and_process_path()
        
        # Publish global path once ready
        if self.path_data is not None:
            self.publish_global_path()

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
            # Keep first point, and any point where distance to previous > 0.01
            # We reconstruct the array.
            clean_points = [points[0]]
            for i in range(len(dists)):
                if dists[i] > 0.01:
                    clean_points.append(points[i+1])
            points = np.array(clean_points)

            if len(points) < 2:
                rospy.logerr("Path has too few points after cleaning.")
                return

            # 3. Interpolation (B-spline)
            # splprep requires a list of arrays [x_coords, y_coords]
            # s=0 means interpolate through all points (no smoothing error allowed), 
            # or small s for smoothing. Let's use s=0 for strict tracking or small value.
            # k=3 is cubic B-spline.
            tck, u = splprep(points.T, s=0.0, k=3)
            
            # Generate dense path
            # Calculate total length to estimate number of points for 0.1m resolution
            # Simple Euclidean sum
            total_dist = np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
            num_points = int(total_dist / 0.1)
            u_new = np.linspace(0, 1, num_points)
            
            x_new, y_new = splev(u_new, tck)
            
            # 4. Calculate Yaw
            # np.arctan2(dy, dx)
            dx = np.gradient(x_new)
            dy = np.gradient(y_new)
            yaw_new = np.arctan2(dy, dx)
            
            # 5. Target Speed
            v_new = np.full_like(x_new, 2.0) # Constant 2.0 m/s
            
            # Store as [x, y, yaw, v]
            self.path_data = np.column_stack((x_new, y_new, yaw_new, v_new))
            
            rospy.loginfo(f"Path loaded and smoothed successfully.")
            rospy.loginfo(f"Original points: {len(points)}, Interpolated points: {len(self.path_data)}")

        except Exception as e:
            rospy.logerr(f"Failed to process path: {e}")

    def get_path_data(self):
        """
        Returns the processed path data.
        :return: Numpy array of shape (N, 4) -> [x, y, yaw, v] or None if not loaded
        """
        return self.path_data

    def publish_global_path(self):
        msg = Path()
        msg.header.frame_id = "world"
        msg.header.stamp = rospy.Time.now()
        
        for pt in self.path_data:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = pt[0]
            pose.pose.position.y = pt[1]
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
            
        self.global_path_pub.publish(msg)