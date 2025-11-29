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
from tf.transformations import euler_from_quaternion

class PathManager:
    def __init__(self, param_namespace='~path_manager'):
        """
        Initialize PathManager.
        Subscribes to raw path topic and processes it.
        """
        self.param_namespace = param_namespace
        
        # Load parameters
        self.load_parameters()
        
        self.path_data = None # [x, y, yaw]
        self.path_velocities = None # [v]
        self.tree = None
        self.path_seq = 0 # Sequence number for path updates
        self.target_speed = 5.0 # Default, will be updated by controller
        
        # Publisher for processed global path visualization
        if self.publish_path:
            self.global_path_pub = rospy.Publisher('/gem/global_path', Path, queue_size=1, latch=True)
            self.target_marker_pub = rospy.Publisher('/gem/target_point', Marker, queue_size=1)
        else:
            self.global_path_pub = None
            self.target_marker_pub = None
            
        # Subscriber for raw path
        self.raw_path_sub = rospy.Subscriber('/gem/raw_path', Path, self.raw_path_callback)
        rospy.loginfo("PathManager: Waiting for path on /gem/raw_path ...")
    
    def load_parameters(self):
        """Load parameters from ROS parameter server."""
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
        
        # Velocity Profile Parameters
        # Loaded from path_params.yaml
        self.a_lat_max = rospy.get_param(
            self.param_namespace + '/velocity_profile/a_lat_max', 1.0)
        self.smoothing_window = rospy.get_param(
            self.param_namespace + '/velocity_profile/smoothing_window', 20)
        self.v_min = rospy.get_param(
            self.param_namespace + '/velocity_profile/v_min', 1.0)
        
        rospy.loginfo(f"PathManager Params: a_lat_max={self.a_lat_max}, window={self.smoothing_window}, v_min={self.v_min}")
        
        # Visualization parameters
        self.frame_id = rospy.get_param(
            self.param_namespace + '/visualization/frame_id', 'world')
        self.publish_path = rospy.get_param(
            self.param_namespace + '/visualization/publish_path', True)
            
    def set_target_speed(self, speed):
        """Update target speed and recalculate profile if path exists."""
        self.target_speed = speed
        if self.path_data is not None:
            self.calculate_velocity_profile()

    def raw_path_callback(self, msg):
        """Callback for raw path message."""
        rospy.loginfo(f"PathManager: Received new path with {len(msg.poses)} points.")
        
        try:
            # Extract points
            points = []
            for pose in msg.poses:
                points.append([pose.pose.position.x, pose.pose.position.y])
            
            points = np.array(points)
            
            # Process the path
            self.process_path(points)
            
        except Exception as e:
            rospy.logerr(f"PathManager: Failed to process incoming path: {e}")

    def process_path(self, points):
        """
        Removes duplicates, interpolates, and builds KDTree from points array.
        """
        if len(points) < 2:
            rospy.logerr("Path must have at least 2 points.")
            return

        try:
            # 1. Cleaning: Remove duplicate points
            diffs = np.diff(points, axis=0)
            dists = np.linalg.norm(diffs, axis=1)
            
            clean_points = [points[0]]
            for i in range(len(dists)):
                if dists[i] > self.duplicate_threshold:
                    clean_points.append(points[i+1])
            points = np.array(clean_points)

            if len(points) < 2:
                rospy.logerr("Path has too few points after cleaning.")
                return

            # 2. Interpolation (B-spline)
            # splprep requires a list of arrays [x_coords, y_coords]
            tck, u = splprep(points.T, s=self.interpolation_smoothness, k=self.interpolation_degree)
            
            # Generate dense path
            total_dist = np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
            num_points = int(total_dist / self.interpolation_resolution)
            u_new = np.linspace(0, 1, num_points)
            
            x_new, y_new = splev(u_new, tck)
            
            # 3. Calculate Yaw
            dx = np.gradient(x_new)
            dy = np.gradient(y_new)
            yaw_new = np.arctan2(dy, dx)
            
            # Store as [x, y, yaw]
            self.path_data = np.column_stack((x_new, y_new, yaw_new))
            
            # Build KDTree
            self.tree = KDTree(self.path_data[:, :2])
            
            # Calculate Velocity Profile
            self.calculate_velocity_profile()
            
            # Publish Processed Path
            if self.global_path_pub is not None:
                self.publish_global_path()
            
            self.path_seq += 1
            rospy.loginfo(f"Path processed successfully. Interpolated points: {len(self.path_data)} (Seq: {self.path_seq})")

        except Exception as e:
            rospy.logerr(f"Failed to process path: {e}")

    def get_path_data(self):
        return self.path_data

    def get_path_seq(self):
        return self.path_seq

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

    def calculate_velocity_profile(self):
        """
        Calculates a velocity profile for the path based on curvature.
        Uses self.target_speed.
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
        dists = np.maximum(dists, 1e-6)
        
        # Calculate curvature
        dyaw = np.diff(yaw)
        dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi
        curvature = np.abs(dyaw / dists)
        curvature = np.append(curvature, curvature[-1])
        
        # Calculate max velocity
        # Use parameter for lateral acceleration limit
        v_profile = np.sqrt(self.a_lat_max / (curvature + 1e-6))
        
        # Clamp velocities
        # Use parameter for minimum velocity
        v_profile = np.clip(v_profile, self.v_min, self.target_speed)
        
        # Smooth
        # Use parameter for window size
        window_size = self.smoothing_window
        kernel = np.ones(window_size) / window_size
        v_profile = np.convolve(v_profile, kernel, mode='same')
        
        # Deceleration at the end
        decel_dist = 10.0 # meters
        final_speed = 0.0
        
        # Calculate cumulative distance from end
        dist_from_end = 0.0
        for i in range(len(self.path_data) - 1, -1, -1):
            if i < len(self.path_data) - 1:
                p1 = self.path_data[i]
                p2 = self.path_data[i+1]
                d = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                dist_from_end += d
            
            if dist_from_end > decel_dist:
                break
            
            # Linear ramp down
            ratio = dist_from_end / decel_dist
            target_v = final_speed + (self.target_speed - final_speed) * ratio
            
            # Take minimum of existing profile and deceleration ramp
            v_profile[i] = min(v_profile[i], target_v)

        self.path_velocities = v_profile
        rospy.loginfo(f"Velocity profile generated with max speed {self.target_speed:.2f} m/s and deceleration over last {decel_dist}m.")
        
        # Print details
        self.print_path_details()
        
    def get_reference(self, robot_x, robot_y, horizon_size, dt):
        """
        Finds the nearest point and returns the next N points.
        """
        if self.tree is None:
            return np.zeros((horizon_size, 4))

        # Query KDTree
        dist, idx = self.tree.query([robot_x, robot_y])
        
        ref_traj_list = []
        curr_idx = idx
        
        # First point
        first_pt = self.path_data[curr_idx]
        first_v = self.path_velocities[curr_idx] if self.path_velocities is not None else self.target_speed
        ref_traj_list.append(np.append(first_pt, first_v))
        
        # Subsequent points
        for _ in range(horizon_size - 1):
            current_ref_v = self.path_velocities[curr_idx] if self.path_velocities is not None else self.target_speed
            target_dist = dt * current_ref_v
            
            accumulated_dist = 0.0
            
            while accumulated_dist < target_dist and curr_idx < len(self.path_data) - 1:
                p1 = self.path_data[curr_idx]
                p2 = self.path_data[curr_idx + 1]
                seg_dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                
                accumulated_dist += seg_dist
                curr_idx += 1
            
            pt = self.path_data[curr_idx]
            v = self.path_velocities[curr_idx] if self.path_velocities is not None else self.target_speed
            ref_traj_list.append(np.append(pt, v))
            
        ref_path = np.array(ref_traj_list)
        
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

    def print_path_details(self):
        """Prints a formatted summary of the path and velocity profile."""
        if self.path_data is None or self.path_velocities is None:
            return

        print("\n" + "="*95)
        print(f"PATH PROFILE SUMMARY ({len(self.path_data)} points) - Sampled")
        print("="*95)
        print(f"{'Idx':<6} | {'X (m)':<10} | {'Y (m)':<10} | {'Yaw (rad)':<10} | {'Speed (m/s)':<12} | {'Curvature':<10} | {'Note'}")
        print("-" * 95)
        
        # Re-calculate curvature for display
        x = self.path_data[:, 0]
        y = self.path_data[:, 1]
        yaw = self.path_data[:, 2]
        
        dx = np.diff(x)
        dy = np.diff(y)
        dists = np.sqrt(dx**2 + dy**2)
        dists = np.maximum(dists, 1e-6)
        
        dyaw = np.diff(yaw)
        dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi
        curvature = np.abs(dyaw / dists)
        curvature = np.append(curvature, 0.0) # Pad last
        
        # Show points (Sampled)
        step = max(1, len(self.path_data) // 40) # Show ~40 points
        
        for i in range(0, len(self.path_data), step):
            note = ""
            if i == 0: note = "START"
            elif i >= len(self.path_data) - step: note = "END"
            elif self.path_velocities[i] < self.target_speed - 0.5: note = "SLOW (Turn)"
            
            print(f"{i:<6} | {x[i]:<10.2f} | {y[i]:<10.2f} | {yaw[i]:<10.2f} | {self.path_velocities[i]:<12.2f} | {curvature[i]:<10.4f} | {note}")
            
        # Always print the last point
        last = len(self.path_data) - 1
        if last % step != 0:
             print(f"{last:<6} | {x[last]:<10.2f} | {y[last]:<10.2f} | {yaw[last]:<10.2f} | {self.path_velocities[last]:<12.2f} | {curvature[last]:<10.4f} | END")

        print("="*95 + "\n")

if __name__ == '__main__':
    try:
        rospy.init_node('path_manager_node')
        pm = PathManager()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass