#!/usr/bin/env python3

import rospy
import numpy as np
import pandas as pd
import os
from scipy.interpolate import splprep, splev
from scipy.spatial import KDTree
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA

class PathManager:
    def __init__(self, path_file):
        """
        Initialize PathManager.
        :param path_file: Absolute path to the CSV file.
        """
        self.path_file = path_file
        self.path_data = None # [x, y, yaw, v]
        self.tree = None
        
        # Publishers
        self.global_path_pub = rospy.Publisher('/gem/global_path', Path, queue_size=1, latch=True)
        self.target_marker_pub = rospy.Publisher('/gem/target_point', Marker, queue_size=1)
        
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
            
            # 6. Build KDTree
            self.tree = KDTree(self.path_data[:, :2])
            
            rospy.loginfo(f"Path loaded successfully. Original: {len(points)}, Interpolated: {len(self.path_data)}")
            rospy.loginfo(f"First 5 points: {self.path_data[:5, :2]}")

        except Exception as e:
            rospy.logerr(f"Failed to process path: {e}")
            import traceback
            traceback.print_exc()

    def get_reference(self, robot_x, robot_y, horizon_size):
        """
        Finds the nearest point and returns the next N points.
        :param robot_x: Robot X position
        :param robot_y: Robot Y position
        :param horizon_size: Number of points to return
        :return: Numpy array of shape (horizon_size, 4) -> [x, y, yaw, v]
        """
        if self.tree is None:
            return np.zeros((horizon_size, 4))

        # Query KDTree for nearest neighbor
        dist, idx = self.tree.query([robot_x, robot_y])
        
        # Get slice
        end_idx = idx + horizon_size
        
        if end_idx < len(self.path_data):
            ref_path = self.path_data[idx:end_idx]
        else:
            # We are near the end, take what's left and pad
            ref_path = self.path_data[idx:]
            rows_missing = horizon_size - len(ref_path)
            if rows_missing > 0:
                # Pad with the last point
                last_point = ref_path[-1]
                padding = np.tile(last_point, (rows_missing, 1))
                ref_path = np.vstack((ref_path, padding))
        
        # Visualize the target point (the first point in the reference)
        self.publish_target_marker(ref_path[0])
        
        return ref_path

    def get_cross_track_error(self, robot_x, robot_y):
        """
        Calculates perpendicular distance to the nearest path point.
        """
        if self.tree is None:
            return 0.0
            
        dist, idx = self.tree.query([robot_x, robot_y])
        
        # To determine sign (left or right), we can use the cross product
        # Vector from path point to robot
        path_pt = self.path_data[idx]
        dx = robot_x - path_pt[0]
        dy = robot_y - path_pt[1]
        
        # Path tangent (yaw)
        yaw = path_pt[2]
        
        # Rotate robot vector by -yaw to align with path frame
        # If y component is positive, it's to the left (positive CTE), else right.
        # Alternatively: Cross product of Path Tangent and Vector to Robot
        # Tangent = [cos(yaw), sin(yaw)]
        # Vector = [dx, dy]
        # Cross = cos(yaw)*dy - sin(yaw)*dx
        
        cross_prod = np.cos(yaw) * dy - np.sin(yaw) * dx
        
        return cross_prod # Signed distance

    def publish_global_path(self):
        msg = Path()
        msg.header.frame_id = "odom"
        msg.header.stamp = rospy.Time.now()
        
        for pt in self.path_data:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = pt[0]
            pose.pose.position.y = pt[1]
            # Orientation is not strictly needed for the line strip, but good to have
            # We could convert yaw to quaternion if needed, but for Path visualization 
            # usually position is enough.
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
            
        self.global_path_pub.publish(msg)

    def publish_target_marker(self, target_pt):
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "target_point"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = target_pt[0]
        marker.pose.position.y = target_pt[1]
        marker.pose.position.z = 0.5 # Slightly elevated
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        marker.color = ColorRGBA(1.0, 0.0, 0.0, 1.0) # Red
        
        self.target_marker_pub.publish(marker)

if __name__ == '__main__':
    rospy.init_node('path_manager_test')
    
    # Create a dummy CSV for testing
    import tempfile
    
    # Create a figure-8 path or circle
    t = np.linspace(0, 2*np.pi, 50)
    x = 10 * np.cos(t)
    y = 10 * np.sin(t)
    
    # Add some duplicates to test cleaning
    x = np.insert(x, 10, x[10])
    y = np.insert(y, 10, y[10])
    
    df = pd.DataFrame({'x': x, 'y': y})
    
    # Use a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        df.to_csv(tmp.name, index=False)
        tmp_path = tmp.name
        
    rospy.loginfo(f"Created dummy path at {tmp_path}")
    
    try:
        pm = PathManager(tmp_path)
        
        rate = rospy.Rate(10)
        # Simulate a robot moving near the path
        sim_idx = 0
        
        while not rospy.is_shutdown():
            # Fake robot position (just following the generated path for test)
            if pm.path_data is not None:
                # Move index
                sim_idx = (sim_idx + 1) % len(pm.path_data)
                rx, ry = pm.path_data[sim_idx, 0], pm.path_data[sim_idx, 1]
                
                # Add some noise
                rx += 0.5
                ry -= 0.2
                
                ref = pm.get_reference(rx, ry, 10)
                cte = pm.get_cross_track_error(rx, ry)
                
                rospy.loginfo_throttle(1, f"Robot: ({rx:.2f}, {ry:.2f}) | CTE: {cte:.3f} | Target: ({ref[0,0]:.2f}, {ref[0,1]:.2f})")
                
                # Republish global path occasionally to ensure it shows up in RViz
                pm.publish_global_path()
                
            rate.sleep()
            
    except rospy.ROSInterruptException:
        pass
    finally:
        os.remove(tmp_path)
