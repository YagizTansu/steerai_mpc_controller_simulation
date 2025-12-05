#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
import pandas as pd
import numpy as np
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
try:
    from tf_transformations import quaternion_from_euler
except ImportError:
    def quaternion_from_euler(ai, aj, ak):
        ai /= 2.0
        aj /= 2.0
        ak /= 2.0
        ci = np.cos(ai)
        si = np.sin(ai)
        cj = np.cos(aj)
        sj = np.sin(aj)
        ck = np.cos(ak)
        sk = np.sin(ak)
        cc = ci*ck
        cs = ci*sk
        sc = si*ck
        ss = si*sk

        q = np.empty((4, ))
        q[0] = cj*sc - sj*cs
        q[1] = cj*ss + sj*cc
        q[2] = cj*cs - sj*sc
        q[3] = cj*cc + sj*ss

        return q

from ament_index_python.packages import get_package_share_directory
import os

class PathPublisher(Node):
    def __init__(self):
        super().__init__('path_publisher')
        
        self.logger = self.get_logger()
        
        # Parameters
        self.declare_parameter('path_file', 'paths/steerai_path.csv')
        self.declare_parameter('frame_id', 'world')
        self.declare_parameter('topic_name', '/gem/raw_path')
        
        self.path_file = self.get_parameter('path_file').value
        self.frame_id = self.get_parameter('frame_id').value
        self.topic_name = self.get_parameter('topic_name').value
        
        if not self.path_file:
            self.logger.warn("No path_file parameter provided.")
            return

        # Resolve path if relative
        if not os.path.isabs(self.path_file):
            try:
                pkg_path = get_package_share_directory('steerai_mpc')
                
                # Handle case where default included package name
                if self.path_file.startswith('steerai_mpc/'):
                    self.path_file = self.path_file.replace('steerai_mpc/', '', 1)
                    
                self.path_file = os.path.join(pkg_path, self.path_file)
            except Exception as e:
                self.logger.error(f"Could not resolve package path: {e}")
        
        qos = QoSProfile(depth=1)
        self.pub = self.create_publisher(Path, self.topic_name, qos)
        
        self.load_and_publish()
        
    def load_and_publish(self):
        if not self.path_file or not os.path.exists(self.path_file):
            self.logger.error(f"Path file not found: {self.path_file}")
            return

        try:
            self.logger.info(f"Loading path from: {self.path_file}")
            
            # Read CSV
            # Support multiple formats as in the original manager
            try:
                df = pd.read_csv(self.path_file)
                
                if 'x' in df.columns and 'y' in df.columns:
                    points = df[['x', 'y']].values
                    yaws = df['yaw'].values if 'yaw' in df.columns else np.zeros(len(df))
                elif 'curr_x' in df.columns and 'curr_y' in df.columns:
                    points = df[['curr_x', 'curr_y']].values
                    yaws = df['curr_yaw'].values if 'curr_yaw' in df.columns else np.zeros(len(df))
                else:
                    # No header fallback
                    df_no_header = pd.read_csv(self.path_file, header=None)
                    points = df_no_header.iloc[:, 0:2].astype(float).values
                    yaws = np.zeros(len(df_no_header))
                    
            except Exception as e:
                self.logger.error(f"Error parsing CSV: {e}")
                return

            # Create Path message
            msg = Path()
            msg.header.frame_id = self.frame_id
            msg.header.stamp = self.get_clock().now().to_msg()
            
            for i in range(len(points)):
                pose = PoseStamped()
                pose.header = msg.header
                pose.pose.position.x = points[i][0]
                pose.pose.position.y = points[i][1]
                pose.pose.position.z = 0.0
                
                # Convert yaw to quaternion
                q = quaternion_from_euler(0, 0, yaws[i])
                pose.pose.orientation.x = q[0]
                pose.pose.orientation.y = q[1]
                pose.pose.orientation.z = q[2]
                pose.pose.orientation.w = q[3]
                
                msg.poses.append(pose)
            
            self.pub.publish(msg)
            self.logger.info(f"Published raw path with {len(points)} points to {self.topic_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to load/publish path: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = PathPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
