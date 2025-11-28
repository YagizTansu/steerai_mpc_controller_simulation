#!/usr/bin/env python3

import rospy
import pandas as pd
import numpy as np
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_euler
import os

class PathPublisher:
    def __init__(self):
        rospy.init_node('path_publisher')
        
        # Parameters
        self.path_file = rospy.get_param('~path_file', '')
        self.frame_id = rospy.get_param('~frame_id', 'world')
        self.topic_name = rospy.get_param('~topic_name', '/gem/raw_path')
        
        if not self.path_file:
            # Try to find it in the package if not absolute
            # For now, just warn if empty
            rospy.logwarn("No path_file parameter provided. Waiting for parameter update or manual publish.")
        
        self.pub = rospy.Publisher(self.topic_name, Path, queue_size=1, latch=True)
        
        self.load_and_publish()
        
    def load_and_publish(self):
        if not self.path_file or not os.path.exists(self.path_file):
            rospy.logerr(f"Path file not found: {self.path_file}")
            return

        try:
            rospy.loginfo(f"Loading path from: {self.path_file}")
            
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
                rospy.logerr(f"Error parsing CSV: {e}")
                return

            # Create Path message
            msg = Path()
            msg.header.frame_id = self.frame_id
            msg.header.stamp = rospy.Time.now()
            
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
            rospy.loginfo(f"Published raw path with {len(points)} points to {self.topic_name}")
            
        except Exception as e:
            rospy.logerr(f"Failed to load/publish path: {e}")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = PathPublisher()
        node.run()
    except rospy.ROSInterruptException:
        pass
