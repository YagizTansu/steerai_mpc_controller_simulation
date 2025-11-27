#!/usr/bin/env python3

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry

class TFBroadcaster:
    def __init__(self):
        rospy.init_node('tf_broadcaster')
        
        self.br = tf2_ros.TransformBroadcaster()
        self.sub = rospy.Subscriber('/gem/base_footprint/odom', Odometry, self.odom_callback)
        
        rospy.loginfo("TF Broadcaster Started: world -> base_footprint")

    def odom_callback(self, msg):
        t = TransformStamped()
        
        # Use the timestamp from the message
        t.header.stamp = msg.header.stamp
        t.header.frame_id = "world"
        t.child_frame_id = "base_footprint"
        
        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z
        
        t.transform.rotation = msg.pose.pose.orientation
        
        self.br.sendTransform(t)

if __name__ == '__main__':
    try:
        TFBroadcaster()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
