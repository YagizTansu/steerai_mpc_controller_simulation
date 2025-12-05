from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('steerai_mpc')
    
    # Configuration files
    mpc_config = os.path.join(pkg_share, 'config', 'mpc_params.yaml')
    path_config = os.path.join(pkg_share, 'config', 'path_params.yaml')

    return LaunchDescription([
        Node(
            package='steerai_mpc',
            executable='mpc_controller',
            name='mpc_controller',
            output='screen',
            parameters=[
                mpc_config,
                path_config
            ]
        ),
        Node(
            package='steerai_mpc',
            executable='tf_broadcaster',
            name='tf_broadcaster',
            output='screen'
        )
    ])
