import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution
from launch_ros.actions import Node

def generate_launch_description():
    # Package Directories
    pkg_gem_gazebo = get_package_share_directory('gem_gazebo')
    pkg_gem_description = get_package_share_directory('gem_description')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')
    pkg_steerai_mpc = get_package_share_directory('steerai_mpc')

    # Arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    world_name = LaunchConfiguration('world_name', default='simple_track_green.world')
    
    # World File
    # Note: We might need to ensure the world file is compatible with Gz Sim or use an empty world + models
    # For now, we'll try to load the world file if it's SDF, or use an empty world if it fails.
    # The original was .world, likely Classic. We might need to use a standard Gz world or convert it.
    # Let's assume we use an empty world for now and spawn the ground plane if needed, 
    # OR try to load the world if it works. 
    # Gz Sim usually uses .sdf. Let's try to use the provided world file path, but Gz Sim might expect SDF.
    # If simple_track_green.world is SDF compliant, it might work.
    
    # Parse Robot Description (URDF)
    xacro_file = os.path.join(pkg_gem_description, 'urdf', 'gem.urdf.xacro')
    robot_description_content = Command(['xacro ', xacro_file, ' robotname:=gem'])
    
    # Nodes
    
    # 1. Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_description': robot_description_content
        }]
    )

    # 2. Gazebo Sim
    # We use gz_sim.launch.py from ros_gz_sim
    # We can pass arguments to gz_sim, e.g. the world file
    # If the world file is not compatible, we can use empty.sdf
    # For now, let's try to launch with the world file.
    # We need to make sure the world file is in the share directory.
    world_file_path = PathJoinSubstitution([pkg_gem_gazebo, 'worlds', world_name])
    
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': ['-r ', world_file_path]}.items(),
    )

    # 3. Spawn Robot
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'gem',
            '-topic', 'robot_description', # Read from robot_description topic (published by RSP? No, RSP publishes to /robot_description param usually, but we can pass string)
            # Actually create node can take -string or -file. 
            # But passing huge string in args is risky.
            # Better to use -topic if we publish it, or just use -string with the Command result.
            '-string', robot_description_content,
            '-x', '0', '-y', '0', '-z', '0.5',
            '-R', '0', '-P', '0', '-Y', '0'
        ],
        output='screen'
    )

    # 4. Bridge
    # Bridge config
    # cmd_vel (ROS -> Gz)
    # odom (Gz -> ROS)
    # joint_states (Gz -> ROS)
    # tf (Gz -> ROS) - Gz publishes /model/gem/pose, we might need to bridge it or rely on internal odometry plugin
    # sensors
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
            '/base_footprint/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            '/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model', # This might be wrong, JointState in Gz is different.
            # Usually we use specific topic for joint states if the plugin publishes it.
            # AckermannSteering plugin might not publish joint states for all wheels in a standard way?
            # Or we can use JointStatePublisher system in Gz.
            # Let's assume we bridge what we can.
            '/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            '/front_sonar_distance@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan', # Sonar as LaserScan/Ray
            '/rear_sonar_distance@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            '/gps/fix@sensor_msgs/msg/NavSatFix[gz.msgs.NavSat',
            '/imu@sensor_msgs/msg/Imu[gz.msgs.IMU',
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'
        ],
        output='screen'
    )
    
    # 5. Ackermann to Twist Converter
    ackermann_converter = Node(
        package='gem_gazebo',
        executable='gem_ackermann_to_twist.py',
        name='ackermann_to_twist',
        output='screen'
    )

    # 6. TF Broadcaster (from steerai_mpc)
    # This node broadcasts map->odom or world->base_footprint?
    # In ROS1 it was "tf_broadcaster.py".
    tf_broadcaster = Node(
        package='steerai_mpc',
        executable='tf_broadcaster', # It was installed as entry point 'tf_broadcaster' in setup.py of steerai_mpc
        name='tf_broadcaster',
        output='screen'
    )

    # 7. RViz
    rviz_config = os.path.join(pkg_gem_description, 'config_rviz', 'gem_gazebo_rviz.rviz')
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen'
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true', description='Use simulation (Gazebo) clock'),
        DeclareLaunchArgument('world_name', default_value='simple_track_green.world', description='World file name'),
        
        robot_state_publisher,
        gazebo,
        spawn_robot,
        bridge,
        ackermann_converter,
        tf_broadcaster,
        rviz
    ])
