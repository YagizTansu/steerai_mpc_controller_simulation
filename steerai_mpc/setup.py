from setuptools import setup
import os
from glob import glob

package_name = 'steerai_mpc'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'paths'), glob('paths/*.csv')),
    ],
    install_requires=['setuptools', 'casadi'],
    zip_safe=True,
    maintainer='yagiz',
    maintainer_email='yagiz@todo.todo',
    description='MPC Controller for SteerAI',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mpc_controller = steerai_mpc.mpc_controller:main',
            'tf_broadcaster = steerai_mpc.tf_broadcaster:main',
            'path_publisher = steerai_mpc.path_publisher:main',
        ],
    },
)
