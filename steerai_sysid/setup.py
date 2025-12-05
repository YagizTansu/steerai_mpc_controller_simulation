from setuptools import setup
import os
from glob import glob

package_name = 'steerai_sysid'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='The steerai_sysid package for system identification',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'train_dynamics = steerai_sysid.train_dynamics:main',
        ],
    },
)
