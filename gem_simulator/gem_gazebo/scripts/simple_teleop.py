#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDrive
import sys, select, termios, tty

msg = """
Control The GEM Vehicle!
---------------------------
Moving around:
   w
a  s  d
   x

w/x : increase/decrease linear velocity
a/d : increase/decrease steering angle
s   : force stop

CTRL-C to quit
"""

moveBindings = {
    'w':(1,0),
    'x':(-1,0),
    'a':(0,1),
    'd':(0,-1),
    's':(0,0),
}

def getKey(settings):
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def main():
    settings = termios.tcgetattr(sys.stdin)
    
    rclpy.init()
    node = rclpy.create_node('gem_teleop')
    pub = node.create_publisher(AckermannDrive, '/gem/ackermann_cmd', 10)

    speed = 0.0
    turn = 0.0
    target_speed = 0.0
    target_turn = 0.0
    
    # Parameters
    max_speed = 2.5
    max_turn = 0.61
    speed_step = 0.5
    turn_step = 0.1

    try:
        print(msg)
        while(1):
            key = getKey(settings)
            if key in moveBindings.keys():
                if key == 's':
                    target_speed = 0.0
                    target_turn = 0.0
                else:
                    target_speed += moveBindings[key][0] * speed_step
                    target_turn += moveBindings[key][1] * turn_step
                    
                    # Clamp values
                    target_speed = max(min(target_speed, max_speed), -max_speed)
                    target_turn = max(min(target_turn, max_turn), -max_turn)
                    
                print("speed: %s\tturn: %s" % (target_speed, target_turn))

            elif (key == '\x03'):
                break

            ackermann_cmd = AckermannDrive()
            ackermann_cmd.speed = target_speed
            ackermann_cmd.steering_angle = target_turn
            pub.publish(ackermann_cmd)

    except Exception as e:
        print(e)

    finally:
        ackermann_cmd = AckermannDrive()
        ackermann_cmd.speed = 0.0
        ackermann_cmd.steering_angle = 0.0
        pub.publish(ackermann_cmd)

        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        node.destroy_node()
        rclpy.shutdown()

if __name__=="__main__":
    main()
