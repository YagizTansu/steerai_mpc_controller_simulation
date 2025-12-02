#!/usr/bin/env python3
"""
Path Generator Script
You can create paths using different geometric shapes with this script.
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import os

class PathGenerator:
    def __init__(self):
        self.waypoints = []
        self.x_points = []
        self.y_points = []
    
    def add_straight(self, length, num_points=20):
        """Add a straight path"""
        if len(self.x_points) == 0:
            start_x, start_y = 0.0, 0.0
        else:
            start_x = self.x_points[-1]
            start_y = self.y_points[-1]
        
        x = np.linspace(start_x, start_x + length, num_points)
        y = np.full(num_points, start_y)
        
        self.x_points.extend(x.tolist())
        self.y_points.extend(y.tolist())
        print(f"Straight path added: {length}m, {num_points} points")
    
    def add_sine_curve(self, length, amplitude, frequency, num_points=50):
        """Add a sine curve"""
        if len(self.x_points) == 0:
            start_x, start_y = 0.0, 0.0
        else:
            start_x = self.x_points[-1]
            start_y = self.y_points[-1]
        
        x = np.linspace(0, length, num_points)
        y = amplitude * np.sin(2 * np.pi * frequency * x / length)
        
        self.x_points.extend((x + start_x).tolist())
        self.y_points.extend((y + start_y).tolist())
        print(f"Sine curve added: length={length}m, amplitude={amplitude}m, frequency={frequency}")
    
    def add_circular_arc(self, radius, angle_degrees, num_points=30, direction='left'):
        """Add a circular arc (angle_degrees: between 0-360)"""
        if len(self.x_points) == 0:
            start_x, start_y = 0.0, 0.0
        else:
            start_x = self.x_points[-1]
            start_y = self.y_points[-1]
        
        angle_rad = np.radians(angle_degrees)
        theta = np.linspace(0, angle_rad, num_points)
        
        if direction == 'left':
            # Left turn
            center_x = start_x
            center_y = start_y + radius
            x = center_x + radius * np.sin(theta)
            y = center_y - radius * np.cos(theta)
        else:
            # Right turn
            center_x = start_x
            center_y = start_y - radius
            x = center_x + radius * np.sin(theta)
            y = center_y + radius * np.cos(theta)
        
        self.x_points.extend(x.tolist())
        self.y_points.extend(y.tolist())
        print(f"Circular arc added: radius={radius}m, angle={angle_degrees}°, direction={direction}")
    
    def add_s_curve(self, length, amplitude, num_points=50):
        """Add an S-curve"""
        if len(self.x_points) == 0:
            start_x, start_y = 0.0, 0.0
        else:
            start_x = self.x_points[-1]
            start_y = self.y_points[-1]
        
        x = np.linspace(0, length, num_points)
        # S-curve: using tanh function
        y = amplitude * np.tanh(4 * (x/length - 0.5))
        
        self.x_points.extend((x + start_x).tolist())
        self.y_points.extend((y + start_y).tolist())
        print(f"S-curve added: length={length}m, amplitude={amplitude}m")
    
    def add_chicane(self, length, amplitude, num_curves=2, num_points=50):
        """Chicane (zigzag) ekle"""
        if len(self.x_points) == 0:
            start_x, start_y = 0.0, 0.0
        else:
            start_x = self.x_points[-1]
            start_y = self.y_points[-1]
        
        x = np.linspace(0, length, num_points)
        y = amplitude * np.sin(2 * np.pi * num_curves * x / length)
        
        self.x_points.extend((x + start_x).tolist())
        self.y_points.extend((y + start_y).tolist())
        print(f"Chicane added: length={length}m, amplitude={amplitude}m, number of curves={num_curves}")
    
    def add_spiral(self, turns, max_radius, num_points=100):
        """Add a spiral"""
        if len(self.x_points) == 0:
            start_x, start_y = 0.0, 0.0
        else:
            start_x = self.x_points[-1]
            start_y = self.y_points[-1]
        
        theta = np.linspace(0, turns * 2 * np.pi, num_points)
        r = np.linspace(0, max_radius, num_points)
        
        x = r * np.cos(theta) + start_x
        y = r * np.sin(theta) + start_y
        
        self.x_points.extend(x.tolist())
        self.y_points.extend(y.tolist())
        print(f"Spiral added: turns={turns}, max radius={max_radius}m")
    
    def add_hairpin(self, radius, num_points=40):
        """Add a hairpin (180 degree turn)"""
        self.add_circular_arc(radius, 180, num_points, direction='left')
        print(f"Hairpin added: radius={radius}m")
    
    def clear(self):
        """Clear the entire path"""
        self.x_points.clear()
        self.y_points.clear()
        print("Path cleared")
    
    def save_to_csv(self, filename):
        """Save path to CSV file"""
        # Ensure the paths directory exists
        dir_path = os.path.dirname(filename)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['curr_x', 'curr_y'])
            for x, y in zip(self.x_points, self.y_points):
                writer.writerow([x, y])
        
        print(f"\nPath saved: {filename}")
        print(f"Total number of points: {len(self.x_points)}")
    
    def plot(self, show_points=False, save_fig=None):
        """Visualize the path"""
        plt.figure(figsize=(12, 8))
        
        if show_points:
            plt.plot(self.x_points, self.y_points, 'b-', linewidth=2, label='Path')
            plt.plot(self.x_points, self.y_points, 'ro', markersize=3, label='Waypoints')
        else:
            plt.plot(self.x_points, self.y_points, 'b-', linewidth=2, label='Path')
        
        # Mark start and end points
        if len(self.x_points) > 0:
            plt.plot(self.x_points[0], self.y_points[0], 'go', markersize=10, label='Start')
            plt.plot(self.x_points[-1], self.y_points[-1], 'ro', markersize=10, label='End')
        
        plt.grid(True, alpha=0.3)
        plt.xlabel('X (m)', fontsize=12)
        plt.ylabel('Y (m)', fontsize=12)
        plt.title('Generated Path', fontsize=14, fontweight='bold')
        plt.legend()
        plt.axis('equal')
        
        if save_fig:
            plt.savefig(save_fig, dpi=150, bbox_inches='tight')
            print(f"Figure saved: {save_fig}")
        
        plt.show()


# ============================================================================
# EXAMPLE USAGE - You can modify it as you wish!
# ============================================================================

if __name__ == "__main__":
    # Create path generator
    pg = PathGenerator()
    
    # ========== SCENARIO 1: Complex Test Track (Default) ==========
    print("\n=== Creating Path ===")
    
    # Initial straight path
    pg.add_straight(length=15, num_points=20)
    
    # Sine curve (slight curves)
    pg.add_sine_curve(length=20, amplitude=3, frequency=1, num_points=40)
    
    # Straight path
    pg.add_straight(length=20, num_points=20)
    
    # Chicane (zigzag)
    pg.add_chicane(length=20, amplitude=3, num_curves=1, num_points=40)
    
    pg.add_straight(length=5, num_points=20)

    # Hairpin turn
    pg.add_hairpin(radius=13, num_points=40)
    
    # Straight path
    pg.add_straight(length=-15, num_points=30)
    
    # Circular turn
    pg.add_circular_arc(radius=10, angle_degrees=-360, num_points=30, direction='left')
    
    pg.add_straight(length=-20, num_points=30)

    pg.add_chicane(length=-20, amplitude=3, num_curves=1, num_points=40)
    
    pg.add_sine_curve(length=-30, amplitude=3 , frequency=1, num_points=40)

    pg.add_circular_arc(radius=13, angle_degrees=-180, num_points=30, direction='right')


    # Straight path
    # pg.add_straight(length=20, num_points=20)
    
    # Save and visualize the path
    print("\n=== Finalizing ===")
    
    # Save as CSV
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, '../paths/reference_path_generated.csv')
    pg.save_to_csv(csv_path)
    
    # Save the figuree
    fig_path = os.path.join(script_dir, '../paths/reference_path_generated.png')
    
    # Plot
    pg.plot(show_points=False, save_fig=fig_path)
    
    print("\n=== Completed! ===")
    print(f"CSV: {csv_path}")
    print(f"PNG: {fig_path}")
