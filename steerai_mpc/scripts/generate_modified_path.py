import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_path():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to steerai_mpc, then down to paths
    paths_dir = os.path.join(os.path.dirname(script_dir), 'paths')
    
    # Parameters
    step_size = 0.1  # meters
    
    # 1. Straight: (-50, 0) to (50, 0)
    x1 = np.arange(-50, 50, step_size)
    y1 = np.zeros_like(x1)
    
    # 2. Wide Turn: 180-degree left turn from (50, 0) to (50, 30) (Radius = 15m)
    # Center of turn is (50, 15)
    # Angle goes from -pi/2 to pi/2
    radius = 100.0
    center_x1 = 50.0
    center_y1 = 100.0
    theta2 = np.arange(-np.pi/2, np.pi/2, step_size/radius)
    x2 = center_x1 + radius * np.cos(theta2)
    y2 = center_y1 + radius * np.sin(theta2)
    
    # 3. Straight: (50, 200) to (-50, 200)
    # Moving backwards in x
    x3 = np.arange(50, -50, -step_size)
    y3 = np.full_like(x3, 200.0)
    
    # 4. Wide Turn: 180-degree left turn from (0, 30) to (0, 0) (Radius = 15m)
    # Center of turn is (0, 15)
    # Angle goes from pi/2 to 3*pi/2
    center_x2 = -50.0
    center_y2 = 100.0
    theta4 = np.arange(np.pi/2, 3*np.pi/2, step_size/radius)
    x4 = center_x2 + radius * np.cos(theta4)
    y4 = center_y2 + radius * np.sin(theta4)
    
    # 5. Straight: (0, 0) to (20, 0) (Arrival) - REMOVED to make it a perfect loop
    # x5 = np.arange(0, 20, step_size)
    # y5 = np.zeros_like(x5)
    
    # Concatenate all segments
    x = np.concatenate([x1, x2, x3, x4])
    y = np.concatenate([y1, y2, y3, y4])
    
    # Create DataFrame
    df = pd.DataFrame({'curr_x': x, 'curr_y': y})
    
    # Save to CSV
    output_path = os.path.join(paths_dir, 'reference_path_sim.csv')
    df.to_csv(output_path, index=False)
    print(f"Path generated and saved to {output_path}")
    
    # Plot for verification
    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.title("Generated Reference Path")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis('equal')
    plt.grid(True)
    plot_path = os.path.join(paths_dir, 'path_verification.png')
    plt.savefig(plot_path)
    print(f"Verification plot saved to {plot_path}")

if __name__ == "__main__":
    generate_path()
