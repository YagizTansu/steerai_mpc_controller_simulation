#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import rospkg
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the Neural Network
class DynamicsModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DynamicsModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act1 = nn.Softplus()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = nn.Softplus()
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.act1(out)
        out = self.fc2(out)
        out = self.act2(out)
        out = self.fc3(out)
        return out

def train_model():
    # 1. Load Data
    rospack = rospkg.RosPack()
    data_dir = os.path.join(rospack.get_path('steerai_data_collector'), 'data')
    
    # Find latest training_data_*.csv or training_data.csv
    files = [f for f in os.listdir(data_dir) if f.startswith('training_data') and f.endswith('.csv')]
    if not files:
        print(f"Error: No training data found in {data_dir}")
        return
        
    # Sort by modification time
    files.sort(key=lambda x: os.path.getmtime(os.path.join(data_dir, x)), reverse=True)
    latest_file = files[0]
    data_path = os.path.join(data_dir, latest_file)
    print(f"Loading data from: {data_path}")

    df = pd.read_csv(data_path)
    
    # 2. Feature Engineering
    # Inputs: [curr_speed, curr_yaw_rate, cmd_speed, cmd_steering_angle]
    # Targets: [delta_speed, delta_yaw]
    
    # DOWNSAMPLING: 50Hz -> 10Hz
    # Take every 5th sample to match MPC dt=0.1s
    # Data Collector dt = 0.02s (50Hz)
    # Target dt = 0.10s (10Hz)
    df = df.iloc[::5].reset_index(drop=True)
    print(f"Data downsampled to 10Hz. New shape: {df.shape}")
    
    # Shift data to get next state
    df['next_speed'] = df['curr_speed'].shift(-1)
    df['next_yaw'] = df['curr_yaw'].shift(-1)
    
    # Calculate delta_speed
    df['delta_speed'] = df['next_speed'] - df['curr_speed']

    # Calculate delta_yaw (handle wrapping if necessary, but for small steps simple diff is usually ok)
    # For robust wrapping: (angle + pi) % (2*pi) - pi
    # Calculate delta_yaw with robust wrapping
    # (angle + pi) % (2*pi) - pi
    diff = df['next_yaw'] - df['curr_yaw']
    df['delta_yaw'] = (diff + np.pi) % (2 * np.pi) - np.pi
    
    # Drop last row (NaN due to shift)
    df = df.dropna()
    
    X = df[['curr_speed', 'curr_yaw_rate', 'cmd_speed', 'cmd_steering_angle']].values
    y = df[['delta_speed', 'delta_yaw']].values
    
    # 3. Preprocessing
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    # Save scalers
    output_dir = rospack.get_path('steerai_sysid')
    joblib.dump(scaler_X, os.path.join(output_dir, 'scaler_X.pkl'))
    joblib.dump(scaler_y, os.path.join(output_dir, 'scaler_y.pkl'))
    
    # Split Data
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    
    # Convert to Tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)
    
    # 4. Model Setup
    input_dim = 4
    hidden_dim = 64
    output_dim = 2
    
    model = DynamicsModel(input_dim, hidden_dim, output_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 5. Training Loop
    num_epochs = 100
    train_losses = []
    val_losses = []
    
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
            
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
            
    # Save Model
    torch.save(model.state_dict(), os.path.join(output_dir, 'dynamics_model.pth'))
    print("Model saved.")
    
    # 6. Evaluation & Visualization
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_val_tensor).cpu().numpy()
        
    # Inverse transform to get real units
    y_val_real = scaler_y.inverse_transform(y_val)
    y_pred_real = scaler_y.inverse_transform(y_pred_scaled)
    
    rmse_speed = np.sqrt(mean_squared_error(y_val_real[:, 0], y_pred_real[:, 0]))
    rmse_yaw = np.sqrt(mean_squared_error(y_val_real[:, 1], y_pred_real[:, 1]))
    
    print(f"Validation RMSE Speed: {rmse_speed:.4f} m/s")
    print(f"Validation RMSE Delta Yaw: {rmse_yaw:.4f} rad")
    
    # Plotting
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Loss
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training vs Validation Loss')
    
    # Plot 2: Speed Prediction
    plt.subplot(1, 3, 2)
    plt.plot(y_val_real[:100, 0], label='Ground Truth')
    plt.plot(y_pred_real[:100, 0], label='Predicted')
    plt.xlabel('Sample')
    plt.ylabel('Speed (m/s)')
    plt.legend()
    plt.title('Speed Prediction (First 100 samples)')
    
    # Plot 3: Delta Yaw Prediction
    plt.subplot(1, 3, 3)
    plt.plot(y_val_real[:100, 1], label='Ground Truth')
    plt.plot(y_pred_real[:100, 1], label='Predicted')
    plt.xlabel('Sample')
    plt.ylabel('Delta Yaw (rad)')
    plt.legend()
    plt.title('Delta Yaw Prediction (First 100 samples)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_results.png'))
    print(f"Plots saved to {os.path.join(output_dir, 'training_results.png')}")
    # plt.show() # Cannot show plot in headless environment

if __name__ == '__main__':
    train_model()
