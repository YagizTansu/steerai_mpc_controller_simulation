#!/usr/bin/env python3
"""
Path Generator Script
Bu script ile farklı geometrik şekiller kullanarak path oluşturabilirsiniz.
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
        """Düz bir yol ekle"""
        if len(self.x_points) == 0:
            start_x, start_y = 0.0, 0.0
        else:
            start_x = self.x_points[-1]
            start_y = self.y_points[-1]
        
        x = np.linspace(start_x, start_x + length, num_points)
        y = np.full(num_points, start_y)
        
        self.x_points.extend(x.tolist())
        self.y_points.extend(y.tolist())
        print(f"Düz yol eklendi: {length}m, {num_points} nokta")
    
    def add_sine_curve(self, length, amplitude, frequency, num_points=50):
        """Sinüs eğrisi ekle"""
        if len(self.x_points) == 0:
            start_x, start_y = 0.0, 0.0
        else:
            start_x = self.x_points[-1]
            start_y = self.y_points[-1]
        
        x = np.linspace(0, length, num_points)
        y = amplitude * np.sin(2 * np.pi * frequency * x / length)
        
        self.x_points.extend((x + start_x).tolist())
        self.y_points.extend((y + start_y).tolist())
        print(f"Sinüs eğrisi eklendi: uzunluk={length}m, genlik={amplitude}m, frekans={frequency}")
    
    def add_circular_arc(self, radius, angle_degrees, num_points=30, direction='left'):
        """Dairesel yay ekle (angle_degrees: 0-360 arası)"""
        if len(self.x_points) == 0:
            start_x, start_y = 0.0, 0.0
        else:
            start_x = self.x_points[-1]
            start_y = self.y_points[-1]
        
        angle_rad = np.radians(angle_degrees)
        theta = np.linspace(0, angle_rad, num_points)
        
        if direction == 'left':
            # Sola dönüş
            center_x = start_x
            center_y = start_y + radius
            x = center_x + radius * np.sin(theta)
            y = center_y - radius * np.cos(theta)
        else:
            # Sağa dönüş
            center_x = start_x
            center_y = start_y - radius
            x = center_x + radius * np.sin(theta)
            y = center_y + radius * np.cos(theta)
        
        self.x_points.extend(x.tolist())
        self.y_points.extend(y.tolist())
        print(f"Dairesel yay eklendi: yarıçap={radius}m, açı={angle_degrees}°, yön={direction}")
    
    def add_s_curve(self, length, amplitude, num_points=50):
        """S eğrisi ekle"""
        if len(self.x_points) == 0:
            start_x, start_y = 0.0, 0.0
        else:
            start_x = self.x_points[-1]
            start_y = self.y_points[-1]
        
        x = np.linspace(0, length, num_points)
        # S-curve: tanh fonksiyonu kullanarak
        y = amplitude * np.tanh(4 * (x/length - 0.5))
        
        self.x_points.extend((x + start_x).tolist())
        self.y_points.extend((y + start_y).tolist())
        print(f"S eğrisi eklendi: uzunluk={length}m, genlik={amplitude}m")
    
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
        print(f"Chicane eklendi: uzunluk={length}m, genlik={amplitude}m, viraj sayısı={num_curves}")
    
    def add_spiral(self, turns, max_radius, num_points=100):
        """Spiral ekle"""
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
        print(f"Spiral eklendi: dönüş={turns}, max yarıçap={max_radius}m")
    
    def add_hairpin(self, radius, num_points=40):
        """Hairpin (180 derece dönüş) ekle"""
        self.add_circular_arc(radius, 180, num_points, direction='left')
        print(f"Hairpin eklendi: yarıçap={radius}m")
    
    def clear(self):
        """Tüm path'i temizle"""
        self.x_points.clear()
        self.y_points.clear()
        print("Path temizlendi")
    
    def save_to_csv(self, filename):
        """Path'i CSV dosyasına kaydet"""
        # Ensure the paths directory exists
        dir_path = os.path.dirname(filename)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['curr_x', 'curr_y'])
            for x, y in zip(self.x_points, self.y_points):
                writer.writerow([x, y])
        
        print(f"\nPath kaydedildi: {filename}")
        print(f"Toplam nokta sayısı: {len(self.x_points)}")
    
    def plot(self, show_points=False, save_fig=None):
        """Path'i görselleştir"""
        plt.figure(figsize=(12, 8))
        
        if show_points:
            plt.plot(self.x_points, self.y_points, 'b-', linewidth=2, label='Path')
            plt.plot(self.x_points, self.y_points, 'ro', markersize=3, label='Waypoints')
        else:
            plt.plot(self.x_points, self.y_points, 'b-', linewidth=2, label='Path')
        
        # Start ve end noktalarını işaretle
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
            print(f"Görsel kaydedildi: {save_fig}")
        
        plt.show()


# ============================================================================
# ÖRNEK KULLANIM - İstediğiniz gibi modifiye edebilirsiniz!
# ============================================================================

if __name__ == "__main__":
    # Path generator oluştur
    pg = PathGenerator()
    
    # ========== SENARYO 1: Karmaşık Test Parkuru (Varsayılan) ==========
    print("\n=== Path Oluşturuluyor ===\n")
    
    # Başlangıç düz yol
    pg.add_straight(length=20, num_points=20)
    
    # Sinüs eğrisi (hafif kıvrımlı)
    pg.add_sine_curve(length=20, amplitude=3, frequency=1, num_points=40)
    
    # Düz yol
    pg.add_straight(length=20, num_points=20)
    
    # S-curve (keskin)
    pg.add_s_curve(length=20, amplitude=3, num_points=30)
    
    # Chicane (zigzag)
    pg.add_chicane(length=20, amplitude=4, num_curves=1, num_points=40)
    
    # Hairpin dönüş
    pg.add_hairpin(radius=13, num_points=40)
    
    # Düz yol
    pg.add_straight(length=-15, num_points=30)
    
    # Dairesel viraj
    pg.add_circular_arc(radius=10, angle_degrees=-90, num_points=30, direction='left')
    
    # Düz yol
    pg.add_straight(length=20, num_points=20)
    
    # Path'i kaydet ve görselleştir
    print("\n=== İşlem Tamamlanıyor ===\n")
    
    # CSV olarak kaydet
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, '../paths/reference_path_generated.csv')
    pg.save_to_csv(csv_path)
    
    # Görseli kaydet
    fig_path = os.path.join(script_dir, '../paths/reference_path_generated.png')
    
    # Plot et
    pg.plot(show_points=False, save_fig=fig_path)
    
    print("\n=== Tamamlandı! ===")
    print(f"CSV: {csv_path}")
    print(f"PNG: {fig_path}")
