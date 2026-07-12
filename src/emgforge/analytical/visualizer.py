"""
Visualizer for cylindrical limb model and EMG signals.

This module provides visualization tools for the model geometry and
generated signals.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Visualizer:
    """
    Visualizer for model geometry and EMG signals.
    
    This class provides methods to visualize the cylindrical limb model,
    motor unit positions, electrode positions, and generated signals.
    
    Parameters
    ----------
    r : float
        External radius of the model (in mm)
    h : float
        Thickness of the fat layer (in mm)
    d : float
        Thickness of the skin layer (in mm)
    r_bone : float
        Radius of the bone layer (in mm)
    model_length : float
        Total length of the cylinder (in mm)
    """
    
    def __init__(self, r, h, d, r_bone, model_length):
        self.r = r
        self.h = h
        self.d = d
        self.r_bone = r_bone
        self.model_length = model_length
        
    def plot_geometry(self, motor_unit, detection_system, fiber_pos_x, 
                     fiber_depth, center_z, alpha_deg):
        """
        Plot the 3D geometry of the cylindrical model.
        
        Parameters
        ----------
        motor_unit : MotorUnit
            Motor unit object
        detection_system : DetectionSystem
            Detection system object
        fiber_pos_x : float
            Fiber position in x direction
        fiber_depth : float
            Fiber depth
        center_z : float
            Center z position
        alpha_deg : float
            Angle in degrees
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Define angles and z positions for cylinder
        theta = np.linspace(np.pi / 2, 2 * np.pi + np.pi / 2, 100)
        zeta = np.linspace(-self.model_length / 2, self.model_length / 2, 50)
        
        # Outer skin layer
        x = self.r * np.cos(theta)
        y = self.r * np.sin(theta)
        Z, X = np.meshgrid(zeta, x)
        Y = np.tile(y[:, np.newaxis], (1, len(zeta)))
        ax.plot_surface(Z, X, Y, alpha=0.3, color='blue', label='Skin')
        
        # Fat layer
        r_fat = self.r - self.d
        x = r_fat * np.cos(theta)
        y = r_fat * np.sin(theta)
        Z, X = np.meshgrid(zeta, x)
        Y = np.tile(y[:, np.newaxis], (1, len(zeta)))
        ax.plot_surface(Z, X, Y, alpha=0.3, color='yellow', label='Fat')
        
        # Muscle layer
        r_muscle = self.r - self.d - self.h
        x = r_muscle * np.cos(theta)
        y = r_muscle * np.sin(theta)
        Z, X = np.meshgrid(zeta, x)
        Y = np.tile(y[:, np.newaxis], (1, len(zeta)))
        ax.plot_surface(Z, X, Y, alpha=0.3, color='red', label='Muscle')
        
        # Bone layer
        x = self.r_bone * np.cos(theta)
        y = self.r_bone * np.sin(theta)
        Z, X = np.meshgrid(zeta, x)
        Y = np.tile(y[:, np.newaxis], (1, len(zeta)))
        ax.plot_surface(Z, X, Y, alpha=0.3, color='green', label='Bone')
        
        # Plot motor unit fibers
        positions = motor_unit.get_positions()
        ax.scatter(positions[:, 2], positions[:, 0], positions[:, 1], 
                  c='black', marker='o', s=20, label='MU fibers')
        
        # Plot electrodes
        th_center, z_center = detection_system.get_electrode_centers()
        for i in range(detection_system.channels):
            # Convert electrode position to Cartesian coordinates
            # Electrodes are on the skin surface
            x_elec = self.r * np.cos(th_center[i] + np.pi / 2)
            y_elec = self.r * np.sin(th_center[i] + np.pi / 2)
            ax.scatter(z_center[i], x_elec, y_elec, c='white', 
                      marker='o', s=100, edgecolors='black', linewidths=2)
        
        ax.set_xlabel('Z axis (mm)')
        ax.set_ylabel('X axis (mm)')
        ax.set_zlabel('Y axis (mm)')
        ax.set_title('Cylindrical Limb Model - 3D Geometry')
        # Matplotlib 3D surfaces don't always cooperate with legends across versions.
        # This model's plots are still correct without a legend, so keep it best-effort.
        try:
            ax.legend()
        except Exception:
            pass
        
        return fig, ax
    
    def plot_situation_2d(self, motor_unit, detection_system, distfib_deg):
        """
        Plot the 2D cross-sectional situation.
        
        Parameters
        ----------
        motor_unit : MotorUnit
            Motor unit object
        detection_system : DetectionSystem
            Detection system object
        distfib_deg : float
            Distance fiber in degrees
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot innervation zone
        Inn = motor_unit.Inn
        zi = motor_unit.zi
        L1 = motor_unit.L1
        L2 = motor_unit.L2
        Ten1 = motor_unit.Ten1
        Ten2 = motor_unit.Ten2
        
        ax.plot([0, 0], [zi - Inn / 2, zi + Inn / 2], 'b-', linewidth=4)
        
        # Plot muscle fiber region
        see = [0.3, 0.3, -0.3, -0.3, 0.3]
        dee = [zi - L1, zi + L2, zi + L2, zi - L1, zi - L1]
        ax.fill(see, dee, 'r', alpha=0.3)
        
        # Plot tendon regions
        ax.plot([0, 0], [zi + L2 + Ten2 / 2, zi + L2 - Ten2 / 2], 'b-', linewidth=4)
        ax.plot([0, 0], [zi - L1 - Ten1 / 2, zi - L1 + Ten1 / 2], 'b-', linewidth=4)
        
        # Mark key points
        ax.plot(0, zi, 'ko', markersize=8)
        ax.plot(0, zi - L1, 'ko', markersize=8)
        ax.plot(0, zi + L2, 'ko', markersize=8)
        
        # Plot electrodes
        distfib_rad = distfib_deg * np.pi / 180
        distfib_x = (self.r - self.h - self.d - motor_unit.y0) * np.sin(distfib_rad)
        
        alpha = detection_system.alpha
        dint = detection_system.dint
        center = detection_system.center
        channels = detection_system.channels
        dintsf = detection_system.dintsf
        
        distfib_x = distfib_x - (channels - 1) * dint * np.sin(alpha) / 2
        center_z = center - (channels - 1) * dint * np.cos(alpha) / 2
        
        for channel in range(channels):
            x_pos = distfib_x + channel * dint * np.sin(alpha)
            z_pos = center_z + channel * dint * np.cos(alpha)
            ax.plot(x_pos, z_pos, 'ko', markersize=10, markerfacecolor='white', 
                   markeredgewidth=2)
            
            # Draw electrode shape
            if detection_system.electrode_type == 'circ':
                circle = plt.Circle((x_pos, z_pos), detection_system.dim1, 
                                  fill=True, color='blue', alpha=0.3)
                ax.add_patch(circle)
            elif detection_system.electrode_type == 'rect':
                # Draw rectangle
                dim1 = detection_system.dim1
                dim2 = detection_system.dim2
                rect_x = [x_pos - dim1 / 2, x_pos + dim1 / 2, 
                         x_pos + dim1 / 2, x_pos - dim1 / 2, x_pos - dim1 / 2]
                rect_z = [z_pos - dim2 / 2, z_pos - dim2 / 2, 
                         z_pos + dim2 / 2, z_pos + dim2 / 2, z_pos - dim2 / 2]
                ax.fill(rect_x, rect_z, 'blue', alpha=0.3)
        
        ax.set_xlabel('X axis (mm)')
        ax.set_ylabel('Z axis (mm)')
        ax.set_title('2D Situation - Fiber and Electrode Layout')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        
        # Set reasonable axis limits
        margin = self.model_length / 4
        ax.set_xlim([-margin, margin])
        ax.set_ylim([-margin, margin])
        
        return fig, ax
    
    def plot_signals(self, t, sig_arr, channels):
        """
        Plot the generated EMG signals.
        
        Parameters
        ----------
        t : ndarray
            Time vector (in ms)
        sig_arr : ndarray
            Signal array (channels x time)
        channels : int
            Number of channels
        """
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Normalize and offset signals for visualization
        sig_arr2 = sig_arr / np.max(np.abs(sig_arr)) * 1.5
        sig_plot = np.zeros_like(sig_arr2)
        
        for u in range(channels):
            sig_plot[u, :] = sig_arr2[u, :] + (u) * np.ones(sig_arr2.shape[1])
        
        # Plot all channels
        for u in range(channels):
            ax.plot(t, sig_plot[u, :], 'k-', linewidth=0.8)
        
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Channel')
        ax.set_title('Simulated Motor Unit Action Potentials')
        ax.set_ylim([-2, channels + 1])
        ax.set_xlim([0, t[-1]])
        ax.grid(True, alpha=0.3)
        
        # Add channel labels
        ax.set_yticks(range(channels))
        ax.set_yticklabels([f'Ch {i+1}' for i in range(channels)])
        
        return fig, ax
    
    @staticmethod
    def show_all():
        """Display all figures."""
        plt.show()
    
    @staticmethod
    def save_figure(fig, filename, dpi=300):
        """
        Save a figure to file.
        
        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure to save
        filename : str
            Output filename
        dpi : int
            Resolution in dots per inch
        """
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"Saved figure to {filename}")


