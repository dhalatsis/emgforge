"""
Motor unit model for fiber positioning and parameters.

This module handles motor unit fiber placement and geometric parameters.
"""

import numpy as np


class MotorUnit:
    """
    Motor unit model with multiple muscle fibers.
    
    This class manages the positioning and properties of muscle fibers
    within a motor unit.
    
    Parameters
    ----------
    n_fibers : int
        Number of fibers in the motor unit
    radius : float
        Radius of the motor unit (circular cross-section) in mm
    y0 : float
        Mean depth of the fibers in the muscle (in mm)
    innervation_spread : float
        Spread of the innervation zone (in mm)
    zi : float
        z position of the center of the innervation zone (in mm)
    L1 : float
        Lower semifiber length (in mm)
    L2 : float
        Upper semifiber length (in mm)
    Ten1 : float
        Spread of the lower tendon region (in mm)
    Ten2 : float
        Spread of the upper tendon region (in mm)
    distfib : float
        Distance between the MU and the geometrical center of the detection
        system in the transversal (angular) direction (in degrees)
    r : float
        External radius of the limb model (in mm)
    h : float
        Thickness of the fat layer (in mm)
    d : float
        Thickness of the skin layer (in mm)
    """
    
    def __init__(self, n_fibers, radius, y0, innervation_spread, zi, L1, L2,
                 Ten1, Ten2, distfib, r, h, d):
        self.n_fibers = n_fibers
        self.radius = radius
        self.y0 = y0
        self.Inn = innervation_spread
        self.zi = zi
        self.L1 = L1
        self.L2 = L2
        self.Ten1 = Ten1
        self.Ten2 = Ten2
        self.distfib_deg = distfib
        self.distfib = distfib * np.pi / 180  # Convert to radians
        self.r = r
        self.h = h
        self.d = d
        
        # Generate fiber positions
        self.positions = self._generate_fiber_positions()
        self.fiber_pos_th = self.distfib
        self.fin1, self.fin2, self.len1, self.len2 = self._generate_fiber_lengths()
        
    def _generate_fiber_positions(self):
        """
        Generate random positions for fibers within the motor unit.
        
        Returns
        -------
        positions : ndarray
            Array of shape (n_fibers, 3) containing (x, y, z) coordinates
        """
        # Random x positions within the MU radius
        posx = (np.random.rand(self.n_fibers) - 0.5) * 2 * self.radius
        
        # Compute valid y range for each x position
        max_value = np.sqrt(self.radius**2 - posx**2)
        min_value = np.maximum(-np.sqrt(self.radius**2 - posx**2), 
                               -self.y0 * np.ones(self.n_fibers))
        
        # Random y positions within valid range
        posy = np.random.rand(self.n_fibers) * (max_value - min_value) + min_value
        posy = posy + self.y0
        
        # Transform to cylindrical coordinates
        posx = posx + (self.r - self.h - self.d - self.y0) * np.sin(self.distfib)
        posy = (self.r - self.h - self.d - posy) * np.cos(self.distfib)
        
        # Random z positions (along fiber length)
        posz = (np.random.rand(self.n_fibers) - 0.5) * self.Inn + self.zi
        
        positions = np.column_stack([posx, posy, posz])
        return positions
    
    def _generate_fiber_lengths(self):
        """
        Generate random fiber lengths and tendon positions.
        
        Returns
        -------
        fin1 : ndarray
            Lower fiber end positions
        fin2 : ndarray
            Upper fiber end positions
        len1 : ndarray
            Lower semifiber lengths
        len2 : ndarray
            Upper semifiber lengths
        """
        fin1 = self.L1 + (np.random.rand(self.n_fibers) - 0.5) * self.Ten1
        fin2 = self.L2 + (np.random.rand(self.n_fibers) - 0.5) * self.Ten2
        len1 = fin1 - self.positions[:, 2]
        len2 = fin2 + self.positions[:, 2]
        return fin1, fin2, len1, len2
    
    def get_positions(self):
        """
        Get fiber positions.
        
        Returns
        -------
        positions : ndarray
            Array of fiber positions (n_fibers, 3)
        """
        return self.positions
    
    def get_fiber_info(self):
        """
        Get complete fiber information.
        
        Returns
        -------
        info : dict
            Dictionary containing fiber positions, lengths, and parameters
        """
        return {
            'positions': self.positions,
            'fin1': self.fin1,
            'fin2': self.fin2,
            'len1': self.len1,
            'len2': self.len2,
            'fiber_pos_th': self.fiber_pos_th,
            'n_fibers': self.n_fibers
        }


