"""
Detection system model for electrode configuration.

This module handles the electrode array configuration and spatial filtering.
"""

import numpy as np
from scipy.special import j1


class DetectionSystem:
    """
    Detection system with electrode array and spatial filtering.
    
    This class manages the configuration of surface electrodes and
    implements various spatial filters.
    
    Parameters
    ----------
    channels : int
        Number of acquired channels
    dint : float
        Distance between different detection points (in mm)
    center : float
        z position of the barycenter of the detection system (in mm)
    alpha : float
        Angle of inclination of the MU w.r.t. the detection system (in degrees)
    det_type : int
        Spatial filter type:
        1 -> monopolar
        2 -> single differential
        3 -> double differential
        4 -> laplacian
        5 -> L2 filter
        6 -> longitudinal double differential
    dintsf : float
        Interelectrode distance for the spatial filter (in mm)
    electrode_type : str
        Type of recording electrodes ('rect', 'circ', 'conc', or 'point')
    dim1 : float
        First dimension of the electrodes (in x direction) or radius for circular (in mm)
    dim2 : float
        Second dimension of the electrodes (in z direction), not used for circular (in mm)
    r : float
        External radius of the limb model (in mm)
    """
    
    def __init__(self, channels, dint, center, alpha, det_type, dintsf, 
                 electrode_type, dim1, dim2, r):
        self.channels = channels
        self.dint = dint
        self.center = center
        self.alpha_deg = alpha
        self.alpha = alpha * np.pi / 180  # Convert to radians
        self.det_type = det_type
        self.dintsf = dintsf
        self.electrode_type = electrode_type
        self.dim1 = dim1
        self.dim2 = dim2
        self.r = r
        
        # Compute electrode centers
        self.th_center, self.z_center = self._compute_electrode_centers()
        
    def _compute_electrode_centers(self):
        """
        Compute the center positions of all electrodes.
        
        Returns
        -------
        th_center : ndarray
            Angular positions of electrode centers
        z_center : ndarray
            z positions of electrode centers
        """
        z_center = np.zeros(self.channels)
        th_center = np.zeros(self.channels)
        
        for c in range(self.channels):
            z_center[c] = (-(self.channels - 1) * self.dint * np.cos(self.alpha) / 2 + 
                          c * self.dint * np.cos(self.alpha) + self.center)
            th_center[c] = (-(self.channels - 1) * self.dint * np.sin(self.alpha) / (2 * self.r) + 
                           c * self.dint * np.sin(self.alpha) / self.r)
        
        return th_center, z_center
    
    def compute_spatial_filter(self, ktheta, kz, w=256):
        """
        Compute the spatial filter transfer function.
        
        Parameters
        ----------
        ktheta : ndarray
            Angular spatial frequency
        kz : ndarray
            Longitudinal spatial frequency
        w : int
            Resolution parameter
            
        Returns
        -------
        H : ndarray
            Spatial filter transfer function
        """
        # Adjust electrode distance for differential filters
        if self.det_type == 2:
            delec = self.dintsf / 2
        else:
            delec = self.dintsf
        
        # Transform spatial frequencies according to inclination angle
        KKT = ktheta * np.cos(-self.alpha) + self.r * kz * np.sin(-self.alpha)
        KKZ = -ktheta * np.sin(-self.alpha) / self.r + kz * np.cos(-self.alpha)
        
        # Compute spatial filter based on type
        if self.det_type == 4:  # Laplacian
            H = -2 * np.cos(KKZ * self.dintsf) - 2 * np.cos(KKT * self.dintsf / self.r) + 4
        elif self.det_type == 5:  # L2 filter
            H = 1 - (1 + np.cos(KKZ * self.dintsf)) * (1 + np.cos(KKT * self.dintsf / self.r)) / 4
        elif self.det_type == 6:  # Longitudinal double differential
            H = -(2 * 1j)**2 * (np.sin(KKT * self.dintsf / 2))**2
        elif self.det_type == 2:  # Single differential
            H = (2 * 1j) * np.sin(KKZ * self.dintsf / 2)
        elif self.det_type == 3:  # Double differential
            H = -(2 * 1j)**2 * (np.sin(KKZ * self.dintsf / 2))**2
        elif self.det_type == 1:  # Monopolar
            H = np.ones_like(ktheta)
        else:
            H = np.ones_like(ktheta)
        
        # Normalize
        H = H / np.max(np.abs(H))
        
        return H
    
    def compute_electrode_size_filter(self, ktheta, kz):
        """
        Compute the transfer function of electrode physical dimensions.
        
        Parameters
        ----------
        ktheta : ndarray
            Angular spatial frequency
        kz : ndarray
            Longitudinal spatial frequency
            
        Returns
        -------
        Hsize : ndarray
            Electrode size transfer function
        """
        # Transform spatial frequencies
        KKT = ktheta * np.cos(-self.alpha) + self.r * kz * np.sin(-self.alpha)
        KKZ = -ktheta * np.sin(-self.alpha) / self.r + kz * np.cos(-self.alpha)
        
        if self.electrode_type == 'rect':
            # Rectangular electrodes
            Hsize = np.sinc(KKT * self.dim1 / (2 * np.pi * self.r)) * np.sinc(KKZ * self.dim2 / (2 * np.pi))
        elif self.electrode_type == 'circ':
            # Circular electrodes
            k_mag = np.sqrt((ktheta / self.r)**2 + kz**2)
            with np.errstate(divide='ignore', invalid='ignore'):
                Hsize = 2 * j1(self.dim1 * k_mag) / (self.dim1 * k_mag)
                # Handle singularities
                Hsize[np.isnan(Hsize)] = 1
        else:
            # Point electrodes
            Hsize = np.ones_like(KKT)
        
        return Hsize
    
    def compute_complete_filter(self, ktheta, kz, w=256):
        """
        Compute the complete detection system transfer function.
        
        Parameters
        ----------
        ktheta : ndarray
            Angular spatial frequency
        kz : ndarray
            Longitudinal spatial frequency
        w : int
            Resolution parameter
            
        Returns
        -------
        H : ndarray
            Complete detection system transfer function
        """
        H_spatial = self.compute_spatial_filter(ktheta, kz, w)
        H_size = self.compute_electrode_size_filter(ktheta, kz)
        H = H_spatial * H_size
        return H
    
    def get_electrode_centers(self):
        """
        Get electrode center positions.
        
        Returns
        -------
        th_center : ndarray
            Angular positions
        z_center : ndarray
            z positions
        """
        return self.th_center, self.z_center


