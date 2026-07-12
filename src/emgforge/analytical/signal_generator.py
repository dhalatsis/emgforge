"""
Signal generator for MUAP synthesis.

This module handles the generation of Motor Unit Action Potentials (MUAPs)
by combining volume conductor transfer function, motor unit geometry,
and detection system properties.
"""

import numpy as np


class SignalGenerator:
    """
    Signal generator for MUAP synthesis.
    
    This class combines the volume conductor, motor unit, and detection
    system to generate surface EMG signals.
    
    Parameters
    ----------
    volume_conductor : CylindricalVolumeConductor
        Volume conductor model
    motor_unit : MotorUnit
        Motor unit model
    detection_system : DetectionSystem
        Detection system model
    v : float
        Conduction velocity (in m/s)
    fsamp : int
        Sampling frequency (in Hz)
    w : int
        Resolution parameter (default: 256)
    """
    
    def __init__(self, volume_conductor, motor_unit, detection_system, v, fsamp, w=256):
        self.vc = volume_conductor
        self.mu = motor_unit
        self.det = detection_system
        self.v = v
        self.fsamp = fsamp
        self.w = w
        
        # Compute frequency parameters
        self.fm = fsamp / (2 * v * 1000)
        self.tstep = 2 * np.pi / 128
        self.zstep = 1 / (2 * self.fm)
        
    def _radon_transform(self, fx, fy, H, section):
        """
        Implement the Radon transform to obtain a section of 2-D function.
        
        This function implements the Radon transform as described in:
        D. Farina and R. Merletti, "A novel approach for precise simulation 
        of the EMG signal detected by surface electrodes", 
        IEEE Trans. Biomed. Eng., vol. 48, pp. 637-646, 2001
        
        Parameters
        ----------
        fx : ndarray
            Frequency vector in x direction
        fy : ndarray
            Frequency vector in y direction
        H : ndarray
            2-D Fourier transform of the function
        section : float
            The coordinate of the computed section
            
        Returns
        -------
        sig : ndarray
            The section in the x direction
        """
        F = H * np.exp(1j * 2 * np.pi * fy * section)
        sig = np.real(np.fft.fftshift(np.fft.ifft(np.fft.fftshift(np.sum(F, axis=0)))) / len(H)) * (fy[1, 0] - fy[0, 0])
        return sig
    
    def generate_muap(self):
        """
        Generate Motor Unit Action Potentials (MUAPs) for all channels.
        
        Returns
        -------
        t : ndarray
            Time vector (in ms)
        sig_arr : ndarray
            Matrix with detected signals (one for each row/channel)
        MU : ndarray
            Positions of the fibers of the MU in the muscle
        """
        # Compute volume conductor transfer function
        print("Computing volume conductor transfer function...")
        Htissue = self.vc.compute_transfer_function(w=self.w, fsamp=self.fsamp)
        Htissue = Htissue.T
        
        # Get fiber information
        fiber_info = self.mu.get_fiber_info()
        positions = fiber_info['positions']
        fiber_pos_th = fiber_info['fiber_pos_th']
        len1 = fiber_info['len1']
        len2 = fiber_info['len2']
        posz = positions[:, 2]
        Nfib = fiber_info['n_fibers']
        
        # Get electrode centers
        th_center, z_center = self.det.get_electrode_centers()
        channels = self.det.channels
        
        # Define spatial frequencies
        ktheta, kz = np.meshgrid(
            np.arange(-1 / (2 * self.tstep) * 2 * np.pi, 
                     1 / (2 * self.tstep) * 2 * np.pi, 
                     1 / (2 * np.pi) * 2 * np.pi),
            np.arange(-1 / (2 * self.zstep) * 2 * np.pi, 
                     (1 / (2 * self.zstep) - 1 / (self.w / (2 * self.fm))) * 2 * np.pi + 2 * np.pi / (self.w / (2 * self.fm)), 
                     1 / (self.w / (2 * self.fm)) * 2 * np.pi)
        )
        
        # Define intracellular action potential
        kzz = 2 * np.pi * np.arange(-2, 2, (2 * self.fm) / self.w)
        z = np.arange(0, 15.25, 0.25)
        V2 = 96 * (np.exp(-z) * (3 * z**2 - z**3))
        V2 = np.concatenate([V2, np.zeros(len(kzz) - len(z))])
        V2 = -np.flip(V2)
        spe2 = np.fft.fftshift(np.fft.fft(V2))
        spe2 = spe2[len(spe2) // 2 - self.w // 2:len(spe2) // 2 + self.w // 2]
        
        # Compute detection system filter
        H = self.det.compute_complete_filter(ktheta, kz, self.w)
        
        # Initialize arrays
        E = [np.zeros((self.w, self.w), dtype=complex) for _ in range(channels)]
        C = [{} for _ in range(channels)]
        
        # Compute transfer function for each fiber
        print(f"Processing {Nfib} fibers...")
        for u in range(Nfib):
            # MATLAB check: size(Htissue)~=w checks if any dimension doesn't equal w
            # But based on the frequency grid, Htissue should be (256, 128)
            # if Htissue.shape[0] != self.w:
            #     raise ValueError(f'Transfer function has wrong first dimension: {Htissue.shape[0]} != {self.w}')
            
            Phi_mat = Htissue * H
            
            for channel in range(channels):
                Phi_mat_true = Phi_mat * np.exp(1j * (fiber_pos_th + th_center[channel]) * ktheta)
                # MATLAB: sum(Phi_mat_true') sums over the transposed matrix
                # This is equivalent to summing over axis 1 (columns)
                C[channel][u] = np.sum(Phi_mat_true.T, axis=0)
        
        # Define temporal-spatial frequencies
        kz_t, kt = np.meshgrid(
            np.arange(-self.fm * 2 * np.pi, 
                     self.fm * (1 - 2 / self.w) * 2 * np.pi + (2 * self.fm) / self.w * 2 * np.pi, 
                     (2 * self.fm) / self.w * 2 * np.pi),
            np.arange(-self.fm * 2 * np.pi * self.v, 
                     self.fm * (1 - 2 / self.w) * 2 * np.pi * self.v + (2 * self.fm) / self.w * 2 * np.pi * self.v, 
                     (2 * self.fm) / self.w * 2 * np.pi * self.v)
        )
        
        kalpha = kz_t + kt / self.v
        kbeta = kz_t - kt / self.v
        
        # Compute signals
        print("Generating signals...")
        for u in range(Nfib):
            pare = (np.exp(-1j * len1[u] / 2 * kalpha) * len1[u] * np.sinc(len1[u] / 2 * kalpha / np.pi) - 
                   np.exp(1j * len2[u] / 2 * kbeta) * len2[u] * np.sinc(len2[u] / 2 * kbeta / np.pi))
            
            for channel in range(channels):
                E[channel] = E[channel] + pare * np.exp(1j * kz_t * posz[u]) * (np.ones((self.w, 1)) @ C[channel][u].reshape(1, -1))
        
        # Generate final signals
        sig_arr = np.zeros((channels, self.w))
        for channel in range(channels):
            E1 = 1 / self.v * (np.ones((self.w, 1)) @ spe2.reshape(1, -1)).T * E[channel] * (1j * kz_t)
            sig_arr[channel, :] = np.flip(self._radon_transform(kt.T / (2 * np.pi), kz_t.T / (2 * np.pi), 
                                                                 E1.T, z_center[channel]))
        
        # Generate time vector
        t = np.arange(0, self.w / self.fsamp * 1000, 1 / self.fsamp * 1000)
        
        return t, sig_arr, positions


    def generate_muap_debug(self, max_fibers=1, max_channels=4, seed=0):
        """        Debug/teaching version of `generate_muap()`.

        Returns a dict with intermediate matrices so you can plot/inspect them.

        Notes
        -----
        - To keep memory reasonable, this limits the number of fibers/channels stored.
        - The forward model is still computed the same way as `generate_muap()`.
        """
        if seed is not None:
            np.random.seed(seed)

        # Volume conductor transfer function (k_theta, k_z)
        Htissue = self.vc.compute_transfer_function(w=self.w, fsamp=self.fsamp).T

        # Fiber information
        fiber_info = self.mu.get_fiber_info()
        positions = fiber_info['positions']
        fiber_pos_th = fiber_info['fiber_pos_th']
        len1 = fiber_info['len1']
        len2 = fiber_info['len2']
        posz = positions[:, 2]
        Nfib = int(fiber_info['n_fibers'])

        # Electrode centers
        th_center, z_center = self.det.get_electrode_centers()
        channels = int(self.det.channels)

        # Spatial frequency grids used for 2-D spatial filtering
        ktheta, kz = np.meshgrid(
            np.arange(-1 / (2 * self.tstep) * 2 * np.pi,
                      1 / (2 * self.tstep) * 2 * np.pi,
                      1 / (2 * np.pi) * 2 * np.pi),
            np.arange(-1 / (2 * self.zstep) * 2 * np.pi,
                      (1 / (2 * self.zstep) - 1 / (self.w / (2 * self.fm))) * 2 * np.pi + 2 * np.pi / (self.w / (2 * self.fm)),
                      1 / (self.w / (2 * self.fm)) * 2 * np.pi)
        )

        # Intracellular action potential spectrum (1-D)
        kzz = 2 * np.pi * np.arange(-2, 2, (2 * self.fm) / self.w)
        z = np.arange(0, 15.25, 0.25)
        V2 = 96 * (np.exp(-z) * (3 * z**2 - z**3))
        V2 = np.concatenate([V2, np.zeros(len(kzz) - len(z))])
        V2 = -np.flip(V2)
        spe2 = np.fft.fftshift(np.fft.fft(V2))
        spe2 = spe2[len(spe2) // 2 - self.w // 2:len(spe2) // 2 + self.w // 2]

        # Detection system transfer function
        H = self.det.compute_complete_filter(ktheta, kz, self.w)

        # Per-fiber “spatial transfer” collapsed over k_theta to 1-D in kz
        Phi_mat = Htissue * H

        # Limit debug storage
        nf_keep = min(max_fibers, Nfib)
        ch_keep = min(max_channels, channels)

        C = [[None for _ in range(nf_keep)] for _ in range(ch_keep)]
        for u in range(nf_keep):
            for ch in range(ch_keep):
                Phi_mat_true = Phi_mat * np.exp(1j * (fiber_pos_th + th_center[ch]) * ktheta)
                C[ch][u] = np.sum(Phi_mat_true.T, axis=0)  # length w

        # Temporal/spatial frequency grids (k_t, k_z)
        kz_t, kt = np.meshgrid(
            np.arange(-self.fm * 2 * np.pi,
                      self.fm * (1 - 2 / self.w) * 2 * np.pi + (2 * self.fm) / self.w * 2 * np.pi,
                      (2 * self.fm) / self.w * 2 * np.pi),
            np.arange(-self.fm * 2 * np.pi * self.v,
                      self.fm * (1 - 2 / self.w) * 2 * np.pi * self.v + (2 * self.fm) / self.w * 2 * np.pi * self.v,
                      (2 * self.fm) / self.w * 2 * np.pi * self.v)
        )

        kalpha = kz_t + kt / self.v
        kbeta = kz_t - kt / self.v

        # Build E(channel) in the (k_t, k_z) domain
        E = [np.zeros((self.w, self.w), dtype=complex) for _ in range(ch_keep)]
        for u in range(nf_keep):
            pare = (
                np.exp(-1j * len1[u] / 2 * kalpha) * len1[u] * np.sinc(len1[u] / 2 * kalpha / np.pi)
                - np.exp(1j * len2[u] / 2 * kbeta) * len2[u] * np.sinc(len2[u] / 2 * kbeta / np.pi)
            )
            for ch in range(ch_keep):
                E[ch] = E[ch] + pare * np.exp(1j * kz_t * posz[u]) * (np.ones((self.w, 1)) @ np.asarray(C[ch][u]).reshape(1, -1))

        # Convert to final detected MUAP signals per channel via Radon section
        sig_arr = np.zeros((ch_keep, self.w))
        E1_list = []
        for ch in range(ch_keep):
            E1 = 1 / self.v * (np.ones((self.w, 1)) @ spe2.reshape(1, -1)).T * E[ch] * (1j * kz_t)
            E1_list.append(E1)
            sig_arr[ch, :] = np.flip(self._radon_transform(
                kt.T / (2 * np.pi),
                kz_t.T / (2 * np.pi),
                E1.T,
                z_center[ch]
            ))

        t = np.arange(0, self.w / self.fsamp * 1000, 1 / self.fsamp * 1000)

        return {
            'Htissue': Htissue,
            'H': H,
            'Phi_mat': Phi_mat,
            'ktheta': ktheta,
            'kz': kz,
            'kz_t': kz_t,
            'kt': kt,
            'kalpha': kalpha,
            'kbeta': kbeta,
            'spe2': spe2,
            'V2': V2,
            'C': C,
            'E': E,
            'E1': E1_list,
            'sig_arr': sig_arr,
            't': t,
            'positions': positions,
            'th_center': th_center,
            'z_center': z_center,
            'nf_keep': nf_keep,
            'ch_keep': ch_keep,
        }

