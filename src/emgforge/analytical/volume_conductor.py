"""
Volume conductor model for cylindrical limb.

This module implements the transfer function computation for a multi-layer
cylindrical volume conductor.
"""

import numpy as np
from scipy.special import iv, kv
import warnings


class CylindricalVolumeConductor:
    """
    Multi-layer cylindrical volume conductor model.
    
    This class computes the transfer function of a cylindrical volume conductor
    with four layers: bone, muscle, fat, and skin.
    
    Parameters
    ----------
    r_bone : float
        Radius of the bone layer (in mm)
    r_muscle : float
        Outer radius of the muscle layer (in mm)
    r_fat : float
        Outer radius of the fat layer (in mm)
    r_skin : float
        Outer radius of the skin layer (total radius) (in mm)
    fiber_depth : float
        Distance of the fiber from the center of the cylinder (in mm)
    conductivities : dict
        Dictionary containing conductivity values for each layer in r, theta, z directions
    v : float
        Conduction velocity (in m/s)
    """
    
    def __init__(self, r_bone, r_muscle, r_fat, r_skin, fiber_depth, 
                 conductivities, v):
        self.a = r_bone
        self.b = r_muscle
        self.c = r_fat
        self.d = r_skin
        self.R = fiber_depth
        self.v = v
        
        # Conductivities for each layer (bone, muscle, fat, skin)
        self.sig1r = conductivities['bone']['r']
        self.sig1th = conductivities['bone']['theta']
        self.sig1z = conductivities['bone']['z']
        
        self.sig2r = conductivities['muscle']['r']
        self.sig2th = conductivities['muscle']['theta']
        self.sig2z = conductivities['muscle']['z']
        
        self.sig3r = conductivities['fat']['r']
        self.sig3th = conductivities['fat']['theta']
        self.sig3z = conductivities['fat']['z']
        
        self.sig4r = conductivities['skin']['r']
        self.sig4th = conductivities['skin']['theta']
        self.sig4z = conductivities['skin']['z']
        
    def compute_transfer_function(self, w=256, fsamp=4096):
        """
        Compute the transfer function of the volume conductor.
        
        Parameters
        ----------
        w : int
            Resolution parameter (default: 256)
        fsamp : int
            Sampling frequency in Hz (default: 4096)
            
        Returns
        -------
        phik : ndarray
            Transfer function of the volume conductor (2D array)
        """
        warnings.filterwarnings('ignore')
        
        nmax = 65
        fmax = fsamp / (2 * self.v * 1000)
        print(f'fmax = {int(fmax * 1000)} m^-1')
        
        fres = 2 * fmax / w
        kz = np.arange((2 * fmax) / w * 2 * np.pi, 
                       fmax * 2 * np.pi + (2 * fmax) / w * 2 * np.pi / 2, 
                       (2 * fmax) / w * 2 * np.pi)
        
        tstep = np.pi / (nmax - 1)
        theta = np.arange(0, 2 * np.pi, tstep)
        st = len(theta)
        
        diam = 25e-3
        s1 = np.sqrt(self.sig1z / self.sig1r)
        s2 = np.sqrt(self.sig2z / self.sig2r)
        s3 = np.sqrt(self.sig3z / self.sig3r)
        s4 = np.sqrt(self.sig4z / self.sig4r)
        
        skz = len(kz)
        
        # Pre-allocate arrays for Bessel functions
        # Need extra space for i+1 access when i can be up to nmax+nmax
        Ia1 = np.zeros((2 * nmax + 2, skz))
        Ia2 = np.zeros((2 * nmax + 2, skz))
        Ib2 = np.zeros((2 * nmax + 2, skz))
        Ib3 = np.zeros((2 * nmax + 2, skz))
        Ic3 = np.zeros((2 * nmax + 2, skz))
        Ic4 = np.zeros((2 * nmax + 2, skz))
        Id4 = np.zeros((2 * nmax + 2, skz))
        IR2 = np.zeros((2 * nmax + 2, skz))
        Ka2 = np.zeros((2 * nmax + 2, skz))
        Kb2 = np.zeros((2 * nmax + 2, skz))
        Kb3 = np.zeros((2 * nmax + 2, skz))
        Kc3 = np.zeros((2 * nmax + 2, skz))
        Kc4 = np.zeros((2 * nmax + 2, skz))
        Kd4 = np.zeros((2 * nmax + 2, skz))
        KR2 = np.zeros((2 * nmax + 2, skz))
        
        # Compute Bessel functions
        for n in range(-1, nmax + 1):
            i = n + nmax
            m1 = n * np.sqrt(self.sig1th / self.sig1r)
            m2 = n * np.sqrt(self.sig2th / self.sig2r)
            m3 = n * np.sqrt(self.sig3th / self.sig3r)
            m4 = n * np.sqrt(self.sig4th / self.sig4r)
            
            Ia1[i, :] = iv(m1, self.a * s1 * kz)
            Ia2[i, :] = iv(m2, self.a * s2 * kz)
            Ib2[i, :] = iv(m2, self.b * s2 * kz)
            Ib3[i, :] = iv(m3, self.b * s3 * kz)
            Ic3[i, :] = iv(m3, self.c * s3 * kz)
            Ic4[i, :] = iv(m4, self.c * s4 * kz)
            Id4[i, :] = iv(m4, self.d * s4 * kz)
            IR2[i, :] = iv(m2, self.R * s2 * kz)
            Ka2[i, :] = kv(m2, self.a * s2 * kz)
            Kb2[i, :] = kv(m2, self.b * s2 * kz)
            Kb3[i, :] = kv(m3, self.b * s3 * kz)
            Kc3[i, :] = kv(m3, self.c * s3 * kz)
            Kc4[i, :] = kv(m4, self.c * s4 * kz)
            Kd4[i, :] = kv(m4, self.d * s4 * kz)
            KR2[i, :] = kv(m2, self.R * s2 * kz)
        
        # Build the linear system
        # Size needs to accommodate the indexing scheme: (nmax-1+nmax)*7 = (2*nmax-1)*7
        max_row_index = (nmax - 1 + nmax) * 7  # This gives us enough rows
        A = np.zeros((max_row_index + 7, 7 * skz))  # Add 7 for safety
        B = np.zeros((max_row_index + 7, skz))
        
        for n in range(0, nmax):
            i = n + nmax + 1  # MATLAB 1-based indexing equivalent
            j = (i - 1) * 7 - 6  # Convert to 0-based: subtract 1 for Python indexing
            
            # Fill matrix A (boundary conditions)
            A[j, 0:skz] = Ia1[i, :] / Ia1[i, :]
            A[j, skz:2*skz] = -Ia2[i, :] / Ib2[i, :]
            A[j, 2*skz:3*skz] = -Ka2[i, :] / Kb2[i, :]
            
            A[j+1, 0:skz] = np.sqrt(self.sig1z * self.sig1r) * (Ia1[i-1, :] + Ia1[i+1, :]) / 2 / Ia1[i, :]
            A[j+1, skz:2*skz] = -np.sqrt(self.sig2z * self.sig2r) * (Ia2[i-1, :] + Ia2[i+1, :]) / 2 / Ib2[i, :]
            A[j+1, 2*skz:3*skz] = -np.sqrt(self.sig2z * self.sig2r) * (-(Ka2[i-1, :] + Ka2[i+1, :])) / 2 / Kb2[i, :]
            
            A[j+2, skz:2*skz] = Ib2[i, :] / Ib2[i, :]
            A[j+2, 2*skz:3*skz] = Kb2[i, :] / Kb2[i, :]
            A[j+2, 3*skz:4*skz] = -Ib3[i, :] / Ic3[i, :]
            A[j+2, 4*skz:5*skz] = -Kb3[i, :] / Kc3[i, :]
            
            A[j+3, skz:2*skz] = np.sqrt(self.sig2z * self.sig2r) * (Ib2[i-1, :] + Ib2[i+1, :]) / 2 / Ib2[i, :]
            A[j+3, 2*skz:3*skz] = np.sqrt(self.sig2z * self.sig2r) * (-(Kb2[i-1, :] + Kb2[i+1, :])) / 2 / Kb2[i, :]
            A[j+3, 3*skz:4*skz] = -np.sqrt(self.sig3z * self.sig3r) * (Ib3[i-1, :] + Ib3[i+1, :]) / 2 / Ic3[i, :]
            A[j+3, 4*skz:5*skz] = -np.sqrt(self.sig3z * self.sig3r) * (-(Kb3[i-1, :] + Kb3[i+1, :])) / 2 / Kc3[i, :]
            
            A[j+4, 3*skz:4*skz] = Ic3[i, :] / Ic3[i, :]
            A[j+4, 4*skz:5*skz] = Kc3[i, :] / Kc3[i, :]
            A[j+4, 5*skz:6*skz] = -Ic4[i, :] / Id4[i, :]
            A[j+4, 6*skz:7*skz] = -Kc4[i, :] / Kd4[i, :]
            
            A[j+5, 3*skz:4*skz] = np.sqrt(self.sig3z * self.sig3r) * (Ic3[i-1, :] + Ic3[i+1, :]) / 2 / Ic3[i, :]
            A[j+5, 4*skz:5*skz] = np.sqrt(self.sig3z * self.sig3r) * (-(Kc3[i-1, :] + Kc3[i+1, :])) / 2 / Kc3[i, :]
            A[j+5, 5*skz:6*skz] = -np.sqrt(self.sig4z * self.sig4r) * (Ic4[i-1, :] + Ic4[i+1, :]) / 2 / Id4[i, :]
            A[j+5, 6*skz:7*skz] = -np.sqrt(self.sig4z * self.sig4r) * (-(Kc4[i-1, :] + Kc4[i+1, :])) / 2 / Kd4[i, :]
            
            A[j+6, 5*skz:6*skz] = np.sqrt(self.sig4z * self.sig4r) * (Id4[i-1, :] + Id4[i+1, :]) / 2 / Id4[i, :]
            A[j+6, 6*skz:7*skz] = np.sqrt(self.sig4z * self.sig4r) * (-(Kd4[i-1, :] + Kd4[i+1, :])) / 2 / Kd4[i, :]
            
            # Fill vector B (source terms)
            B[j, :] = (Ia2[i, :] * KR2[i, :]) / self.sig2r
            B[j+1, :] = s2 * (Ia2[i-1, :] + Ia2[i+1, :]) / 2 * KR2[i, :]
            B[j+2, :] = -(IR2[i, :] * Kb2[i, :]) / self.sig2r
            B[j+3, :] = -s2 * IR2[i, :] * (-(Kb2[i-1, :] + Kb2[i+1, :]) / 2)
            B[j+4, :] = 0
            B[j+5, :] = 0
            B[j+6, :] = 0
        
        B = diam * B
        
        # For transfer function computation (q==1 in MATLAB code)
        spe2 = np.ones(w // 2)
        
        # Solve the linear system for each frequency
        coeff = np.zeros_like(B)  # Same size as B
        for jj in range(skz):
            for n in range(0, nmax):
                i = 7 * (n + nmax) - 6  # MATLAB: i = 7*(n+nmax)-6, convert to 0-based
                Ank = A[i:i+7, jj::skz]
                Bnk = B[i:i+7, jj] * spe2[jj]
                coeff[i:i+7, jj] = np.linalg.solve(Ank, Bnk)
        
        # Compute potential over the skin
        phi4k = np.zeros((nmax, skz))
        for n in range(0, nmax):
            i = n + nmax + 1  # MATLAB uses 1-based indexing
            phi4k[n, :] = coeff[(i-1) * 7 - 1, :] + coeff[(i-1) * 7, :]
        
        # Construct the full transfer function
        mat = np.zeros((2 * nmax - 2, skz))
        # MATLAB line 160 assigns to 'at' but it appears to be intended for mat
        mat[0:nmax, :] = np.flipud(phi4k[0:nmax, :])
        # MATLAB line 161
        mat[nmax:2*(nmax-1), :] = phi4k[1:nmax-1, :]
        
        # Add zero column for kz = 0
        colonna = np.zeros((2 * nmax - 2, 1))
        
        # Fourier transform: -kzmax <= kz <= kzmax-Dkz, -nmax+1 <= ktheta <= nmax-2
        phik = np.hstack([np.fliplr(mat), colonna, mat[:, 0:skz-1]])
        
        # Return without transpose - main function will transpose
        return phik

    def compute_transfer_function_debug(self, w=256, fsamp=4096, include_bessel_tables=True):
        """Compute the transfer function and return intermediate matrices for debugging/visualization."""
        debug = {}

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')

            nmax = 65
            fmax = fsamp / (2 * self.v * 1000)
            print(f'fmax = {int(fmax * 1000)} m^-1')

            fres = 2 * fmax / w
            kz = np.arange((2 * fmax) / w * 2 * np.pi,
                           fmax * 2 * np.pi + (2 * fmax) / w * 2 * np.pi / 2,
                           (2 * fmax) / w * 2 * np.pi)

            tstep = np.pi / (nmax - 1)
            theta = np.arange(0, 2 * np.pi, tstep)
            st = len(theta)

            diam = 25e-3
            s1 = np.sqrt(self.sig1z / self.sig1r)
            s2 = np.sqrt(self.sig2z / self.sig2r)
            s3 = np.sqrt(self.sig3z / self.sig3r)
            s4 = np.sqrt(self.sig4z / self.sig4r)

            skz = len(kz)

            Ia1 = np.zeros((2 * nmax + 2, skz))
            Ia2 = np.zeros((2 * nmax + 2, skz))
            Ib2 = np.zeros((2 * nmax + 2, skz))
            Ib3 = np.zeros((2 * nmax + 2, skz))
            Ic3 = np.zeros((2 * nmax + 2, skz))
            Ic4 = np.zeros((2 * nmax + 2, skz))
            Id4 = np.zeros((2 * nmax + 2, skz))
            IR2 = np.zeros((2 * nmax + 2, skz))
            Ka2 = np.zeros((2 * nmax + 2, skz))
            Kb2 = np.zeros((2 * nmax + 2, skz))
            Kb3 = np.zeros((2 * nmax + 2, skz))
            Kc3 = np.zeros((2 * nmax + 2, skz))
            Kc4 = np.zeros((2 * nmax + 2, skz))
            Kd4 = np.zeros((2 * nmax + 2, skz))
            KR2 = np.zeros((2 * nmax + 2, skz))

            for n in range(-1, nmax + 1):
                i = n + nmax
                m1 = n * np.sqrt(self.sig1th / self.sig1r)
                m2 = n * np.sqrt(self.sig2th / self.sig2r)
                m3 = n * np.sqrt(self.sig3th / self.sig3r)
                m4 = n * np.sqrt(self.sig4th / self.sig4r)

                Ia1[i, :] = iv(m1, self.a * s1 * kz)
                Ia2[i, :] = iv(m2, self.a * s2 * kz)
                Ib2[i, :] = iv(m2, self.b * s2 * kz)
                Ib3[i, :] = iv(m3, self.b * s3 * kz)
                Ic3[i, :] = iv(m3, self.c * s3 * kz)
                Ic4[i, :] = iv(m4, self.c * s4 * kz)
                Id4[i, :] = iv(m4, self.d * s4 * kz)
                IR2[i, :] = iv(m2, self.R * s2 * kz)
                Ka2[i, :] = kv(m2, self.a * s2 * kz)
                Kb2[i, :] = kv(m2, self.b * s2 * kz)
                Kb3[i, :] = kv(m3, self.b * s3 * kz)
                Kc3[i, :] = kv(m3, self.c * s3 * kz)
                Kc4[i, :] = kv(m4, self.c * s4 * kz)
                Kd4[i, :] = kv(m4, self.d * s4 * kz)
                KR2[i, :] = kv(m2, self.R * s2 * kz)

            max_row_index = (nmax - 1 + nmax) * 7
            A = np.zeros((max_row_index + 7, 7 * skz))
            B = np.zeros((max_row_index + 7, skz))

            for n in range(0, nmax):
                i = n + nmax + 1
                j = (i - 1) * 7 - 6

                A[j, 0:skz] = Ia1[i, :] / Ia1[i, :]
                A[j, skz:2*skz] = -Ia2[i, :] / Ib2[i, :]
                A[j, 2*skz:3*skz] = -Ka2[i, :] / Kb2[i, :]

                A[j+1, 0:skz] = (self.sig1z * self.sig1r) ** 0.5 * (Ia1[i-1, :] + Ia1[i+1, :]) / 2 / Ia1[i, :]
                A[j+1, skz:2*skz] = -(self.sig2z * self.sig2r) ** 0.5 * (Ia2[i-1, :] + Ia2[i+1, :]) / 2 / Ib2[i, :]
                A[j+1, 2*skz:3*skz] = -(self.sig2z * self.sig2r) ** 0.5 * (-(Ka2[i-1, :] + Ka2[i+1, :])) / 2 / Kb2[i, :]

                A[j+2, skz:2*skz] = Ib2[i, :] / Ib2[i, :]
                A[j+2, 2*skz:3*skz] = Kb2[i, :] / Kb2[i, :]
                A[j+2, 3*skz:4*skz] = -Ib3[i, :] / Ic3[i, :]
                A[j+2, 4*skz:5*skz] = -Kb3[i, :] / Kc3[i, :]

                A[j+3, skz:2*skz] = (self.sig2z * self.sig2r) ** 0.5 * (Ib2[i-1, :] + Ib2[i+1, :]) / 2 / Ib2[i, :]
                A[j+3, 2*skz:3*skz] = (self.sig2z * self.sig2r) ** 0.5 * (-(Kb2[i-1, :] + Kb2[i+1, :])) / 2 / Kb2[i, :]
                A[j+3, 3*skz:4*skz] = -(self.sig3z * self.sig3r) ** 0.5 * (Ib3[i-1, :] + Ib3[i+1, :]) / 2 / Ic3[i, :]
                A[j+3, 4*skz:5*skz] = -(self.sig3z * self.sig3r) ** 0.5 * (-(Kb3[i-1, :] + Kb3[i+1, :])) / 2 / Kc3[i, :]

                A[j+4, 3*skz:4*skz] = Ic3[i, :] / Ic3[i, :]
                A[j+4, 4*skz:5*skz] = Kc3[i, :] / Kc3[i, :]
                A[j+4, 5*skz:6*skz] = -Ic4[i, :] / Id4[i, :]
                A[j+4, 6*skz:7*skz] = -Kc4[i, :] / Kd4[i, :]

                A[j+5, 3*skz:4*skz] = (self.sig3z * self.sig3r) ** 0.5 * (Ic3[i-1, :] + Ic3[i+1, :]) / 2 / Ic3[i, :]
                A[j+5, 4*skz:5*skz] = (self.sig3z * self.sig3r) ** 0.5 * (-(Kc3[i-1, :] + Kc3[i+1, :])) / 2 / Kc3[i, :]
                A[j+5, 5*skz:6*skz] = -(self.sig4z * self.sig4r) ** 0.5 * (Ic4[i-1, :] + Ic4[i+1, :]) / 2 / Id4[i, :]
                A[j+5, 6*skz:7*skz] = -(self.sig4z * self.sig4r) ** 0.5 * (-(Kc4[i-1, :] + Kc4[i+1, :])) / 2 / Kd4[i, :]

                A[j+6, 5*skz:6*skz] = (self.sig4z * self.sig4r) ** 0.5 * (Id4[i-1, :] + Id4[i+1, :]) / 2 / Id4[i, :]
                A[j+6, 6*skz:7*skz] = (self.sig4z * self.sig4r) ** 0.5 * (-(Kd4[i-1, :] + Kd4[i+1, :])) / 2 / Kd4[i, :]

                B[j, :] = (Ia2[i, :] * KR2[i, :]) / self.sig2r
                B[j+1, :] = s2 * (Ia2[i-1, :] + Ia2[i+1, :]) / 2 * KR2[i, :]
                B[j+2, :] = -(IR2[i, :] * Kb2[i, :]) / self.sig2r
                B[j+3, :] = -s2 * IR2[i, :] * (-(Kb2[i-1, :] + Kb2[i+1, :]) / 2)
                B[j+4, :] = 0
                B[j+5, :] = 0
                B[j+6, :] = 0

            B = diam * B
            spe2 = np.ones(w // 2)

            coeff = np.zeros_like(B)
            for jj in range(skz):
                for n in range(0, nmax):
                    i0 = 7 * (n + nmax) - 6
                    Ank = A[i0:i0+7, jj::skz]
                    Bnk = B[i0:i0+7, jj] * spe2[jj]
                    coeff[i0:i0+7, jj] = np.linalg.solve(Ank, Bnk)

            phi4k = np.zeros((nmax, skz))
            for n in range(0, nmax):
                i = n + nmax + 1
                phi4k[n, :] = coeff[(i-1) * 7 - 1, :] + coeff[(i-1) * 7, :]

            mat = np.zeros((2 * nmax - 2, skz))
            mat[0:nmax, :] = np.flipud(phi4k[0:nmax, :])
            mat[nmax:2*(nmax-1), :] = phi4k[1:nmax-1, :]

            colonna = np.zeros((2 * nmax - 2, 1))
            phik = np.hstack([np.fliplr(mat), colonna, mat[:, 0:skz-1]])

        debug.update({
            'w': w,
            'fsamp': fsamp,
            'nmax': nmax,
            'fmax': fmax,
            'fres': fres,
            'kz': kz,
            'tstep': tstep,
            'theta': theta,
            'st': st,
            'skz': skz,
            'A': A,
            'B': B,
            'coeff': coeff,
            'phi4k': phi4k,
            'mat': mat,
            'phik': phik,
        })

        if include_bessel_tables:
            debug['bessel'] = {
                'Ia1': Ia1, 'Ia2': Ia2, 'Ib2': Ib2, 'Ib3': Ib3, 'Ic3': Ic3, 'Ic4': Ic4, 'Id4': Id4,
                'IR2': IR2,
                'Ka2': Ka2, 'Kb2': Kb2, 'Kb3': Kb3, 'Kc3': Kc3, 'Kc4': Kc4, 'Kd4': Kd4,
                'KR2': KR2,
            }

        return debug

