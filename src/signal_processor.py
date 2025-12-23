import numpy as np
from scipy.optimize import curve_fit

class SignalProcessor:
    def __init__(self):
        pass

    def _gaussian(self, t, A, mu, sigma, C):
        """Single Gaussian model."""
        return A * np.exp(-(t - mu)**2 / (2 * sigma**2)) + C

    def _multi_gaussian_3(self, t, *params):
        """Sum of 3 Gaussians + Constant background. Params size: 3*3 + 1 = 10"""
        A1, mu1, s1, A2, mu2, s2, A3, mu3, s3, C = params
        g1 = A1 * np.exp(-(t - mu1)**2 / (2 * s1**2))
        g2 = A2 * np.exp(-(t - mu2)**2 / (2 * s2**2))
        g3 = A3 * np.exp(-(t - mu3)**2 / (2 * s3**2))
        return g1 + g2 + g3 + C

    def get_amplitude_region1(self, time, signal):
        """Fits single Gaussian to Region 1."""
        try:
            # Initial guess: Amplitude=max, mu=time of max, sigma=small, C=min
            p0 = [np.max(signal), time[np.argmax(signal)], 0.001, np.min(signal)]
            
            popt, _ = curve_fit(self._gaussian, time, signal, p0=p0, maxfev=2000)
            return abs(popt[0]) # Amplitude A
        except Exception:
            return np.nan # Fit failed

    def get_amplitude_region2(self, time, signal):
        """Fits 3 Gaussians to Region 2 and returns amplitude of the FIRST pulse."""
        try:
            # Heuristic guesses for 3 pulses
            t_len = time[-1] - time[0]
            start = time[0]
            
            # Guess 3 peaks spaced out: Start, Start + 1/3, Start + 2/3
            p0 = [
                np.max(signal), start + t_len*0.1, 0.001,  # Pulse 1 (Target)
                np.max(signal), start + t_len*0.3, 0.001,  # Pulse 2
                np.max(signal), start + t_len*0.6, 0.001,  # Pulse 3
                np.min(signal)                             # Offset
            ]
            
            popt, _ = curve_fit(self._multi_gaussian_3, time, signal, p0=p0, maxfev=5000)
            
            # Extract (A, mu) pairs
            pulses = [
                (popt[0], popt[1]), # A1, mu1
                (popt[3], popt[4]), # A2, mu2
                (popt[6], popt[7])  # A3, mu3
            ]
            
            # Sort by mu (time) to find the EARLIEST pulse
            pulses.sort(key=lambda x: x[1])
            
            # Return Amplitude of earliest pulse
            return abs(pulses[0][0])
            
        except Exception:
            return np.nan