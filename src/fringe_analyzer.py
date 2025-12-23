import numpy as np
from scipy.optimize import curve_fit

class FringeAnalyzer:
    def __init__(self):
        pass

    def _sine_model(self, x_deg, A, phi, offset):
        """
        Model: y = Offset + A * sin(x_rad + phi)
        Note: Frequency is fixed by the physics (1 cycle per 360 deg).
        """
        x_rad = np.deg2rad(x_deg)
        return offset + A * np.sin(x_rad + phi)

    def estimate_phase(self, scan_params, ratios):
        """
        Fits sine wave to Data vs Ratio.
        Returns: phase (mrad), uncertainty (mrad)
        """
        # Remove NaNs (failed fits from previous step)
        valid_mask = ~np.isnan(ratios)
        x_clean = scan_params[valid_mask]
        y_clean = ratios[valid_mask]

        if len(x_clean) < 10: # Safety check
            return np.nan, np.nan

        try:
            # Guesses: A=(max-min)/2, phi=0, offset=mean
            p0 = [
                (np.max(y_clean) - np.min(y_clean)) / 2,
                0,
                np.mean(y_clean)
            ]
            
            popt, pcov = curve_fit(self._sine_model, x_clean, y_clean, p0=p0)
            
            phase_rad = popt[1]
            perr = np.sqrt(np.diag(pcov))
            phase_uncertainty_rad = perr[1]
            
            # Convert to milliradians
            return phase_rad * 1000, phase_uncertainty_rad * 1000
            
        except Exception:
            return np.nan, np.nan