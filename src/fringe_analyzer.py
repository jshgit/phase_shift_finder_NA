import numpy as np
from scipy.optimize import curve_fit

class FringeAnalyzer:
    def __init__(self, **kwargs):
        # Allow passing config dicts safely
        pass

    def _sine_model(self, x_deg, A, phi, C):
        x_rad = np.deg2rad(x_deg)
        return C + A * np.sin(x_rad + phi)

    def fit(self, scan_params, ratios, ratio_errs, r1_r2, r2_r2, 
            r2_threshold=0.8, uncertainty_reject_count=2, residual_z_score=2.5):
        """
        Fits sine wave with strict ordered rejection:
        1. REJECT: R^2 < r2_threshold (poor trace fits).
        2. REJECT: Residual Z-Score (Ratio Outliers) based on preliminary fit.
        3. REJECT: Top 'uncertainty_reject_count' highest uncertainties from REMAINING.
        4. FIT: Final model on remaining data.
        """
        # Ensure inputs are numpy arrays
        scan_params = np.asarray(scan_params)
        ratios = np.asarray(ratios)
        ratio_errs = np.asarray(ratio_errs)
        r1_r2 = np.asarray(r1_r2)
        r2_r2 = np.asarray(r2_r2)

        # 0. Basic Finite Mask
        valid_mask = np.isfinite(ratios) & np.isfinite(ratio_errs) & (ratio_errs > 0)
        
        # --- STAGE 1: R^2 Rejection (Signal Quality) ---
        r1_valid = np.nan_to_num(r1_r2, nan=-1.0)
        r2_valid = np.nan_to_num(r2_r2, nan=-1.0)
        r2_mask = (r1_valid >= r2_threshold) & (r2_valid >= r2_threshold)
        
        # Identify R2 failures for reporting
        mask_rej_r2 = valid_mask & (~r2_mask)
        
        # Update valid mask (Filter Step 1)
        valid_mask = valid_mask & r2_mask
        
        if np.sum(valid_mask) < 4:
            return None

        # Prepare for Coarse Fit (Ratio Outlier Detection)
        x = scan_params[valid_mask]
        y = ratios[valid_mask]
        sigma = ratio_errs[valid_mask]

        try:
            guess_A = (np.max(y) - np.min(y)) / 2
            guess_C = np.mean(y)
            guess_phi = 0.0
            p0 = [guess_A, guess_phi, guess_C]

            # --- COARSE FIT ---
            popt_1, _ = curve_fit(
                self._sine_model, x, y, p0=p0, sigma=sigma, absolute_sigma=True, maxfev=5000
            )

            # --- STAGE 2: Ratio Outlier Rejection ---
            y_model = self._sine_model(x, *popt_1)
            residuals = y - y_model
            resid_std = np.std(residuals)
            
            # Identify outliers in the currently valid set
            is_outlier = np.abs(residuals) > (residual_z_score * resid_std + 1e-12)
            
            # Map back to global indices
            full_indices = np.arange(len(ratios))
            current_indices = full_indices[valid_mask]
            rejected_ratio_idx = current_indices[is_outlier]
            mask_rej_ratio = np.isin(full_indices, rejected_ratio_idx)

            # Update valid mask (Filter Step 2)
            valid_mask = valid_mask & (~mask_rej_ratio)

            if np.sum(valid_mask) < 4:
                return None

            # --- STAGE 3: Uncertainty Rejection ---
            # Rank based on uncertainty of the *remaining* ratios
            indices_remaining = np.where(valid_mask)[0]
            
            if len(indices_remaining) > uncertainty_reject_count and uncertainty_reject_count > 0:
                current_errs = ratio_errs[indices_remaining]
                # Find indices of top N highest uncertainty
                worst_local_idx = np.argsort(current_errs)[-uncertainty_reject_count:]
                rejected_unc_idx = indices_remaining[worst_local_idx]
                
                mask_rej_unc = np.isin(full_indices, rejected_unc_idx)
                
                # Update valid mask (Filter Step 3)
                valid_mask = valid_mask & (~mask_rej_unc)
            else:
                mask_rej_unc = np.zeros_like(ratios, dtype=bool)

            if np.sum(valid_mask) < 4:
                return None

            # --- FINAL FIT ---
            x_final = scan_params[valid_mask]
            y_final = ratios[valid_mask]
            sigma_final = ratio_errs[valid_mask]

            popt_final, pcov_final = curve_fit(
                self._sine_model, x_final, y_final, p0=popt_1, sigma=sigma_final, absolute_sigma=True, maxfev=5000
            )

            perr = np.sqrt(np.diag(pcov_final))
            
            return {
                "phase_rad": float(popt_final[1]),
                "phase_err": float(perr[1]),
                "amplitude": float(popt_final[0]),
                "offset": float(popt_final[2]),
                "params": popt_final,
                "pcov": pcov_final,
                "n_used": int(len(x_final)),
                "valid_mask": valid_mask,
                "mask_rej_unc": mask_rej_unc,
                "mask_rej_ratio": mask_rej_ratio,
                "mask_rej_r2": mask_rej_r2
            }

        except Exception as e:
            # print(f"Fringe fit failed: {e}") # Optional: suppress print to keep logs clean
            return None