import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


class SignalProcessor:
    """Gaussian pulse fitter + ratio estimator.

    This version expresses *all* key bounds and initial guesses as multiples of dt
    (the sample spacing), so it’s easy to see feasibility (p0 inside bounds).

    Notes:
    - `dt` is computed from the provided `time` array per-trace.
    - We still allow partial-capture behavior by allowing mu to extend a bit
      outside the fit window, but those extensions are now specified in dt units.
    """

    def __init__(
        self,
        region2_model="single",
        multi_pulse_return_mode="first",
        fit_window=None,
    ):
        self.region2_model = region2_model
        self.multi_pulse_return_mode = multi_pulse_return_mode
        self.fit_window = fit_window

        # ===== Peak-search grid (for sum peak) =====
        self.sum_peak_grid_mult = 5  # densify peak search grid

        # ===== GLOBAL dt-based safety margins =====
        self.mu_global_margin_dt = 50        # global mu allowed outside window by this many dt
        self.sigma_min_dt = 2.0              # sigma lower bound = sigma_min_dt * dt

        # ===== REGION 1 (single gaussian) dt-based bounds & initial guesses =====
        # Mu bounds are anchored to the edge (passed in), not to window length.
        self.r1_mu_lo_extra_dt = 80          # allow mu this far before window start
        self.r1_mu_hi_after_edge_dt = 30     # allow mu this far AFTER t_edge (partial capture)

        self.r1_sigma_hi_dt = 150            # sigma upper bound
        self.r1_sigma0_dt = 20               # initial sigma guess

        # ===== REGION 2 (triple) dt-based priors =====
        # Two near-edge narrow peaks + one later broader peak.
        # These are all *multipliers of dt*.
        #
        # Mu bounds relative to edge:
        self.r2_mu1_span_dt = 5             # mu1 in [edge - span, edge + span]
        self.r2_mu2_before_edge_dt = 0       # mu2 can start a bit before edge
        self.r2_mu2_after_edge_dt = 10       # mu2 can be within this many dt after edge

        self.r2_mu3_lo_dt = 40              # mu3 lower bound = edge + this*dt
        self.r2_mu3_hi_dt = 70              # mu3 upper bound = edge + this*dt

        # Sigma bounds:
        self.r2_s1_hi_dt = 10                # sigma1 upper bound (narrow)
        self.r2_s2_hi_dt = 16                # sigma2 upper bound (narrow-ish)
        self.r2_s3_hi_dt = 50                # sigma3 upper bound (broad)

        # Initial guesses (dt-based):
        self.r2_mu1_0_dt = 3                 # mu1_0 = edge + this*dt
        self.r2_mu2_0_dt = 7                 # mu2_0 = edge + this*dt
        self.r2_mu3_0_dt = 56                # mu3_0 = edge + this*dt

        self.r2_s1_0_dt = 6                  # sigma1_0 = this*dt
        self.r2_s2_0_dt = 10                 # sigma2_0 = this*dt
        self.r2_s3_0_dt = 40                # sigma3_0 = this*dt

        self.r2_A3_frac = 0.3                # A3 initial amplitude fraction vs amp_guess

        # ===== Diagnostics printing =====
        self.print_infeasible_p0_warnings = True

    # --- MODELS ---
    def _gaussian(self, t, A, mu, sigma, C):
        return A * np.exp(-((t - mu) ** 2) / (2.0 * sigma**2)) + C

    def _multi_gaussian_2(self, t, *params):
        A1, mu1, s1, A2, mu2, s2, C = params
        return (
            A1 * np.exp(-((t - mu1) ** 2) / (2.0 * s1**2))
            + A2 * np.exp(-((t - mu2) ** 2) / (2.0 * s2**2))
            + C
        )

    def _multi_gaussian_3(self, t, *params):
        A1, mu1, s1, A2, mu2, s2, A3, mu3, s3, C = params
        return (
            A1 * np.exp(-((t - mu1) ** 2) / (2.0 * s1**2))
            + A2 * np.exp(-((t - mu2) ** 2) / (2.0 * s2**2))
            + A3 * np.exp(-((t - mu3) ** 2) / (2.0 * s3**2))
            + C
        )

    # --- HELPERS ---
    def _dt(self, time):
        time = np.asarray(time, dtype=float)
        if len(time) < 2:
            return np.nan
        d = np.diff(time)
        d = d[np.isfinite(d) & (d > 0)]
        if len(d) == 0:
            return np.nan
        return float(np.median(d))

    def _calculate_r2(self, y_true, y_pred):
        residuals = y_true - y_pred
        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        if ss_tot == 0.0:
            return np.nan
        return 1.0 - (ss_res / ss_tot)

    def _rmse(self, y_true, y_pred):
        e = y_true - y_pred
        return float(np.sqrt(np.mean(e**2)))

    def _repair_to_psd(self, pcov):
        pcov = np.asarray(pcov, dtype=float)
        pcov_sym = 0.5 * (pcov + pcov.T)
        try:
            w, V = np.linalg.eigh(pcov_sym)
            w = np.maximum(w, 0.0)
            return (V * w) @ V.T
        except Exception:
            return pcov_sym

    def _check_p0_in_bounds(self, p0, lower, upper, label=""):
        if not self.print_infeasible_p0_warnings:
            return True

        p0 = np.asarray(p0, dtype=float)
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)

        if p0.shape != lower.shape or p0.shape != upper.shape:
            print(f"[WARN]{label} p0/bounds shape mismatch: p0={p0.shape}, lower={lower.shape}, upper={upper.shape}")
            return False

        bad_lo = p0 < lower
        bad_hi = p0 > upper
        ok = not (bad_lo.any() or bad_hi.any())

        if not ok:
            idxs = np.where(bad_lo | bad_hi)[0]
            print(f"[WARN]{label} Initial guess p0 is OUTSIDE bounds (curve_fit may fail: x0 is infeasible).")
            for k in idxs:
                side = "LOW" if bad_lo[k] else "HIGH"
                print(
                    f"    idx {k:2d}: p0={p0[k]: .6g}  bound=[{lower[k]: .6g}, {upper[k]: .6g}]  ({side})"
                )

        return ok

    def _clip_p0_to_bounds(self, p0, lower, upper, eps=1e-12):
        p0 = np.asarray(p0, dtype=float).copy()
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        p0 = np.minimum(np.maximum(p0, lower), upper)

        # enforce sigma > eps for each pulse sigma index (2,5,8,...), excluding last C
        for s_idx in range(2, len(p0) - 1, 3):
            if p0[s_idx] <= eps:
                p0[s_idx] = max(lower[s_idx], eps)
        return p0

    def _get_sum_peak(self, time, popt, model_func, mu_margin_dt=50, grid_mult=5):
        """Composite peak (above baseline) on a dense grid.

        Uses dt-based mu margin, not t_len-based.
        """
        time = np.asarray(time, dtype=float)
        C = float(popt[-1])
        dt = self._dt(time)
        if not np.isfinite(dt) or dt <= 0:
            # fallback to using time bounds only
            t_lo, t_hi = float(time[0]), float(time[-1])
        else:
            t_lo = float(time[0] - mu_margin_dt * dt)
            t_hi = float(time[-1] + mu_margin_dt * dt)

        n = len(time)
        n_grid = max(int(n * grid_mult), 200)
        t_grid = np.linspace(t_lo, t_hi, n_grid)
        y = model_func(t_grid, *popt)
        return float(np.max(y) - C)

    def _params_within_bounds(self, params, lower, upper, eps=1e-12):
        params = np.asarray(params, dtype=float)
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)

        if params.shape != lower.shape or params.shape != upper.shape:
            return False

        if np.any(params < lower) or np.any(params > upper):
            return False

        # sigma indices (2,5,8,...) must be > eps
        for s_idx in range(2, len(params) - 1, 3):
            if params[s_idx] <= eps:
                return False

        return True

    def _calculate_sum_peak_uncertainty_mc(
        self,
        time,
        popt,
        pcov,
        model_func,
        lower,
        upper,
        n_samples=1000,
        max_draws=20000,
        relax_sigma_lower_factor=0.5,
        fallback_to_clip=True,
    ):
        """Std dev of composite peak via Monte Carlo.

        - Samples theta ~ N(popt, pcov_psd)
        - Rejects draws that violate bounds (optionally relaxed sigma lower bound for MC)
        - Computes composite peak (baseline removed) for each accepted draw
        """
        if pcov is None:
            return np.nan

        pcov = np.asarray(pcov, dtype=float)
        if (not np.isfinite(pcov).all()) or np.isnan(pcov).any():
            return np.nan

        rng = np.random.default_rng()
        pcov_psd = self._repair_to_psd(pcov)

        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)

        # Relax sigma lower bounds for MC only (helps when sigma is pinned to floor)
        lower_mc = lower.copy()
        for s_idx in range(2, len(lower_mc) - 1, 3):
            lower_mc[s_idx] = max(lower_mc[s_idx] * relax_sigma_lower_factor, 1e-12)

        peaks = []
        draws = 0
        while len(peaks) < n_samples and draws < max_draws:
            draws += 1
            sample_params = rng.multivariate_normal(popt, pcov_psd)

            if not self._params_within_bounds(sample_params, lower_mc, upper):
                continue

            peak_val = self._get_sum_peak(
                time,
                sample_params,
                model_func,
                mu_margin_dt=self.mu_global_margin_dt,
                grid_mult=self.sum_peak_grid_mult,
            )
            peaks.append(peak_val)

        min_accept = max(10, n_samples // 10)
        if len(peaks) >= min_accept:
            return float(np.std(peaks, ddof=1))

        if not fallback_to_clip:
            return np.nan

        # Clip fallback (always produces n_samples; slightly biased but robust)
        peaks = []
        for _ in range(n_samples):
            sample_params = rng.multivariate_normal(popt, pcov_psd)
            sample_params = np.clip(sample_params, lower_mc, upper)
            peak_val = self._get_sum_peak(
                time,
                sample_params,
                model_func,
                mu_margin_dt=self.mu_global_margin_dt,
                grid_mult=self.sum_peak_grid_mult,
            )
            peaks.append(peak_val)

        return float(np.std(peaks, ddof=1))

    def _get_empty_result(self):
        return {
            "amplitude": np.nan,
            "uncertainty": np.nan,
            "params": None,
            "pcov": None,
            "success": False,
            "r_squared": np.nan,
            "rmse": np.nan,
            "sigma_hat": np.nan,
        }

    # --- FITTING METHODS ---
    def get_amplitude_region1(self, time, signal, t_edge=None):
        """Fit a single Gaussian in Region 1.

        `t_edge` is used to anchor mu upper bound so mu cannot wander deep into Region 2.
        All mu/sigma bounds and p0 are dt-based.
        """
        time = np.asarray(time, dtype=float)
        signal = np.asarray(signal, dtype=float)
        if len(time) < 5:
            return self._get_empty_result()

        dt = self._dt(time)
        if not np.isfinite(dt) or dt <= 0:
            return self._get_empty_result()

        try:
            # Global mu bounds (dt-based)
            mu_lo_global = float(time[0] - self.mu_global_margin_dt * dt)
            mu_hi_global = float(time[-1] + self.mu_global_margin_dt * dt)

            # Edge-anchored mu bounds for Region 1
            if t_edge is None:
                # fallback: keep mu within global
                mu_lo = mu_lo_global
                mu_hi = mu_hi_global
            else:
                mu_lo = float(time[0] - self.r1_mu_lo_extra_dt * dt)
                mu_hi = float(t_edge + self.r1_mu_hi_after_edge_dt * dt)
                # intersect with global
                mu_lo = max(mu_lo, mu_lo_global)
                mu_hi = min(mu_hi, mu_hi_global)

            sigma_min = float(self.sigma_min_dt * dt)
            sigma_hi = float(self.r1_sigma_hi_dt * dt)

            # Initial guess (dt-based sigma), then clamp into bounds
            A0 = float(np.max(signal) - np.min(signal))
            mu0 = float(time[int(np.argmax(signal))])
            s0 = float(self.r1_sigma0_dt * dt)
            C0 = float(np.min(signal))

            lower = [0.0, mu_lo, sigma_min, -np.inf]
            upper = [np.inf, mu_hi, sigma_hi, np.inf]

            p0 = [A0, mu0, s0, C0]

            self._check_p0_in_bounds(p0, lower, upper, label="[Region1]")
            p0 = self._clip_p0_to_bounds(p0, lower, upper)

            popt, pcov = curve_fit(
                self._gaussian,
                time,
                signal,
                p0=p0,
                bounds=(lower, upper),
                maxfev=4000,
            )

            y_pred = self._gaussian(time, *popt)
            resid = signal - y_pred

            n = len(time)
            p = len(popt)
            dof = max(n - p, 1)
            sigma_hat = float(np.sqrt(np.sum(resid**2) / dof))

            perr = np.sqrt(np.diag(pcov))
            A_hat = float(popt[0])
            A_se = float(perr[0])

            return {
                "amplitude": A_hat,          # above-baseline amplitude
                "uncertainty": A_se,         # 1-sigma SE of A
                "params": popt,
                "pcov": pcov,
                "success": True,
                "model": "single",
                "r_squared": self._calculate_r2(signal, y_pred),
                "rmse": self._rmse(signal, y_pred),
                "sigma_hat": sigma_hat,
            }

        except Exception:
            return self._get_empty_result()

    def get_amplitude_region2(self, time, signal, t_edge, model=None):
        """Fit region 2 with selectable model.

        Bounds and p0 are dt-based and anchored to the edge.
        """
        time = np.asarray(time, dtype=float)
        signal = np.asarray(signal, dtype=float)
        if len(time) < 5:
            return self._get_empty_result()

        if model is None:
            model = self.region2_model

        dt = self._dt(time)
        if not np.isfinite(dt) or dt <= 0:
            return self._get_empty_result()

        edge = float(t_edge)

        try:
            amp_guess = float(np.max(signal) - np.min(signal))
            offset_guess = float(np.min(signal))

            mu_lo_global = float(time[0] - self.mu_global_margin_dt * dt)
            mu_hi_global = float(time[-1] + self.mu_global_margin_dt * dt)

            sigma_min = float(self.sigma_min_dt * dt)

            if model == "single":
                # Reuse Region1 fitter (single gaussian) with edge-aware mu bound
                return self.get_amplitude_region1(time, signal, t_edge=edge)

            elif model == "double":
                func, n_pulses = self._multi_gaussian_2, 2

                # dt-based mu bounds around edge for both components (looser than triple)
                mu1_lo = max(edge - 40 * dt, mu_lo_global)
                mu1_hi = min(edge + 140 * dt, mu_hi_global)
                mu2_lo = max(edge - 10 * dt, mu_lo_global)
                mu2_hi = min(edge + 300 * dt, mu_hi_global)

                s1_hi = 20 * dt
                s2_hi = 60 * dt

                lower = [
                    0.0, mu1_lo, sigma_min,
                    0.0, mu2_lo, sigma_min,
                    -np.inf,
                ]
                upper = [
                    np.inf, mu1_hi, s1_hi,
                    np.inf, mu2_hi, s2_hi,
                    np.inf,
                ]

                p0 = [
                    amp_guess, edge + 10 * dt, min(10 * dt, s1_hi),
                    0.5 * amp_guess, edge + 120 * dt, min(30 * dt, s2_hi),
                    offset_guess,
                ]

            elif model == "triple":
                func, n_pulses = self._multi_gaussian_3, 3

                # --- mu bounds (ALL dt-based) ---
                mu1_lo = edge - self.r2_mu1_span_dt * dt
                mu1_hi = edge + self.r2_mu1_span_dt * dt

                mu2_lo = edge - self.r2_mu2_before_edge_dt * dt
                mu2_hi = edge + self.r2_mu2_after_edge_dt * dt

                mu3_lo = edge + self.r2_mu3_lo_dt * dt
                mu3_hi = edge + self.r2_mu3_hi_dt * dt

                # Intersect with global expanded bounds for partial-capture safety
                mu1_lo, mu1_hi = max(mu1_lo, mu_lo_global), min(mu1_hi, mu_hi_global)
                mu2_lo, mu2_hi = max(mu2_lo, mu_lo_global), min(mu2_hi, mu_hi_global)
                mu3_lo, mu3_hi = max(mu3_lo, mu_lo_global), min(mu3_hi, mu_hi_global)

                # --- sigma bounds (ALL dt-based) ---
                s1_hi = float(self.r2_s1_hi_dt * dt)
                s2_hi = float(self.r2_s2_hi_dt * dt)
                s3_hi = float(self.r2_s3_hi_dt * dt)

                lower = [
                    0.0, mu1_lo, sigma_min,
                    0.0, mu2_lo, sigma_min,
                    0.0, mu3_lo, sigma_min,
                    -np.inf,
                ]
                upper = [
                    np.inf, mu1_hi, s1_hi,
                    np.inf, mu2_hi, s2_hi,
                    np.inf, mu3_hi, s3_hi,
                    np.inf,
                ]

                # --- dt-based initial guess, then clamped into bounds ---
                mu1_0 = edge + self.r2_mu1_0_dt * dt
                mu2_0 = edge + self.r2_mu2_0_dt * dt
                mu3_0 = edge + self.r2_mu3_0_dt * dt

                s1_0 = self.r2_s1_0_dt * dt
                s2_0 = self.r2_s2_0_dt * dt
                s3_0 = self.r2_s3_0_dt * dt

                p0 = [
                    amp_guess, mu1_0, s1_0,
                    amp_guess, mu2_0, s2_0,
                    self.r2_A3_frac * amp_guess, mu3_0, s3_0,
                    offset_guess,
                ]

            else:
                raise ValueError(f"Unknown model: {model}")

            # Shared p0/bounds sanitation
            self._check_p0_in_bounds(p0, lower, upper, label=f"[Region2-{model}]")
            p0 = self._clip_p0_to_bounds(p0, lower, upper)

            popt, pcov = curve_fit(
                func,
                time,
                signal,
                p0=p0,
                bounds=(lower, upper),
                maxfev=8000,
            )

            y_pred = func(time, *popt)
            resid = signal - y_pred

            n = len(time)
            p = len(popt)
            dof = max(n - p, 1)
            sigma_hat = float(np.sqrt(np.sum(resid**2) / dof))

            r2 = self._calculate_r2(signal, y_pred)
            rmse = self._rmse(signal, y_pred)

            if self.multi_pulse_return_mode == "first":
                # earliest pulse by mu
                pulses = []
                for k in range(n_pulses):
                    base = 3 * k
                    pulses.append({
                        "A": float(popt[base]),
                        "mu": float(popt[base + 1]),
                        "sigma": float(popt[base + 2]),
                        "idx": base,
                    })
                pulses.sort(key=lambda d: d["mu"])
                first = pulses[0]

                amp = first["A"]
                perr = np.sqrt(np.diag(pcov))
                err = float(perr[first["idx"]])

            elif self.multi_pulse_return_mode == "sum":
                amp = self._get_sum_peak(
                    time,
                    popt,
                    func,
                    mu_margin_dt=self.mu_global_margin_dt,
                    grid_mult=self.sum_peak_grid_mult,
                )
                err = self._calculate_sum_peak_uncertainty_mc(
                    time,
                    popt,
                    pcov,
                    func,
                    lower=lower,
                    upper=upper,
                )

            else:
                raise ValueError(f"Unknown return_mode: {self.multi_pulse_return_mode}")

            return {
                "amplitude": float(amp),
                "uncertainty": float(err) if np.isfinite(err) else np.nan,
                "params": popt,
                "pcov": pcov,
                "success": True,
                "model": model,
                "r_squared": r2,
                "rmse": rmse,
                "sigma_hat": sigma_hat,
            }

        except Exception:
            return self._get_empty_result()

    # --- PIPELINE ---
    def run(self, time, signal, t_edge, region2_model=None, fit_window=None):
        if region2_model is None:
            region2_model = self.region2_model
        if fit_window is None:
            fit_window = self.fit_window

        time = np.asarray(time, dtype=float)
        signal = np.asarray(signal, dtype=float)

        split_idx = np.searchsorted(time, t_edge)
        t1_full, s1_full = time[:split_idx], signal[:split_idx]
        t2_full, s2_full = time[split_idx:], signal[split_idx:]

        # Apply windowing
        if fit_window is not None:
            mask1 = t1_full >= (t_edge - fit_window)
            t1_fit, s1_fit = t1_full[mask1], s1_full[mask1]

            mask2 = t2_full <= (t_edge + fit_window)
            t2_fit, s2_fit = t2_full[mask2], s2_full[mask2]
        else:
            t1_fit, s1_fit = t1_full, s1_full
            t2_fit, s2_fit = t2_full, s2_full

        r1_res = self.get_amplitude_region1(t1_fit, s1_fit, t_edge=t_edge)
        r2_res = self.get_amplitude_region2(t2_fit, s2_fit, t_edge=t_edge, model=region2_model)

        ratio, ratio_err = np.nan, np.nan
        if r1_res.get("success") and r2_res.get("success"):
            A1 = float(r1_res["amplitude"])
            A2 = float(r2_res["amplitude"])
            if A2 != 0.0 and np.isfinite(A1) and np.isfinite(A2):
                ratio = A1 / A2

                u1 = float(r1_res.get("uncertainty", np.nan))
                u2 = float(r2_res.get("uncertainty", np.nan))

                if np.isfinite(u1) and np.isfinite(u2) and A1 != 0.0:
                    ratio_err = abs(ratio) * np.sqrt((u1 / A1) ** 2 + (u2 / A2) ** 2)

        return {
            "ratio": float(ratio) if np.isfinite(ratio) else np.nan,
            "ratio_uncertainty": float(ratio_err) if np.isfinite(ratio_err) else np.nan,
            "t_edge": float(t_edge),
            "fit_window": fit_window,
            "region1": r1_res,
            "region2": r2_res,
            "time": time,
            "signal": signal,
        }

    # --- VISUALIZATION ---
    def plot_result(self, res, ax=None, show_region2_components=True, mark_component_peaks=True):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        time, signal, t_edge = res["time"], res["signal"], res["t_edge"]
        fit_window = res.get("fit_window")

        ax.scatter(time, signal, color="lightgrey", s=10, label="Raw Signal")
        ax.axvline(t_edge, color="black", linestyle=":", label="t_edge")

        if fit_window is not None:
            ax.axvspan(time[0], t_edge - fit_window, color="red", alpha=0.05, label="Excluded")
            ax.axvspan(t_edge + fit_window, time[-1], color="red", alpha=0.05)

        t_dense = np.linspace(time[0], time[-1], 1000)

        for region_key, color, label in [("region1", "red", "Pulse 1"), ("region2", "blue", "Pulse 2")]:
            r_res = res.get(region_key, {})
            if not r_res.get("success", False):
                continue

            model = r_res.get("model", "")
            params = r_res.get("params", None)
            if params is None:
                continue

            if model == "single":
                func = self._gaussian
            elif model == "double":
                func = self._multi_gaussian_2
            elif model == "triple":
                func = self._multi_gaussian_3
            else:
                continue

            y_fit = func(t_dense, *params)
            active = (t_dense <= t_edge) if region_key == "region1" else (t_dense >= t_edge)

            r2 = r_res.get("r_squared", np.nan)
            r2_label = f"{label} ($R^2={r2:.3f}$)" if np.isfinite(r2) else label

            ax.plot(t_dense[active], y_fit[active], color=color, linewidth=2, label=r2_label)
            ax.plot(t_dense[~active], y_fit[~active], color=color, alpha=0.3, linestyle="--")

            peak_idx = int(np.argmax(y_fit))
            ax.plot(
                t_dense[peak_idx],
                y_fit[peak_idx],
                marker="*",
                color=color,
                markersize=12,
                markeredgecolor="k",
            )

            # Plot individual Gaussians in Region 2
            if show_region2_components and region_key == "region2" and model in ("double", "triple"):
                C = float(params[-1])
                n_pulses = 2 if model == "double" else 3
                for k in range(n_pulses):
                    base = 3 * k
                    A_k = float(params[base])
                    mu_k = float(params[base + 1])
                    s_k = float(params[base + 2])

                    y_comp = self._gaussian(t_dense, A_k, mu_k, s_k, C)
                    ax.plot(
                        t_dense[active],
                        y_comp[active],
                        linewidth=1.5,
                        linestyle="--",
                        alpha=0.9,
                        label=f"R2 comp {k+1}",
                    )

                    if mark_component_peaks:
                        idx = int(np.argmin(np.abs(t_dense - mu_k)))
                        ax.plot(
                            t_dense[idx],
                            y_comp[idx],
                            marker="o",
                            markersize=6,
                            markeredgecolor="k",
                            alpha=0.9,
                        )

        ratio = res.get("ratio", np.nan)
        ax.set_title(f"Fit Result | Ratio: {ratio:.4f}" if np.isfinite(ratio) else "Fit Result | Ratio: NaN")
        ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        return ax
    def plot_residuals(self, res, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))

        time, signal, t_edge = res['time'], res['signal'], res['t_edge']
        fit_window = res.get('fit_window')

        # Overlay Raw Data (Context)
        # We use a twin axis to show the raw signal shape behind the residuals
        # without messing up the residual scale (which is usually much smaller).
        #ax_raw = ax.twinx()
        ax.plot(time, signal, color='grey', alpha=0.8, linewidth=2, label='Raw Signal')
        #ax_raw.set_ylabel("Raw Signal (Grey)", color='grey')
        #ax_raw.tick_params(axis='y', labelcolor='grey')
        
        # Move the residual axis (ax) to the front so the dots are visible on top of the line
        #ax.set_zorder(ax.get_zorder() + 1)
        ax.patch.set_visible(False)

        # Split data at t_edge
        split_idx = np.searchsorted(time, t_edge)
        t1_full, s1_full = time[:split_idx], signal[:split_idx]
        t2_full, s2_full = time[split_idx:], signal[split_idx:]

        # Apply windowing if recorded in results
        if fit_window is not None:
            mask1 = t1_full >= (t_edge - fit_window)
            t1_fit, s1_fit = t1_full[mask1], s1_full[mask1]

            mask2 = t2_full <= (t_edge + fit_window)
            t2_fit, s2_fit = t2_full[mask2], s2_full[mask2]
        else:
            t1_fit, s1_fit = t1_full, s1_full
            t2_fit, s2_fit = t2_full, s2_full

        # Region 1 Residuals
        r1_res = res['region1']
        if r1_res.get('success', False):
            # Region 1 uses _gaussian
            y_model_1 = self._gaussian(t1_fit, *r1_res['params'])
            resid_1 = s1_fit - y_model_1
            ax.plot(t1_fit, resid_1, 'r.', label='Region 1 Residuals', markersize=3)

        # Region 2 Residuals
        r2_res = res['region2']
        if r2_res.get('success', False):
            model = r2_res.get('model', 'single')
            if model == 'triple': func = self._multi_gaussian_3
            elif model == 'double': func = self._multi_gaussian_2
            else: func = self._gaussian
            
            y_model_2 = func(t2_fit, *r2_res['params'])
            resid_2 = s2_fit - y_model_2
            ax.plot(t2_fit, resid_2, 'b.', label='Region 2 Residuals', markersize=3)

        ax.set_title("Raw Residuals (Data - Model)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Residual Amplitude")
        ax.axhline(0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(t_edge, color='k', linestyle=':', alpha=0.5, label='t_edge')
        
        # Combine legends from both axes
        lines_1, labels_1 = ax.get_legend_handles_labels()
        #lines_2, labels_2 = ax_raw.get_legend_handles_labels()
        ax.legend(loc='upper right')
        
        ax.grid(True, alpha=0.3)
        return ax