
# Quantum Sensor Phase Analysis Pipeline

This repository contains a Python-based pipeline for analyzing noisy quantum sensor data. It processes raw signal traces, extracts ratios between signal regions using multi-Gaussian fitting, and estimates phase shifts using a fringe analysis engine with robust outlier rejection.

## 📂 Project Structure

The project is designed to be modular, separating data loading, signal processing, and physics analysis.

```text
.
├── main.py                  # Entry point for the analysis pipeline
├── config.yaml              # Central configuration file
├── data/                    # Directory for raw .npz input files
├── results/                 # Output directory for CSVs and plots
└── src/
    ├── data_loader.py       # Handles .npz file loading and batching
    ├── signal_processor.py  # Gaussian fitting and ratio extraction
    └── fringe_analyzer.py   # Sine fitting with outlier rejection logic
```

## 🚀 Getting Started

### 1. Installation

Ensure you have a Python environment (e.g., Anaconda or venv) set up. Install the required dependencies:

```bash
pip install numpy pandas matplotlib scipy pyyaml
```

### 2. Data Setup

Place your raw data files (`.npz` format) inside the `data/` directory.

* The pipeline expects files to contain `signals_a`, `signals_b`, `scan_param_a`, `scan_param_b`, and `t_edge` arrays.
* If the `data/` folder does not exist, create it in the root directory.

### 3. Running the Analysis

To run the full pipeline, simply execute `main.py` from the root directory:

```bash
python main.py
```

By default, this will process **all** files in the `data/` folder and save results to `results/`.

---

## ⚙️ Configuration Guide (`config.yaml`)

All high-level parameters, file selection, and processing switches are controlled via `config.yaml`. You do not need to modify the Python code to change these settings.

### 1. File Selection (Single vs. Batch Mode)

You can choose to process the entire dataset or focus on a single file for debugging.

```yaml
experiment:
  data_dir: "data"

  # Set to null to process ALL files in the directory (Batch Mode)
  # Set to "run_001.npz" to process ONLY that specific file
  input_filename: null
```

### 2. Toggling Outputs (On/Off Switches)

You can turn off plotting or CSV saving by setting their values to `null`.

```yaml
output:
  results_dir: "results"

  # filename.csv -> Saves summary of all runs to CSV
  # null         -> Disables CSV saving
  summary_file: "phase_estimates.csv"

  # "figures"    -> Saves plots to results/figures/
  # null         -> Disables plotting (faster)
  figures_folder: "figures"
```

### 3. Tuning the Signal Processor

Control how the raw traces are fit. This is useful if your Region 2 pulse shape changes or if you need to restrict the fitting window.

```yaml
signal_processor:
  # Model for Region 2 pulse complex: "single", "double", or "triple"
  region2_model: "triple"

  # How to handle multi-pulse results: "first" (earliest peak) or "sum" (total area)
  multi_pulse_return_mode: "sum"

  # Window size around t_edge (in indices) to fit.
  # Set to null to use the full trace.
  fit_window: null
```

### 4. Tuning the Fringe Analyzer

Adjust the strictness of the physics engine's outlier rejection.

```yaml
fringe_analyzer:
  # Minimum R-squared required for a trace fit to be included.
  # Traces below this (e.g., 0.8) are rejected immediately.
  r2_threshold: 0.8

  # Number of "noisiest" points (highest uncertainty) to drop
  # automatically from the final sine fit.
  uncertainty_reject_count: 2

  # Z-score threshold for rejecting ratio outliers (points that deviate
  # significantly from the preliminary sine wave).
  residual_z_score: 2.5
```

---

## 📊 Output Interpretation

### Console / Logs

The script logs progress to the console, including which files are being processed and if any critical failures occur.

### Visualizations (`results/figures/`)

If enabled, the pipeline generates one PNG per file/condition.

* **Blue dots:** Valid data points used for the final phase estimate.
* **Vertical dashed lines:** Rejected data points (colored by rejection reason):

  * **Red:** Rejected due to poor trace fit (< threshold).
  * **Orange:** Rejected as a statistical outlier.
  * **Gold:** Rejected due to high uncertainty.
* **Green line:** The final fitted sine wave model.
* **Title:** Displays the estimated phase shift ± uncertainty in milliradians (mrad).

### Summary CSV (`results/phase_estimates.csv`)

A table containing the final results for every processed file:

* `Filename`: Name of the source `.npz` file.
* `Condition`: A or B.
* `Phase_mrad`: Estimated phase shift.
* `Uncertainty_mrad`: Propagated uncertainty of the estimate.
* `N_Used`: Number of valid data points used in the final fit.

```
```
