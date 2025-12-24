import os
import yaml
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging

from src.data_loader import DataLoader
from src.signal_processor import SignalProcessor
from src.fringe_analyzer import FringeAnalyzer

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="CMLE Challenge Processor")
    parser.add_argument('--config', type=str, default='config.yaml', help="Path to configuration file")
    args = parser.parse_args()

    # 1. Setup Logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    # 2. Load Configuration
    if not os.path.exists(args.config):
        logging.error(f"Configuration file '{args.config}' not found.")
        return
    
    config = load_config(args.config)
    
    # Extract Paths
    data_dir = config['experiment']['data_dir']
    specific_file = config['experiment'].get('input_filename')
    results_dir = config['output']['results_dir']
    summary_filename = config['output'].get('summary_file')
    figures_folder_name = config['output'].get('figures_folder')
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    should_plot = figures_folder_name is not None
    figs_dir = None
    if should_plot:
        figs_dir = os.path.join(results_dir, figures_folder_name)
        os.makedirs(figs_dir, exist_ok=True)

    # 3. Initialize Modules
    loader = DataLoader(data_dir)
    
    # Init SignalProcessor
    sp_config = config.get('signal_processor', {})
    logging.info(f"SignalProcessor Config: {sp_config}")
    processor = SignalProcessor(**sp_config)
    
    # Init FringeAnalyzer
    analyzer = FringeAnalyzer()
    fa_config = config.get('fringe_analyzer', {})
    logging.info(f"FringeAnalyzer Config: {fa_config}")

    # 4. Determine File List
    try:
        all_files = loader.get_file_list()
    except FileNotFoundError:
        logging.error(f"Data directory '{data_dir}' not found.")
        return

    if not all_files:
        logging.warning(f"No .npz files found in {data_dir}")
        return

    if specific_file:
        if specific_file in all_files:
            files_to_process = [specific_file]
        else:
            logging.error(f"File '{specific_file}' not found.")
            return
    else:
        files_to_process = all_files
        logging.info(f"Processing {len(files_to_process)} files.")

    results = []

    # 5. Main Loop
    for filename in files_to_process:
        logging.info(f"Processing {filename}...")
        try:
            time, cond_a, cond_b = loader.load_run(filename)
        except Exception as e:
            logging.error(f"Failed to load {filename}: {e}")
            continue
        
        for condition_name, cond_data in [('A', cond_a), ('B', cond_b)]:
            scan_params = cond_data['scan_param']
            signals = cond_data['signals']
            t_edge = cond_data['t_edge']
            
            # Arrays to hold trace-by-trace stats
            ratios = []
            uncertainties = []
            r1_r2_scores = [] 
            r2_r2_scores = [] 
            
            # --- TRACE FITTING ---
            for i in range(len(signals)):
                trace = signals[i]
                
                # Fit the individual trace
                res = processor.run(time, trace, t_edge)
                
                # Extract Ratio & Uncertainty
                if np.isfinite(res['ratio']):
                    ratios.append(res['ratio'])
                    uncertainties.append(res['ratio_uncertainty'])
                else:
                    ratios.append(np.nan)
                    uncertainties.append(np.nan)
                
                # Extract R-Squared values for the FringeAnalyzer's filter
                r1_r2_scores.append(res['region1'].get('r_squared', 0.0))
                r2_r2_scores.append(res['region2'].get('r_squared', 0.0))
            
            # Convert to numpy
            ratios = np.array(ratios)
            uncertainties = np.array(uncertainties)
            r1_r2_scores = np.array(r1_r2_scores)
            r2_r2_scores = np.array(r2_r2_scores)
            
            # --- FRINGE ANALYSIS ---
            fringe_res = analyzer.fit(
                scan_params, 
                ratios, 
                uncertainties, 
                r1_r2_scores, 
                r2_r2_scores,
                **fa_config 
            )
            
            # Handle results
            if fringe_res:
                phase_mrad = fringe_res['phase_rad'] * 1000 
                phase_unc_mrad = fringe_res['phase_err'] * 1000
                
                results.append({
                    'Filename': filename,
                    'Condition': condition_name,
                    'Phase_mrad': phase_mrad,
                    'Uncertainty_mrad': phase_unc_mrad,
                    'N_Used': fringe_res['n_used']
                })
                
                # --- PLOTTING (Updated) ---
                if should_plot:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    
                    # 1. Valid Data Points
                    valid = fringe_res['valid_mask']
                    ax.errorbar(
                        scan_params[valid], 
                        ratios[valid], 
                        yerr=uncertainties[valid], 
                        fmt='o', 
                        color='blue', 
                        label='Valid Data', 
                        alpha=0.8, 
                        capsize=2
                    )
                    
                    # 2. Rejected Data (Vertical Lines Only)
                    # We iterate through unique x-values of rejected points to avoid drawing 
                    # multiple lines if multiple repeats at the same scan param are rejected.
                    
                    # R^2 Failures (Red)
                    rej_r2_x = np.unique(scan_params[fringe_res['mask_rej_r2']])
                    for x_val in rej_r2_x:
                        ax.axvline(x_val, color='red', linestyle='--', alpha=0.5, 
                                   label='Poor Trace Fit' if x_val == rej_r2_x[0] else "")

                    # Outliers (Orange)
                    rej_ratio_x = np.unique(scan_params[fringe_res['mask_rej_ratio']])
                    for x_val in rej_ratio_x:
                        ax.axvline(x_val, color='orange', linestyle='--', alpha=0.5, 
                                   label='Outlier' if x_val == rej_ratio_x[0] else "")

                    # High Uncertainty (Gold)
                    rej_unc_x = np.unique(scan_params[fringe_res['mask_rej_unc']])
                    for x_val in rej_unc_x:
                        ax.axvline(x_val, color='gold', linestyle='--', alpha=0.6, 
                                   label='High Uncertainty' if x_val == rej_unc_x[0] else "")

                    # 3. Fit Model
                    x_fit = np.linspace(scan_params.min(), scan_params.max(), 100)
                    y_fit = analyzer._sine_model(x_fit, *fringe_res['params'])
                    ax.plot(x_fit, y_fit, 'g-', linewidth=2, label='Fit Model')

                    # 4. Formatting
                    ax.set_title(
                        f"{filename} [Cond {condition_name}]\n"
                        f"Phase: {phase_mrad:.2f} $\pm$ {phase_unc_mrad:.2f} mrad"
                    )
                    ax.set_xlabel("Scan Parameter")
                    ax.set_ylabel("Ratio")
                    
                    # Deduplicate legend entries (in case loops added multiple labels)
                    handles, labels = ax.get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize='small')
                    
                    ax.grid(True, alpha=0.3)
                    
                    save_path = os.path.join(figs_dir, f"{filename}_{condition_name}.png")
                    plt.savefig(save_path)
                    plt.close()
            else:
                logging.warning(f"Fringe fit failed for {filename} Condition {condition_name}")

    # 6. Save Summary
    if summary_filename and results:
        output_path = os.path.join(results_dir, summary_filename)
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        logging.info(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()