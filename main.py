import os
import yaml
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

    # 1. Load Configuration
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file not found: {args.config}")
    
    config = load_config(args.config)
    
    # Extract Paths
    data_dir = config['experiment']['data_dir']
    results_dir = config['output']['results_dir']
    summary_filename = config['output'].get('summary_file')
    figures_folder_name = config['output'].get('figures_folder')
    
    # Create Results Directory if it doesn't exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Determine if we are plotting
    should_plot = figures_folder_name is not None
    figs_dir = None
    if should_plot:
        figs_dir = os.path.join(results_dir, figures_folder_name)
        os.makedirs(figs_dir, exist_ok=True)

    # Initialize Modules
    loader = DataLoader(data_dir)
    processor = SignalProcessor()
    analyzer = FringeAnalyzer()

    results = []
    
    try:
        files = loader.get_file_list()
    except FileNotFoundError:
        print(f"Error: Data directory '{data_dir}' not found.")
        return

    print(f"Found {len(files)} files in '{data_dir}'. Processing...")

    for filename in files:
        print(f"Processing {filename}...")
        try:
            time, cond_a, cond_b = loader.load_run(filename)
        except Exception as e:
            print(f"Failed to load {filename}: {e}")
            continue
        
        for condition_name, cond_data in [('A', cond_a), ('B', cond_b)]:
            scan_params = cond_data['scan_param']
            signals = cond_data['signals']
            t_edge = cond_data['t_edge']
            
            ratios = []
            
            # --- Signal Extraction ---
            for i in range(len(signals)):
                trace = signals[i]
                (t1, s1), (t2, s2) = loader.split_trace(time, trace, t_edge)
                
                amp1 = processor.get_amplitude_region1(t1, s1)
                amp2 = processor.get_amplitude_region2(t2, s2)
                
                if amp2 > 0:
                    ratios.append(amp1 / amp2)
                else:
                    ratios.append(np.nan)
            
            ratios = np.array(ratios)
            
            # --- Physics Analysis ---
            phase, unc = analyzer.estimate_phase(scan_params, ratios)
            
            results.append({
                'Filename': filename,
                'Condition': condition_name,
                'Phase_mrad': phase,
                'Uncertainty_mrad': unc
            })

            # --- Optional Plotting ---
            if should_plot and not np.isnan(phase):
                plt.figure(figsize=(10, 4))
                
                # Plot Raw Data
                plt.scatter(scan_params, ratios, label='Data', alpha=0.7)
                
                # Reconstruct Sine for Visualization
                # Note: We do a quick local fit just for plotting logic or use parameters if returned
                # For simplicity here, we just visualize the data vs phase result text
                
                plt.title(f"{filename} - Condition {condition_name} (Phase: {phase:.2f} mrad)")
                plt.xlabel("Scan Parameter (Deg)")
                plt.ylabel("Ratio")
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                save_path = os.path.join(figs_dir, f"{filename}_{condition_name}.png")
                plt.savefig(save_path)
                plt.close()

    # --- Save Summary CSV ---
    if summary_filename:
        output_path = os.path.join(results_dir, summary_filename)
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        print(f"\nProcessing complete. Results saved to {output_path}")
    else:
        print("\nProcessing complete. CSV output skipped (filename was None).")

if __name__ == "__main__":
    main()