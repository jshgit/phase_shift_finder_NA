import numpy as np
import os

class DataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def get_file_list(self):
        return sorted([f for f in os.listdir(self.data_dir) if f.endswith('.npz')])

    def load_run(self, filename):
        """Loads a single run and returns structured data for both conditions."""
        filepath = os.path.join(self.data_dir, filename)
        data = np.load(filepath)
        
        # Structure the data for easier access
        time = data['time']
        
        # Pack Condition A
        cond_a = {
            'signals': data['signals_a'],       # Shape (50, 488)
            'scan_param': data['scan_param_a'], # Shape (50,)
            't_edge': float(data['t_edge_a'])   # Scalar
        }
        
        # Pack Condition B
        cond_b = {
            'signals': data['signals_b'],
            'scan_param': data['scan_param_b'],
            't_edge': float(data['t_edge_b'])
        }
        
        return time, cond_a, cond_b

    @staticmethod
    def split_trace(time, signal, t_edge):
        """Splits a single trace into Region 1 and Region 2 based on t_edge."""
        # Find index closest to t_edge
        split_idx = np.searchsorted(time, t_edge)
        
        # Slice
        t1, s1 = time[:split_idx], signal[:split_idx]
        t2, s2 = time[split_idx:], signal[split_idx:]
        
        return (t1, s1), (t2, s2)