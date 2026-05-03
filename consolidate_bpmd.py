import os
import glob
import tarfile
import pandas as pd
import numpy as np
import re

def sum_gaussians(kernels, grid_points, width):
    """Reconstruct FES from Gaussian kernels."""
    fes = np.zeros_like(grid_points)
    sigma2 = 2 * (width**2)
    
    # Vectorized summation for performance
    # columns: 0:time, 1:height, 2:center, 3:width
    heights = kernels[1].values
    centers = kernels[2].values
    
    for h, c in zip(heights, centers):
        diff = grid_points - c
        fes += h * np.exp(-(diff**2) / sigma2)
        
    return -fes # Negative sum for Free Energy

def process_pose(pose_folder, output_csv):
    print(f"\n{'='*60}\n[*] Processing: {os.path.basename(pose_folder)}\n{'='*60}")
    
    # Use a fixed grid and round it to avoid floating point merge issues
    grid_points = np.linspace(0, 10, 201) 
    grid_points = np.round(grid_points, 3)
    
    master_df = pd.DataFrame({'RMSD': grid_points})
    
    # Dynamically find all trial directories (e.g., trial_01, trial_02, ...)
    trial_folders = sorted(glob.glob(os.path.join(pose_folder, "trial_*")))
    
    if not trial_folders:
        print(f"  [!] No trial folders found in {pose_folder}")
        return

    for trial_path in trial_folders:
        trial_dirname = os.path.basename(trial_path)
        # Human readable trial name for the column header
        trial_label = trial_dirname.replace("_", " ").title()
        
        tgz_files = glob.glob(os.path.join(trial_path, "*-out.tgz"))
        if not tgz_files:
            print(f"  [!] No -out.tgz found in {trial_dirname}")
            continue
            
        tgz_path = tgz_files[0]
        extract_dir = os.path.join(trial_path, "extracted_data")
        os.makedirs(extract_dir, exist_ok=True)
        
        print(f"  [*] {trial_label}: Extracting {os.path.basename(tgz_path)}...")
        try:
            with tarfile.open(tgz_path, "r:gz") as tar:
                tar.extractall(path=extract_dir)
        except Exception as e:
            print(f"    [!] Extraction failed: {e}")
            continue
            
        # Reconstruct FES from kernels
        kernel_file = None
        for root, dirs, files in os.walk(extract_dir):
            if "metadynamics_outfile.dat" in files:
                kernel_file = os.path.join(root, "metadynamics_outfile.dat")
                break
        
        if kernel_file:
            try:
                kernels = pd.read_csv(kernel_file, sep=r'\s+', comment='#', header=None, engine='python')
                if kernels.empty:
                    print(f"    [!] Kernel file is empty.")
                    continue
                
                # Default Desmond columns: 0:time, 1:height, 2:center, 3:width
                avg_width = kernels[3].mean() if kernels[3].mean() > 0 else 0.02
                fes_vals = sum_gaussians(kernels, grid_points, avg_width)
                
                trial_df = pd.DataFrame({
                    'RMSD': grid_points,
                    trial_label: fes_vals
                })
                
                master_df = pd.merge(master_df, trial_df, on='RMSD', how='left')
                print(f"    [✓] Reconstructed FES.")
            except Exception as e:
                print(f"    [!] Error: {e}")
        else:
            print(f"    [!] No kernel file found.")
            
    # Save the final file
    master_df.to_csv(output_csv, index=False)
    print(f"\n[SUCCESS] Saved: {output_csv}")

if __name__ == "__main__":
    data_root = "data"
    if os.path.exists(data_root):
        # Recursively find all pose directories (e.g., in data/RUN_NAME/pose_01)
        pose_folders = sorted(glob.glob(os.path.join(data_root, "**/pose_*"), recursive=True))
        
        if not pose_folders:
            print(f"[!] No pose directories found in '{data_root}'")
        else:
            for pose_folder in pose_folders:
                # If nested, try to include the parent name to avoid collisions
                parent_name = os.path.basename(os.path.dirname(pose_folder))
                pose_basename = os.path.basename(pose_folder)
                
                if parent_name != "data":
                    # For nested data, name it 'runname_pose_01'
                    out_name = f"{pose_basename}_all_trials.csv"
                else:
                    out_name = f"{pose_basename}_all_trials.csv"
                
                process_pose(pose_folder, out_name)
    else:
        print(f"[!] Error: Root data directory '{data_root}' not found.")
