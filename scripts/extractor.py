import os
import glob
import tarfile
import pandas as pd
import numpy as np
import shutil
import traceback

def sum_gaussians(kernels, grid_points, width):
    """Reconstruct FES from Gaussian kernels (Hills)."""
    fes = np.zeros_like(grid_points)
    sigma2 = 2 * (width**2)
    
    # columns: 0:time, 1:height, 2:center, 3:width
    heights = kernels[1].values
    centers = kernels[2].values
    
    for h, c in zip(heights, centers):
        diff = grid_points - c
        fes += h * np.exp(-(diff**2) / sigma2)
        
    return -fes # Negative sum for Free Energy

import re

def flatten_cvseq(input_path, output_path):
    """Parses nested bracket format of .cvseq and saves as clean space-separated table."""
    # Pattern to match: [[time [val]] [cv_00 [val]] [potential [val]]]
    # Using a more flexible regex to capture scientific notation correctly
    pattern = re.compile(r'time\s+\[([\d\.eE+-]+)\]\]\s+\[cv_00\s+\[([\d\.eE+-]+)\]\]\s+\[potential\s+\[([\d\.eE+-]+)\]\]')
    
    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            match = pattern.search(line)
            if match:
                try:
                    time_val = float(match.group(1))
                    cv_val   = float(match.group(2))
                    pot_val  = float(match.group(3))
                    
                    # Formatting: Time as int/float, CV/Pot with high precision fixed-point
                    # This avoids 'mangling' scientific notation in tools that don't support it well.
                    # 10 decimal places is usually enough for RMSD and Potential.
                    f_out.write(f"{time_val:g}  {cv_val:.10f}  {pot_val:.10f}\n")
                except ValueError:
                    continue

def extract_and_process():
    data_root = "data"
    output_base = os.path.join(data_root, "extracted_analysis_files")
    
    print(f"\n{'='*60}")
    print(f"[*] Starting Raw Data Extraction and FES Reconstruction")
    print(f"{'='*60}")

    # Find all pose folders
    pose_folders = sorted(glob.glob(os.path.join(data_root, "**/pose_*"), recursive=True))
    
    for pose_folder in pose_folders:
        pose_id = os.path.basename(pose_folder)
        trial_folders = sorted(glob.glob(os.path.join(pose_folder, "trial_*")))
        
        for trial_path in trial_folders:
            trial_id = os.path.basename(trial_path)
            dest_dir = os.path.join(output_base, pose_id, trial_id)
            os.makedirs(dest_dir, exist_ok=True)
            
            tgz_files = glob.glob(os.path.join(trial_path, "*-out.tgz"))
            if not tgz_files:
                continue
                
            tgz_path = tgz_files[0]
            temp_extract = os.path.join(trial_path, "temp_extract")
            os.makedirs(temp_extract, exist_ok=True)
            
            print(f"  [+] Processing {pose_id} | {trial_id}...")
            
            try:
                # 1. Extract specifically needed files to temp
                with tarfile.open(tgz_path, "r:gz") as tar:
                    members = [m for m in tar.getmembers() if "metadynamics_outfile.dat" in m.name or m.name.endswith(".cvseq")]
                    tar.extractall(path=temp_extract, members=members)
                
                # 2. Locate files in temp
                kernel_file = None
                cv_file = None
                
                for root, dirs, files in os.walk(temp_extract):
                    for f in files:
                        if f == "metadynamics_outfile.dat":
                            kernel_file = os.path.join(root, f)
                        elif f.endswith(".cvseq"):
                            cv_file = os.path.join(root, f)
                
                # 3. Process Hills -> .fes
                if kernel_file:
                    kernels = pd.read_csv(kernel_file, sep=r'\s+', comment='#', header=None, engine='python')
                    if not kernels.empty:
                        grid_points = np.linspace(0, 10, 201)
                        avg_width = kernels[3].mean() if kernels[3].mean() > 0 else 0.02
                        fes_vals = sum_gaussians(kernels, grid_points, avg_width)
                        
                        fes_df = pd.DataFrame({'RMSD': grid_points, 'FreeEnergy': fes_vals})
                        fes_out = os.path.join(dest_dir, f"{pose_id}_{trial_id}.fes")
                        fes_df.to_csv(fes_out, index=False, sep='\t')
                        print(f"    ✓ Created .fes")
                
                # 4. Flatten CVSEQ -> .covlar
                if cv_file:
                    cv_out = os.path.join(dest_dir, f"{pose_id}_{trial_id}.covlar")
                    flatten_cvseq(cv_file, cv_out)
                    print(f"    ✓ Created flattened .covlar")
                
            except Exception as e:
                print(f"    [!] Error processing {tgz_path}: {e}")
                traceback.print_exc()
            finally:
                # Cleanup temp
                shutil.rmtree(temp_extract)

    print(f"\n[SUCCESS] All files extracted to: {output_base}")

if __name__ == "__main__":
    extract_and_process()
