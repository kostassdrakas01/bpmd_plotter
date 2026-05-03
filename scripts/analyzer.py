import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

def analyze_and_plot(csv_files, output_plot_old, summary_csv, report_txt):
    """
    Analyzes consolidated BPMD data and generates comparative plots, individual pose plots, and a detailed report.
    """
    sns.set_theme(style="whitegrid")
    
    summary_data = []
    report_lines = ["BPMD STABILITY ANALYSIS REPORT\n", "="*35 + "\n"]
    stability_ranking = []
    all_pose_data = {} # Store data for Top 3 plot
    
    # FIG 1: Comparison with STD bands
    plt.figure(1, figsize=(10, 7), dpi=300)
    # FIG 2: Comparison with Global Average
    plt.figure(2, figsize=(10, 7), dpi=300)
    
    colors = sns.color_palette("husl", len(csv_files))

    for idx, csv_file in enumerate(csv_files):
        pose_id = os.path.basename(csv_file).replace("_all_trials.csv", "")
        pose_name = pose_id.replace("_", " ").title()
        
        df = pd.read_csv(csv_file)
        if df.shape[1] < 2:
            print(f"  [!] {pose_name}: No trial data found. Skipping statistics.")
            continue
            
        rmsd = df['RMSD'].values
        trial_cols = [c for c in df.columns if c != 'RMSD']
        
        mean_raw = df[trial_cols].mean(axis=1).values
        baseline = mean_raw[-1]
        
        df_rel = df.copy()
        for col in trial_cols:
            df_rel[col] = df[col] - baseline
            
        mean_fes = df_rel[trial_cols].mean(axis=1).values
        std_fes = df_rel[trial_cols].std(axis=1).fillna(0).values
        
        color = colors[idx]

        # ---------------------------------------------------------
        # 1. INDIVIDUAL POSE PLOT
        # ---------------------------------------------------------
        plt.figure(figsize=(10, 6), dpi=300)
        for col in trial_cols:
            plt.plot(rmsd, df_rel[col], color='gray', alpha=0.15, linewidth=0.4, label='_nolegend_')
        plt.plot(rmsd, mean_fes, color=color, linewidth=1.2, label=f'Ensemble Mean')
        if len(trial_cols) > 1:
            plt.fill_between(rmsd, mean_fes - std_fes, mean_fes + std_fes, color=color, alpha=0.2, label='Standard Deviation')
        
        plt.title(f"Free Energy Surface - {pose_name}", fontsize=14, fontweight='bold')
        plt.xlabel("RMSD (Å)", fontsize=12)
        plt.ylabel("Relative Free Energy (kcal/mol)", fontsize=12)
        plt.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        plt.legend(frameon=True)
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.savefig(os.path.join("plots", f"{pose_id}_ensemble.png"), bbox_inches='tight')
        plt.close()
        
        # ---------------------------------------------------------
        # 2. COMPARISON FIG 1 (WITH STD)
        # ---------------------------------------------------------
        plt.figure(1)
        plt.plot(rmsd, mean_fes, label=f"{pose_name}", color=color, linewidth=1.0)
        if len(trial_cols) > 1:
            plt.fill_between(rmsd, mean_fes - std_fes, mean_fes + std_fes, color=color, alpha=0.1)

        # ---------------------------------------------------------
        # 3. COMPARISON FIG 2 (WITHOUT STD)
        # ---------------------------------------------------------
        plt.figure(2)
        plt.plot(rmsd, mean_fes, label=f"{pose_name}", color=color, linewidth=1.0)

        # METRICS
        min_idx = np.argmin(mean_fes)
        well_depth = abs(mean_fes[min_idx])
        barrier_height = well_depth
        avg_std = np.mean(std_fes)
        final_std = std_fes[-1]
        
        threshold = -0.05 * barrier_height
        unbinding_idx_candidates = np.where(mean_fes >= threshold)[0]
        after_well_idx = unbinding_idx_candidates[unbinding_idx_candidates > min_idx]
        unbinding_dist = rmsd[after_well_idx[0]] if len(after_well_idx) > 0 else rmsd[-1]

        stability_ranking.append((pose_name, barrier_height))
        summary_data.append({'Pose': pose_name, 'Plateau Height (kcal/mol)': round(barrier_height, 2)})
        
        # Store data for Top 3 plot
        all_pose_data[pose_name] = {
            'rmsd': rmsd,
            'mean': mean_fes,
            'std': std_fes,
            'barrier': barrier_height,
            'trials': len(trial_cols)
        }

        report_lines.append(f"POSE: {pose_name}")
        report_lines.append("-" * (len(pose_name) + 6))
        report_lines.append(f"  * Minimum Energy (Well Depth): -{well_depth:.2f} kcal/mol at {rmsd[min_idx]:.2f} Å")
        report_lines.append(f"  * Plateau Height (Barrier):    {barrier_height:.2f} kcal/mol")
        report_lines.append(f"  * Average Trial Variation:     {avg_std:.2f} kcal/mol")
        report_lines.append(f"  * Final Convergence Error:     {final_std:.2f} kcal/mol")
        report_lines.append(f"  * Unbinding Distance (95%):    {unbinding_dist:.2f} Å\n")

    # ---------------------------------------------------------
    # FINALIZE FIG 1 & 2
    # ---------------------------------------------------------
    plt.figure(1)
    plt.xlabel("RMSD (Å)", fontsize=14, fontweight='bold')
    plt.ylabel("Relative Free Energy (kcal/mol)", fontsize=14, fontweight='bold')
    plt.title("BPMD Stability Comparison (Mean + STD)", fontsize=16, fontweight='bold', pad=20)
    plt.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(frameon=True, fontsize=9, loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "ensemble_fes_std.png"), bbox_inches='tight')
    plt.close()

    plt.figure(2)
    if all_pose_data:
        all_means = [d['mean'] for d in all_pose_data.values()]
        global_mean = np.mean(all_means, axis=0)
        plt.plot(all_pose_data[next(iter(all_pose_data))]['rmsd'], global_mean, color='black', linewidth=2.0, linestyle='--', label='DATASET AVERAGE', alpha=0.8, zorder=10)
        
    plt.xlabel("RMSD (Å)", fontsize=14, fontweight='bold')
    plt.ylabel("Relative Free Energy (kcal/mol)", fontsize=14, fontweight='bold')
    plt.title("BPMD Dataset Benchmarking (Grand Mean Average)", fontsize=16, fontweight='bold', pad=20)
    plt.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(frameon=True, fontsize=9, loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "ensemble_fes_average.png"), bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # 4. TOP 3 LEADS PLOT
    # ---------------------------------------------------------
    top_3 = sorted(stability_ranking, key=lambda x: x[1], reverse=True)[:3]
    plt.figure(figsize=(10, 7), dpi=600) # Ultra high res for leads
    lead_colors = sns.color_palette("dark", 3)
    
    for i, (name, barrier) in enumerate(top_3):
        data = all_pose_data[name]
        plt.plot(data['rmsd'], data['mean'], label=f"{name} (Barrier: {barrier:.2f})", color=lead_colors[i], linewidth=2.0)
        if data['trials'] > 1:
            plt.fill_between(data['rmsd'], data['mean'] - data['std'], data['mean'] + data['std'], color=lead_colors[i], alpha=0.15)
            
    plt.xlabel("RMSD (Å)", fontsize=14, fontweight='bold')
    plt.ylabel("Relative Free Energy (kcal/mol)", fontsize=14, fontweight='bold')
    plt.title("Top 3 Leading Binding Poses (Highest Barriers)", fontsize=16, fontweight='bold', pad=20)
    plt.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=1.0)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(frameon=True, fontsize=12, loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "top_3_leads.png"), bbox_inches='tight')
    plt.close()
    
    if stability_ranking:
        best_pose = top_3[0]
        report_lines.append("CONCLUSION")
        report_lines.append("----------")
        report_lines.append(f"Based on the Plateau Height (Critical Barrier), {best_pose[0]} is the most kinetically stable pose with a barrier of {best_pose[1]:.2f} kcal/mol.\n")

    with open(report_txt, "w") as f:
        f.write("\n".join(report_lines))
    
    pd.DataFrame(summary_data).to_csv(summary_csv, index=False)







if __name__ == "__main__":
    data_files = sorted(glob("pose_*_all_trials.csv"))
    if data_files:
        analyze_and_plot(data_files, "plots/ensemble_fes_comparison.png", "plots/stability_summary.csv", "plots/stability_report.txt")
