import subprocess
import os
from glob import glob
from scripts.analyzer import analyze_and_plot

def run_pipeline():
    print("="*60)
    print("BPMD ANALYSIS PIPELINE ORCHESTRATOR")
    print("="*60)

    # 1. Run Consolidation
    print("\n[*] Phase 1: Data Consolidation...")
    subprocess.run(["python3", "consolidate_bpmd.py"], check=True)

    # 1.5 Run Extraction
    print("\n[*] Phase 1.5: Raw Data Extraction (.fes and .covlar)...")
    subprocess.run(["python3", "scripts/extractor.py"], check=True)

    # 2. Identify master CSVs
    csv_files = sorted(glob("pose_*_all_trials.csv"))
    if not csv_files:
        print("[!] No consolidated CSV files found. Check consolidate_bpmd.py output.")
        return

    # 3. Run Statistical Analysis and Plotting
    print("\n[*] Phase 2: Statistical Analysis and Visualization...")
    output_plot = "plots/ensemble_fes_comparison.png"
    summary_csv = "plots/stability_summary.csv"
    report_txt = "plots/stability_report.txt"
    
    analyze_and_plot(csv_files, output_plot, summary_csv, report_txt)

    print("\n" + "="*60)
    print("[FINISHED] Results are available in the 'plots/' folder.")
    print("="*60)

if __name__ == "__main__":
    run_pipeline()
