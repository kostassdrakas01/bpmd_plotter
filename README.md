# BPMD Plotter & Stability Analyzer

A high-fidelity analysis pipeline for Binding Pose Metadynamics (BPMD) simulations, designed to automate data extraction, ensemble analysis, and publication-quality visualization of Free Energy Surfaces (FES).

## Features

- **Automated Extraction**: Streamlined processing of `.cvseq`, `.ene`, and `.fes` files from Desmond/Schrödinger simulation archives.
- **Ensemble Analysis**: Aggregates multi-trial simulation data to calculate mean trends and standard deviation (SEM/STD) for robust lead prioritization.
- **Publication-Quality Plots**: Generates 2D FES profiles with hairline aesthetics, shaded error regions, and dynamic minima detection.
- **Stability Metrics**: Calculates Critical Barrier, Plateau Height, Unbinding Distance, and Convergence Error to rank ligand stability.
- **Consolidated Reporting**: Automatically generates comparative dashboards and structured CSV summaries for high-throughput screening.

## Project Structure

- `master_analysis.py`: Main orchestration script for the analysis pipeline.
- `consolidate_bpmd.py`: Utility to aggregate multi-trial simulation results into unified datasets.
- `scripts/extractor.py`: Handles raw data extraction and sanitization.
- `scripts/analyzer.py`: Core logic for FES reconstruction and stability metric calculation.
- `plots/`: Directory containing generated ensemble visualizations and stability reports.

## Getting Started

1. Ensure all simulation data is organized in the `data/` directory (ignored by git due to size).
2. Run the master analysis script:
   ```bash
   python master_analysis.py
   ```
3. Check the `plots/` directory for generated FES profiles and the `stability_summary.csv` for lead prioritization.

## Technology Stack

- **Language**: Python 3.x
- **Core Libraries**: NumPy, Matplotlib, SciPy
- **Simulation Source**: Schrödinger Desmond Binding Pose Metadynamics
