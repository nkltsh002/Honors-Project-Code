import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'scripts'))
from analyze_results import *

# Set up arguments
full_dir = 'runs/full_curriculum_20250824_192155'
classic_dir = 'runs/classic_curriculum_20250824_192155'
output_dir = 'results'

print("=== ANALYSIS EXECUTION ===")
print(f"Full directory: {full_dir}")
print(f"Classic directory: {classic_dir}")
print(f"Output directory: {output_dir}")

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Load CSVs
print("\n=== LOADING DATA ===")
full_csvs = find_logs(full_dir)
classic_csvs = find_logs(classic_dir)
print(f"Found FULL CSVs: {full_csvs}")
print(f"Found CLASSIC CSVs: {classic_csvs}")

full_df = load_all_csv(full_csvs)
classic_df = load_all_csv(classic_csvs)
print(f'Loaded FULL data: {len(full_df)} rows')
print(f'Loaded CLASSIC data: {len(classic_df)} rows')

# Generate plots
print("\n=== GENERATING PLOTS ===")
plot_curves(full_df, os.path.join(output_dir, 'full_plots'), 'Full Curriculum')
plot_curves(classic_df, os.path.join(output_dir, 'classic_plots'), 'Classic Curriculum')
print("Plots generated")

# Load summaries from JSON
print("\n=== LOADING SUMMARIES ===")
full_json = load_summary(full_dir)
classic_json = load_summary(classic_dir)
print(f'Found FULL JSON: {full_json}')
print(f'Found CLASSIC JSON: {classic_json}')

full_rows, full_meta = parse_json_summary(full_json) if full_json else ([],{})
classic_rows, classic_meta = parse_json_summary(classic_json) if classic_json else ([],{})

# Write LaTeX tables
print("\n=== GENERATING LATEX TABLES ===")
with open(os.path.join(output_dir, 'table_full.tex'), 'w', encoding='utf-8') as f:
    f.write(latex_table(full_rows, 'Full curriculum training results.', 'tab:full'))
with open(os.path.join(output_dir, 'table_classic.tex'), 'w', encoding='utf-8') as f:
    f.write(latex_table(classic_rows, 'Classic curriculum training results.', 'tab:classic'))
print("LaTeX tables generated")

# CSV summaries
import pandas as pd
def rows_to_df(rows):
    return pd.DataFrame(rows, columns=['env','threshold','best','gens','solved']) if rows else pd.DataFrame(columns=['env','threshold','best','gens','solved'])

rows_to_df(full_rows).to_csv(os.path.join(output_dir, 'summary_full.csv'), index=False)
rows_to_df(classic_rows).to_csv(os.path.join(output_dir, 'summary_classic.csv'), index=False)
print("CSV summaries generated")

# Print meta to console so it's visible in logs
print("\n=== FINAL RESULTS ===")
print('FULL meta:', full_meta)
print('CLASSIC meta:', classic_meta)
print('Outputs in:', os.path.abspath(output_dir))
print("=== ANALYSIS COMPLETE ===")
