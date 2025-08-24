import argparse, json, os, glob, pandas as pd, numpy as np, matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 120

def find_logs(run_dir: str):
    # Look for a curriculum-level CSV or per-env CSVs
    csvs = []
    for root, _, files in os.walk(run_dir):
        for f in files:
            if f.endswith(".csv"):
                csvs.append(os.path.join(root, f))
    return csvs

def load_summary(run_dir: str):
    # Prefer a single summary json if present
    candidates = glob.glob(os.path.join(run_dir, "**", "curriculum_results.json"), recursive=True)
    return candidates[0] if candidates else None

def load_all_csv(csv_files):
    frames = []
    for path in csv_files:
        try:
            df = pd.read_csv(path)
            # Try to standardize common column names
            cols = {c.lower(): c for c in df.columns}
            # Heuristics to map columns
            for k in list(df.columns):
                if k.strip().lower() in ["env","env_id","environment","game"]:
                    df.rename(columns={k: "env_id"}, inplace=True)
                if k.strip().lower() in ["gen","generation","generations"]:
                    df.rename(columns={k: "generation"}, inplace=True)
                if k.strip().lower() in ["mean","mean_score","avg_score","score"]:
                    df.rename(columns={k: "mean_score"}, inplace=True)
                if k.strip().lower() in ["best","best_score"]:
                    df.rename(columns={k: "best_score"}, inplace=True)
                if k.strip().lower() in ["threshold","threshold_score","target"]:
                    df.rename(columns={k: "threshold_score"}, inplace=True)
            if "env_id" in df.columns and "generation" in df.columns and "mean_score" in df.columns:
                frames.append(df)
        except Exception as e:
            print("Skip CSV:", path, e)
    if frames:
        all_df = pd.concat(frames, ignore_index=True)
        # Drop obvious dups
        all_df = all_df.drop_duplicates(subset=[c for c in all_df.columns if c in ["env_id","generation","mean_score"]])
        return all_df
    return pd.DataFrame()

def plot_curves(df: pd.DataFrame, outdir: str, title: str):
    os.makedirs(outdir, exist_ok=True)
    for env, sub in df.groupby("env_id"):
        sub = sub.sort_values(by="generation")
        if "threshold_score" in sub.columns:
            thr = sub["threshold_score"].dropna().iloc[-1] if not sub["threshold_score"].dropna().empty else None
        else:
            thr = None
        plt.figure()
        plt.plot(sub["generation"], sub["mean_score"], label=f"{env} mean")
        if "best_score" in sub.columns:
            plt.plot(sub["generation"], sub["best_score"], linestyle="--", label=f"{env} best")
        if thr is not None:
            plt.axhline(y=thr, linestyle=":", label=f"threshold={thr}")
        plt.xlabel("Generation"); plt.ylabel("Score"); plt.title(f"{title}: {env}")
        plt.legend(); plt.tight_layout()
        path_png = os.path.join(outdir, f"{env.replace('/','_')}.png")
        path_pdf = os.path.join(outdir, f"{env.replace('/','_')}.pdf")
        plt.savefig(path_png); plt.savefig(path_pdf); plt.close()

def latex_table(summary_rows, caption, label):
    # Build a compact LaTeX table
    if not summary_rows:
        return "% No data"
    cols = ["Environment","Threshold","Best","Generations","Solved?"]
    lines = []
    lines.append("\\begin{table}[!ht]")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{lrrrr}")
    lines.append("\\hline")
    lines.append(" & ".join(cols) + " \\\\")
    lines.append("\\hline")
    for r in summary_rows:
        lines.append(f"{r['env']} & {r['threshold']} & {r['best']} & {r['gens']} & {r['solved']} \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")
    return "\n".join(lines)

def parse_json_summary(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            j = json.load(f)
        rows = []
        for t in j.get("tasks", []):
            rows.append({
                "env": t.get("env_id",""),
                "threshold": t.get("threshold_score",""),
                "best": t.get("best_score",""),
                "gens": t.get("generations_trained",""),
                "solved": "Yes" if t.get("solved", False) else "No"
            })
        meta = {
            "tasks_completed": j.get("tasks_completed",None),
            "total_tasks": j.get("total_tasks",None),
            "success_rate": j.get("success_rate",None),
            "total_time_hours": j.get("total_time_hours",None),
        }
        return rows, meta
    except Exception as e:
        return [], {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", required=True, help="path to FULL curriculum run dir")
    ap.add_argument("--classic", required=True, help="path to CLASSIC curriculum run dir")
    ap.add_argument("--outdir", default="results", help="output dir for plots and tables")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Load CSVs
    full_df = load_all_csv(find_logs(args.full))
    classic_df = load_all_csv(find_logs(args.classic))

    # Plots
    plot_curves(full_df, os.path.join(args.outdir, "full_plots"), "Full Curriculum")
    plot_curves(classic_df, os.path.join(args.outdir, "classic_plots"), "Classic Curriculum")

    # Summaries from JSON
    full_json = load_summary(args.full)
    classic_json = load_summary(args.classic)
    full_rows, full_meta = parse_json_summary(full_json) if full_json else ([],{})
    classic_rows, classic_meta = parse_json_summary(classic_json) if classic_json else ([],{})

    # Write LaTeX tables
    with open(os.path.join(args.outdir, "table_full.tex"), "w", encoding="utf-8") as f:
        f.write(latex_table(full_rows, "Full curriculum training results.", "tab:full"))
    with open(os.path.join(args.outdir, "table_classic.tex"), "w", encoding="utf-8") as f:
        f.write(latex_table(classic_rows, "Classic curriculum training results.", "tab:classic"))

    # Also dump a CSV summary for convenience
    def rows_to_df(rows):
        return pd.DataFrame(rows, columns=["env","threshold","best","gens","solved"]) if rows else pd.DataFrame(columns=["env","threshold","best","gens","solved"])
    rows_to_df(full_rows).to_csv(os.path.join(args.outdir, "summary_full.csv"), index=False)
    rows_to_df(classic_rows).to_csv(os.path.join(args.outdir, "summary_classic.csv"), index=False)

    # Print meta to console so it's visible in logs
    print("FULL meta:", full_meta)
    print("CLASSIC meta:", classic_meta)
    print("Outputs in:", os.path.abspath(args.outdir))

if __name__ == "__main__":
    main()
