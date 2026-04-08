import os
import glob
import pandas as pd
import numpy as np
import math

RESULTS_DIR = "results"
TAIL = 200
THRESHOLD = -160
WINDOW = 50

def safe_read(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

def welch_ttest(a, b):
    n1, n2 = len(a), len(b)
    m1, m2 = np.mean(a), np.mean(b)
    v1, v2 = np.var(a, ddof=1), np.var(b, ddof=1)
    se = np.sqrt(v1/n1 + v2/n2)
    if se == 0:
        return 0, 1.0
    t_stat = (m1 - m2) / se
    # normal approx for large N
    z = abs(t_stat)
    p_val = 1.0 - math.erf(z / math.sqrt(2.0))
    return t_stat, p_val

def analyze_significance():
    print("=== 1. Statistical Significance Testing (Last 200 Episodes) ===")
    
    modes = ["baseline", "llm-code", "llm-direct"]
    
    for sparsity in [0, 1]:
        print(f"\n--- Sparsity Level {sparsity} ---")
        
        data = {}
        for mode in modes:
            df = safe_read(f"{RESULTS_DIR}/exp1_{mode}_s{sparsity}.csv")
            if df is not None:
                data[mode] = df['total_reward'].iloc[-TAIL:].values
        
        if "baseline" not in data:
            print("Baseline data missing, skipping tests.")
            continue
            
        base_vals = data["baseline"]
        base_mean = np.mean(base_vals)
        print(f"Baseline Mean: {base_mean:.2f}")
        
        for mode in ["llm-code", "llm-direct"]:
            if mode not in data:
                continue
            comp_vals = data[mode]
            comp_mean = np.mean(comp_vals)
            
            # Independent Welch T-test
            t_stat, p_t = welch_ttest(comp_vals, base_vals)
            
            diff = comp_mean - base_mean
            
            sig_text_t = "Statistically Significant" if p_t < 0.05 else "Not Statistically Significant"
            
            print(f"{mode} Mean: {comp_mean:.2f} (Diff: {diff:+.2f})")
            print(f"  T-test: p={p_t:.4f} -> {sig_text_t}")
            
            if p_t > 0.05:
                print(f"  *Conclusion*: Although there is a change in mean, it is not statistically significant (p > 0.05). We cannot determine if the advantage is due to the algorithm or random fluctuations.")
            else:
                if diff > 0:
                    print(f"  *Conclusion*: Has a statistically significant advantage (p < 0.05)")
                else:
                    print(f"  *Conclusion*: Performance significantly worsened (p < 0.05)")

def analyze_time_to_threshold():
    print(f"\n=== 2. Time-to-Threshold (Sample Efficiency) ===")
    print(f"Target Threshold: {THRESHOLD} (using {WINDOW}-episode rolling mean)")
    
    modes = ["baseline", "llm-code", "llm-direct"]
    
    for sparsity in [0, 1]:
        print(f"\n--- Sparsity Level {sparsity} ---")
        for mode in modes:
            df = safe_read(f"{RESULTS_DIR}/exp1_{mode}_s{sparsity}.csv")
            if df is not None:
                smoothed = df['total_reward'].rolling(window=WINDOW, min_periods=1).mean()
                # Find first index where smoothed >= THRESHOLD
                crossed = smoothed >= THRESHOLD
                if crossed.any():
                    first_ep = df['episode'][crossed.idxmax()]
                    print(f"{mode:12s}: First episode to stably cross the threshold = {first_ep}")
                else:
                    max_val = smoothed.max()
                    print(f"{mode:12s}: Did not reach threshold - Highest rolling mean was {max_val:.2f}")

if __name__ == "__main__":
    report_path = f"{RESULTS_DIR}/extra_analysis_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        import contextlib
        with contextlib.redirect_stdout(f):
            analyze_significance()
            analyze_time_to_threshold()
    
    # Also print to console
    analyze_significance()
    analyze_time_to_threshold()
    print(f"\nAnalysis report saved to {report_path}")
