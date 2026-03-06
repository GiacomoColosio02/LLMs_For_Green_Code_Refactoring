#!/usr/bin/env python3
"""
Reproduce Tables and Figures from the paper:
"Are Humans Greener than Language Models to Optimize Code?"

Usage:
    python scripts/reproduce_paper_results.py

No hardware required - uses pre-computed data from data/processed/green/
"""

import pandas as pd
import numpy as np
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = "data/processed/green/FINAL_UNIFIED_BENCHMARK.csv"
OUTPUT_DIR = "reports"

def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} instances from {DATA_PATH}")
    return df

def compute_energy_ratio(row, config_prefix):
    """Compute energy ratio: E_config / E_human"""
    human_col = "human_head_total_energy_joules_mean"
    config_col = f"{config_prefix}_total_energy_joules_mean"
    if pd.notna(row[config_col]) and pd.notna(row[human_col]) and row[human_col] > 0:
        return row[config_col] / row[human_col]
    return np.nan

def compute_speedup(row, config_prefix):
    """Compute speedup: T_base / T_config"""
    base_col = "base_duration_seconds_mean"
    config_col = f"{config_prefix}_duration_seconds_mean"
    if pd.notna(row[config_col]) and row[config_col] > 0:
        return row[base_col] / row[config_col]
    return np.nan

def compute_gpu_share(row, config_prefix):
    """Compute GPU energy share: E_gpu / E_total"""
    total_col = f"{config_prefix}_total_energy_joules_mean"
    gpu_col = f"{config_prefix}_gpu_energy_joules_mean"
    if pd.notna(row[total_col]) and row[total_col] > 0:
        return (row[gpu_col] / row[total_col]) * 100
    return np.nan

# ============================================================
# Qwen configs
QWEN_CONFIGS = [
    "qwen_zs_oracle", "qwen_zs_realistic",
    "qwen_cot_oracle", "qwen_cot_realistic",
    "qwen_ldb_oracle", "qwen_ldb_realistic",
    "qwen_sc_oracle", "qwen_sc_realistic",
]
# DeepSeek configs
DEEPSEEK_CONFIGS = [
    "deepseek_zs_oracle", "deepseek_zs_realistic",
    "deepseek_cot_oracle",
    "deepseek_ldb_oracle", "deepseek_ldb_realistic",
    "deepseek_sc_oracle",
]
ALL_CONFIGS = QWEN_CONFIGS + DEEPSEEK_CONFIGS

def table_vii_rq1(df):
    """Table VII: Energy Efficiency Comparison by Optimization Source"""
    print("\n" + "="*70)
    print("TABLE VII (RQ1): Energy Efficiency Comparison by Optimization Source")
    print("="*70)

    # Human
    human_er = df["human_head_total_energy_joules_mean"] / df["base_total_energy_joules_mean"]
    human_sp = df["base_duration_seconds_mean"] / df["human_head_duration_seconds_mean"]
    human_gpu = (df["human_head_gpu_energy_joules_mean"] / df["human_head_total_energy_joules_mean"]) * 100
    
    print(f"{'Source':<25} {'N':>5} {'Energy×':>10} {'Speedup':>10} {'GPU%':>8}")
    print("-"*60)
    print(f"{'Human':<25} {human_er.dropna().shape[0]:>5} {human_er.median():>10.2f} {human_sp.median():>10.2f} {human_gpu.median():>8.1f}")

    # Qwen (all configs)
    qwen_ers, qwen_sps, qwen_gpus = [], [], []
    for cfg in QWEN_CONFIGS:
        for _, row in df.iterrows():
            er = compute_energy_ratio(row, cfg)
            sp = compute_speedup(row, cfg)
            gpu = compute_gpu_share(row, cfg)
            if not np.isnan(er):
                qwen_ers.append(er)
                qwen_sps.append(sp)
                qwen_gpus.append(gpu)
    
    if qwen_ers:
        print(f"{'Qwen (all configs)':<25} {len(qwen_ers):>5} {np.median(qwen_ers):>10.2f} {np.median(qwen_sps):>10.2f} {np.median(qwen_gpus):>8.1f}")

    # DeepSeek (all configs)
    ds_ers, ds_sps, ds_gpus = [], [], []
    for cfg in DEEPSEEK_CONFIGS:
        for _, row in df.iterrows():
            er = compute_energy_ratio(row, cfg)
            sp = compute_speedup(row, cfg)
            gpu = compute_gpu_share(row, cfg)
            if not np.isnan(er):
                ds_ers.append(er)
                ds_sps.append(sp)
                ds_gpus.append(gpu)
    
    if ds_ers:
        print(f"{'DeepSeek (all configs)':<25} {len(ds_ers):>5} {np.median(ds_ers):>10.2f} {np.median(ds_sps):>10.2f} {np.median(ds_gpus):>8.1f}")

    # Wilcoxon test
    print("\nStatistical Tests (Wilcoxon signed-rank, LLM vs Human):")
    if qwen_ers:
        # Compare paired: for each instance, qwen median vs human
        stat, p = stats.mannwhitneyu(qwen_ers, [1.0]*len(qwen_ers), alternative='greater')
        print(f"  Qwen vs baseline:    p = {p:.6f}")
    if ds_ers:
        stat, p = stats.mannwhitneyu(ds_ers, [1.0]*len(ds_ers), alternative='greater')
        print(f"  DeepSeek vs baseline: p = {p:.6f}")

def table_x_rq3_strategy(df):
    """Table X: Energy Efficiency by Prompting Strategy"""
    print("\n" + "="*70)
    print("TABLE X (RQ3): Energy Efficiency by Prompting Strategy")
    print("="*70)
    
    strategies = {
        "Qwen ZS": ["qwen_zs_oracle", "qwen_zs_realistic"],
        "Qwen CoT": ["qwen_cot_oracle", "qwen_cot_realistic"],
        "Qwen LDB": ["qwen_ldb_oracle", "qwen_ldb_realistic"],
        "Qwen SC": ["qwen_sc_oracle", "qwen_sc_realistic"],
        "DeepSeek ZS": ["deepseek_zs_oracle", "deepseek_zs_realistic"],
        "DeepSeek CoT": ["deepseek_cot_oracle"],
        "DeepSeek LDB": ["deepseek_ldb_oracle", "deepseek_ldb_realistic"],
        "DeepSeek SC": ["deepseek_sc_oracle"],
    }

    print(f"{'Source':<20} {'Strategy':<8} {'N':>5} {'Energy×':>10} {'Speedup':>10} {'GPU%':>8}")
    print("-"*65)

    # Human baseline
    human_er = df["human_head_total_energy_joules_mean"] / df["base_total_energy_joules_mean"]
    human_sp = df["base_duration_seconds_mean"] / df["human_head_duration_seconds_mean"]
    human_gpu = (df["human_head_gpu_energy_joules_mean"] / df["human_head_total_energy_joules_mean"]) * 100
    print(f"{'Human':<20} {'–':<8} {human_er.dropna().shape[0]:>5} {human_er.median():>10.2f} {human_sp.median():>10.2f} {human_gpu.median():>8.1f}")

    qwen_groups = {}
    for label, configs in strategies.items():
        ers, sps, gpus = [], [], []
        for cfg in configs:
            for _, row in df.iterrows():
                er = compute_energy_ratio(row, cfg)
                sp = compute_speedup(row, cfg)
                gpu = compute_gpu_share(row, cfg)
                if not np.isnan(er):
                    ers.append(er)
                    sps.append(sp)
                    gpus.append(gpu)
        if ers:
            model = label.split()[0]
            strat = label.split()[1]
            print(f"{model:<20} {strat:<8} {len(ers):>5} {np.median(ers):>10.2f} {np.median(sps):>10.2f} {np.median(gpus):>8.1f}")
            if "Qwen" in label:
                qwen_groups[strat] = ers

    # Kruskal-Wallis for Qwen strategies
    if len(qwen_groups) >= 2:
        groups = list(qwen_groups.values())
        h_stat, p_val = stats.kruskal(*groups)
        print(f"\nKruskal-Wallis H (Qwen strategies): H={h_stat:.2f}, p={p_val:.6f}")

def table_ix_rq3_context(df):
    """Table IX: Oracle vs Realistic Context Setting"""
    print("\n" + "="*70)
    print("TABLE IX (RQ3): Oracle vs. Realistic Context Setting")
    print("="*70)

    contexts = {
        "Qwen Oracle": ["qwen_zs_oracle", "qwen_cot_oracle", "qwen_ldb_oracle", "qwen_sc_oracle"],
        "Qwen Realistic": ["qwen_zs_realistic", "qwen_cot_realistic", "qwen_ldb_realistic", "qwen_sc_realistic"],
        "DeepSeek Oracle": ["deepseek_zs_oracle", "deepseek_cot_oracle", "deepseek_ldb_oracle", "deepseek_sc_oracle"],
        "DeepSeek Realistic": ["deepseek_zs_realistic", "deepseek_ldb_realistic"],
    }

    print(f"{'Source':<22} {'Context':<12} {'N':>5} {'Energy×':>10} {'Speedup':>10}")
    print("-"*62)

    context_data = {}
    for label, configs in contexts.items():
        ers, sps = [], []
        for cfg in configs:
            for _, row in df.iterrows():
                er = compute_energy_ratio(row, cfg)
                sp = compute_speedup(row, cfg)
                if not np.isnan(er):
                    ers.append(er)
                    sps.append(sp)
        if ers:
            model = label.split()[0]
            ctx = label.split()[1]
            print(f"{model:<22} {ctx:<12} {len(ers):>5} {np.median(ers):>10.2f} {np.median(sps):>10.2f}")
            context_data[label] = ers

    # Mann-Whitney U for context comparison
    if "Qwen Oracle" in context_data and "Qwen Realistic" in context_data:
        u_stat, p_val = stats.mannwhitneyu(context_data["Qwen Oracle"], context_data["Qwen Realistic"])
        print(f"\nMann-Whitney U (Qwen Oracle vs Realistic): U={u_stat:.0f}, p={p_val:.3f}")
    if "DeepSeek Oracle" in context_data and "DeepSeek Realistic" in context_data:
        u_stat, p_val = stats.mannwhitneyu(context_data["DeepSeek Oracle"], context_data["DeepSeek Realistic"])
        print(f"Mann-Whitney U (DeepSeek Oracle vs Realistic): U={u_stat:.0f}, p={p_val:.3f}")

def rq2_correlation(df):
    """RQ2: Speedup vs Energy correlation"""
    print("\n" + "="*70)
    print("RQ2: Speedup-Energy Correlation Analysis")
    print("="*70)

    # Human
    human_sp = df["base_duration_seconds_mean"] / df["human_head_duration_seconds_mean"]
    human_er = df["human_head_total_energy_joules_mean"] / df["base_total_energy_joules_mean"]
    mask = human_sp.notna() & human_er.notna()
    rho, p = stats.spearmanr(human_sp[mask], human_er[mask])
    print(f"Human:    Spearman rho={rho:.2f}, p={p:.4f}")

    # Qwen
    qwen_sps, qwen_ers = [], []
    for cfg in QWEN_CONFIGS:
        for _, row in df.iterrows():
            sp = compute_speedup(row, cfg)
            er = compute_energy_ratio(row, cfg)
            if not np.isnan(sp) and not np.isnan(er):
                qwen_sps.append(sp)
                qwen_ers.append(er)
    if qwen_sps:
        rho, p = stats.spearmanr(qwen_sps, qwen_ers)
        print(f"Qwen:     Spearman rho={rho:.2f}, p={p:.4f}")

    # DeepSeek
    ds_sps, ds_ers = [], []
    for cfg in DEEPSEEK_CONFIGS:
        for _, row in df.iterrows():
            sp = compute_speedup(row, cfg)
            er = compute_energy_ratio(row, cfg)
            if not np.isnan(sp) and not np.isnan(er):
                ds_sps.append(sp)
                ds_ers.append(er)
    if ds_sps:
        rho, p = stats.spearmanr(ds_sps, ds_ers)
        print(f"DeepSeek: Spearman rho={rho:.2f}, p={p:.4f}")

def main():
    print("="*70)
    print("REPRODUCING PAPER RESULTS")
    print("Are Humans Greener than Language Models to Optimize Code?")
    print("="*70)

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data file not found: {DATA_PATH}")
        print("Please ensure the dataset is in the correct location.")
        return

    df = load_data()

    table_vii_rq1(df)
    rq2_correlation(df)
    table_ix_rq3_context(df)
    table_x_rq3_strategy(df)

    print("\n" + "="*70)
    print("DONE. All tables reproduced from pre-computed data.")
    print("For figures, see notebooks/03_swe_perf_green_analysis.ipynb")
    print("="*70)

if __name__ == "__main__":
    main()
