"""
Generate Live Telemetry Dataset + EDA for CTIA v2.1
---------------------------------------------------
This script connects to your running FastAPI CTIA app, collects live telemetry
snapshots, simulates "final_prob" values using rule-based logic, and performs EDA.

Output:
    docs/live_telemetry_dataset.json
    notebooks/live_telemetry_eda_summary.csv
    notebooks/live_telemetry_correlation.png
    notebooks/live_telemetry_histograms.png
"""

import os
import time
import json
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Optional dependencies
try:
    import requests
except ImportError:
    raise SystemExit("Install requests first: pip install requests")

# ----------------------------
# PATH SETUP
# ----------------------------
ROOT = Path(__file__).resolve().parents[1]
DOCS_PATH = ROOT / "docs"
NOTEBOOKS_PATH = ROOT / "notebooks"
DOCS_PATH.mkdir(parents=True, exist_ok=True)
NOTEBOOKS_PATH.mkdir(parents=True, exist_ok=True)

# ----------------------------
# CONFIGURATION
# ----------------------------
TELEMETRY_URL = "http://127.0.0.1:8000/telemetry"
N_SAMPLES = 2000       # how many telemetry points to capture
DELAY_SEC = 0.5       # wait between samples (in seconds)
OUTPUT_PATH = DOCS_PATH / "live_telemetry_dataset.json"

# ----------------------------
# RULE-BASED FINAL PROBABILITY SIMULATION
# ----------------------------
def simulate_decision(inputs: dict, base_prob: float = None) -> float:
    """Rule-based multiplier logic, similar to simulation script, but with slightly wider base range."""
    if base_prob is None:
        base_prob = np.random.uniform(0.01, 0.15)
    prob = base_prob
    prob *= (1 - inputs["network_load"] * 0.3)
    prob *= (1 + inputs["relay_reputation"] * 0.9)
    prob *= (1 - inputs["budget_utilization"] * 0.35)
    prob *= (1 - inputs["recent_win_rate"] * 5.0)
    return float(max(1e-6, min(prob, 0.5)))

# ----------------------------
# TELEMETRY COLLECTION
# ----------------------------
def collect_live_telemetry(n_samples: int = 200, delay: float = 1.0, url: str = TELEMETRY_URL):
    print(f"📡 Starting live telemetry collection from {url}")
    print(f"Target: {n_samples} samples, delay {delay:.2f}s")
    rows = []
    for i in range(n_samples):
        try:
            r = requests.get(url, timeout=3.0)
            if r.status_code == 200:
                payload = r.json()
                metrics = payload.get("metrics") or payload
                inputs = {
                    "network_load": float(metrics.get("network_load", np.nan)),
                    "relay_reputation": float(metrics.get("relay_reputation", np.nan)),
                    "budget_utilization": float(metrics.get("budget_utilization", np.nan)),
                    "recent_win_rate": float(metrics.get("recent_win_rate", np.nan))
                }
                final_prob = simulate_decision(inputs)
                rows.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "inputs": inputs,
                    "final_prob": final_prob
                })
                print(f"[{i+1}/{n_samples}] network_load={inputs['network_load']:.2f}, "
                      f"relay_rep={inputs['relay_reputation']:.2f}, final_prob≈{final_prob:.4f}")
            else:
                print(f"[{i+1}] ⚠️ HTTP {r.status_code}, skipping sample.")
        except Exception as e:
            print(f"[{i+1}] ⚠️ Error: {e}")
        time.sleep(delay)
    print(f"✅ Completed {len(rows)} telemetry captures.")
    return rows

# ----------------------------
# SAVE DATASET
# ----------------------------
def save_dataset(rows, path=OUTPUT_PATH):
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"💾 Saved dataset -> {path} ({len(rows)} samples)")

# ----------------------------
# PERFORM BASIC EDA
# ----------------------------
def perform_eda(json_path=OUTPUT_PATH):
    with open(json_path, "r") as f:
        data = json.load(f)
    df = pd.DataFrame([{
        "network_load": d["inputs"]["network_load"],
        "relay_reputation": d["inputs"]["relay_reputation"],
        "budget_utilization": d["inputs"]["budget_utilization"],
        "recent_win_rate": d["inputs"]["recent_win_rate"],
        "final_prob": d["final_prob"]
    } for d in data])
    print(f"\n📊 Performing EDA on {len(df)} samples...")
    summary_path = NOTEBOOKS_PATH / "live_telemetry_eda_summary.csv"
    corr_path = NOTEBOOKS_PATH / "live_telemetry_correlation.png"
    hist_path = NOTEBOOKS_PATH / "live_telemetry_histograms.png"

    df.describe().to_csv(summary_path)
    print(f"Summary saved -> {summary_path}")

    corr = df.corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm")
    plt.tight_layout()
    plt.savefig(corr_path)
    plt.close()
    print(f"Correlation heatmap saved -> {corr_path}")

    df.hist(bins=25, figsize=(10, 8))
    plt.tight_layout()
    plt.savefig(hist_path)
    plt.close()
    print(f"Histograms saved -> {hist_path}")
    print("\nEDA complete ✅")

# ----------------------------
# MAIN EXECUTION
# ----------------------------
if __name__ == "__main__":
    data = collect_live_telemetry(N_SAMPLES, DELAY_SEC, TELEMETRY_URL)
    save_dataset(data, OUTPUT_PATH)
    perform_eda(OUTPUT_PATH)
    print("\n✅ Live telemetry dataset generation + EDA complete.")
