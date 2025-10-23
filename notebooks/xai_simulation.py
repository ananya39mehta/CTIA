"""
XAI Simulation Notebook (CTIA V1.5)
-----------------------------------
This notebook simulates how the CTIA agent decides ticket win probabilities
based on contextual parameters and provides explainability.
"""
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.agent import decide_probability
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Simulate varying inputs
network_load = np.linspace(0.1, 0.9, 9)
relay_reputation = np.linspace(0.4, 0.95, 6)
budget_utilization = np.linspace(0.05, 0.8, 8)
recent_win_rate = np.linspace(0.0, 0.3, 7)

rows = []

# Generate combinations (few samples)
for nl in network_load:
    for rr in relay_reputation:
        for bu in budget_utilization:
            for rw in recent_win_rate:
                inputs = {
                    "network_load": round(nl, 2),
                    "relay_reputation": round(rr, 2),
                    "budget_utilization": round(bu, 2),
                    "recent_win_rate": round(rw, 2)
                }
                prob, explanation = decide_probability(inputs)
                rows.append({
                    **inputs,
                    "prob_p_win": prob,
                    "explanation": explanation
                })

df = pd.DataFrame(rows)

print("✅ Simulation completed.")
print(df.head())

# Save dataframe to CSV
df.to_csv("notebooks/xai_decision_simulation.csv", index=False)

# Visualize: effect of reputation on probability
plt.figure(figsize=(8, 5))
for load in [0.1, 0.5, 0.9]:
    subset = df[df["network_load"] == round(load, 2)]
    plt.plot(subset["relay_reputation"], subset["prob_p_win"], label=f"Network load {load}")
plt.xlabel("Relay Reputation")
plt.ylabel("Probability of Win")
plt.title("Agentic XAI Decision Behavior under Different Loads")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("notebooks/xai_behavior_plot.png")
plt.show()
