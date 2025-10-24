"""
Simulate dataset for CTIA Explainable AI (xAI) Model
----------------------------------------------------
This script generates synthetic data that mimics how the current
rule-based system decides the probability of winning based on
four key factors:
    - network_load
    - relay_reputation
    - budget_utilization
    - recent_win_rate
Each sample stores both the inputs and the resulting final_prob.

Output:
    docs/simulated_decisions.json
"""

import random
import json
import os
from datetime import datetime

# --- Core rule-based logic (same as app/agent.py) ---
def simulate_decision(inputs: dict) -> float:
    """Compute final probability using rule-based multipliers."""
    base_prob = 0.01  # base chance
    prob = base_prob

    prob *= (1 - inputs["network_load"] * 0.3)       # higher load → lower prob
    prob *= (1 + inputs["relay_reputation"] * 0.9)   # higher reputation → higher prob
    prob *= (1 - inputs["budget_utilization"] * 0.35) # higher utilization → lower prob
    prob *= (1 - inputs["recent_win_rate"] * 5.0)     # frequent wins → lower prob

    return max(prob, 0.0001)


# --- Generate dataset ---
def generate_dataset(n_samples: int = 1000):
    data = []

    for i in range(n_samples):
        # random realistic inputs
        inputs = {
            "network_load": round(random.uniform(0, 1), 2),
            "relay_reputation": round(random.uniform(0, 1), 2),
            "budget_utilization": round(random.uniform(0, 1), 2),
            "recent_win_rate": round(random.uniform(0, 0.2), 3)
        }

        final_prob = simulate_decision(inputs)
        entry = {
            "id": i + 1,
            "timestamp": datetime.utcnow().isoformat(),
            "inputs": inputs,
            "final_prob": final_prob
        }
        data.append(entry)

    return data


# --- Save to JSON file ---
def save_dataset(data, output_path="docs/simulated_decisions.json"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"✅ Generated {len(data)} samples in {output_path}")


if __name__ == "__main__":
    dataset = generate_dataset(n_samples=1000)
    save_dataset(dataset)
