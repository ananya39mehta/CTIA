import time
import json
from typing import Dict, Any
import math

# Simple rule-based agent for V1.5 explainable decisions.
# Inputs expected: {'network_load': 0..1, 'relay_reputation': 0..1, 'budget_utilization': 0..1, 'recent_win_rate': 0..1}
# Returns: (prob_p_win: float, explanation: dict)

DEFAULT_BASE_PROB = 0.01  # baseline probability

def clamp(x, a=1e-9, b=1.0):
    return max(a, min(b, x))

def decide_probability(inputs: Dict[str, float]) -> (float, Dict[str, Any]):
    # normalize / default missing inputs
    network_load = clamp(inputs.get("network_load", 0.2))
    relay_reputation = clamp(inputs.get("relay_reputation", 0.8))
    budget_utilization = clamp(inputs.get("budget_utilization", 0.0))
    recent_win_rate = clamp(inputs.get("recent_win_rate", 0.01))

    # rule-based adjustments (interpretable and easy to explain)
    rationale = []
    prob = DEFAULT_BASE_PROB

    # if network congested -> reduce probability (to save budget and lower settlement load)
    net_factor = 1.0 - 0.6 * network_load  # if load=1 -> factor=0.4
    prob *= net_factor
    rationale.append({
        "feature": "network_load",
        "value": network_load,
        "effect": f"multiply prob by {net_factor:.3f} (higher load -> lower prob)",
    })

    # relay reputation: better rep increases prob linearly up to 2x
    rep_factor = 0.5 + relay_reputation * 1.5  # reputation=0 ->0.5, =1 ->2.0
    prob *= rep_factor
    rationale.append({
        "feature": "relay_reputation",
        "value": relay_reputation,
        "effect": f"multiply prob by {rep_factor:.3f} (higher rep -> higher prob)"
    })

    # budget_utilization: if we've used lots of budget, scale down
    budget_factor = 1.0 - 0.7 * budget_utilization
    prob *= budget_factor
    rationale.append({
        "feature": "budget_utilization",
        "value": budget_utilization,
        "effect": f"multiply prob by {budget_factor:.3f} (higher utilization -> lower prob)"
    })

    # Recent win rate (if already many winners recently, reduce to avoid overspending)
    winrate_factor = 1.0 - min(0.5, recent_win_rate * 10.0)
    prob *= winrate_factor
    rationale.append({
        "feature": "recent_win_rate",
        "value": recent_win_rate,
        "effect": f"multiply prob by {winrate_factor:.3f} (higher recent wins -> lower prob)"
    })

    # clamp probability to reasonable bounds
    prob = clamp(prob, a=1e-6, b=0.5)

    explanation = {
        "timestamp": time.time(),
        "inputs": {
            "network_load": network_load,
            "relay_reputation": relay_reputation,
            "budget_utilization": budget_utilization,
            "recent_win_rate": recent_win_rate
        },
        "base_prob": DEFAULT_BASE_PROB,
        "final_prob": prob,
        "steps": rationale
    }
    return float(prob), explanation

# small helper to simulate inputs for tests
def make_sample_inputs():
    return {
        "network_load": 0.2,
        "relay_reputation": 0.85,
        "budget_utilization": 0.1,
        "recent_win_rate": 0.01
    }
