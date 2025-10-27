import json

with open('docs/decisions_log.json', 'r') as f:
    data = json.load(f)

print("\n" + "="*70)
print("📊 VARIANCE ANALYSIS - Last 10 Tickets")
print("="*70)

# Get last 10 tickets
recent = data[-10:] if len(data) > 10 else data

for i, ticket in enumerate(recent, 1):
    prob = ticket.get('prob_p_win', 0)
    telemetry = ticket.get('telemetry', {})
    
    print(f"\n🎫 Ticket {i} ({ticket['ticket_id'][:8]}...)")
    print(f"   Value: ${ticket.get('value',0)}")
    print(f"   Probability: {prob:.6f} ({prob*100:.3f}%)")
    print(f"   Expected Payout: ${ticket.get('value',0) * prob:.4f}")
    print(f"   Telemetry:")
    print(f"     - Network Load: {telemetry.get('network_load', 0):.2f}")
    print(f"     - Relay Rep: {telemetry.get('relay_reputation', 0):.2f}")
    print(f"     - Budget Used: {telemetry.get('budget_utilization', 0):.2f}")
    print(f"     - Win Rate: {telemetry.get('recent_win_rate', 0):.3f}")

# Calculate variance
probs = [t.get('prob_p_win', 0) for t in recent]
if probs:
    import statistics
    print(f"\n{'='*70}")
    print("📈 STATISTICAL SUMMARY")
    print(f"{'='*70}")
    print(f"   Mean Probability: {statistics.mean(probs):.6f}")
    print(f"   Std Deviation: {statistics.stdev(probs) if len(probs) > 1 else 0:.6f}")
    print(f"   Min Probability: {min(probs):.6f}")
    print(f"   Max Probability: {max(probs):.6f}")
    print(f"   Range: {max(probs) - min(probs):.6f}")
    print(f"   ✅ Variance Detected: {'YES' if statistics.stdev(probs) > 0 else 'NO'}")
    print("="*70 + "\n")
