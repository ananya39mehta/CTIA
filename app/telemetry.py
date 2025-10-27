# app/telemetry.py
"""
Real Telemetry Service for CTIA
Provides dynamic system metrics for ML decision-making
"""
import random
from datetime import datetime, timedelta
from typing import Dict
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.models import Ticket, Redemption

class TelemetryService:
    """
    Provides real-time system telemetry for ticket probability decisions.
    In production, replace simulation methods with actual network monitoring.
    """
    
    def __init__(self, db: Session, simulation_mode: bool = True):
        self.db = db
        self.simulation_mode = simulation_mode
        self.start_time = datetime.utcnow()
    
    def get_all_metrics(self) -> Dict[str, float]:
        """Get all telemetry metrics at once"""
        return {
            "network_load": self.get_network_load(),
            "relay_reputation": self.get_relay_reputation(),
            "budget_utilization": self.get_budget_utilization(),
            "recent_win_rate": self.get_recent_win_rate()
        }
    
    def get_network_load(self) -> float:
        """
        Measure current network congestion (0.0 to 1.0)
        
        Production implementation would:
        - Query active relay connections
        - Measure bandwidth usage
        - Check request queue depth
        - Monitor latency metrics
        
        Returns:
            float: Network load between 0.0 (idle) and 1.0 (saturated)
        """
        if self.simulation_mode:
            return self._simulate_network_load()
        
        # Production code (example):
        try:
            active_tickets = self.db.query(Ticket)\
                .filter(Ticket.status == "issued")\
                .filter(Ticket.created_at > datetime.utcnow() - timedelta(minutes=5))\
                .count()
            
            # Normalize: assume 100 concurrent tickets = high load
            load = min(active_tickets / 100.0, 1.0)
            return round(load, 2)
        except Exception:
            return 0.3  # Default fallback
    
    def get_relay_reputation(self, relay_id: str = None) -> float:
        """
        Calculate relay/node trust score (0.0 to 1.0)
        
        Production implementation would:
        - Query relay's historical performance
        - Calculate success rate of past transactions
        - Consider uptime and reliability
        - Factor in user feedback/ratings
        
        Args:
            relay_id: Optional specific relay to check
            
        Returns:
            float: Reputation score between 0.0 (untrusted) and 1.0 (excellent)
        """
        if self.simulation_mode:
            return self._simulate_relay_reputation()
        
        # Production code (example):
        try:
            # Query successful vs total redemptions
            week_ago = datetime.utcnow() - timedelta(days=7)
            total = self.db.query(Redemption)\
                .filter(Redemption.redeemed_at > week_ago)\
                .count()
            
            if total == 0:
                return 0.5  # Neutral for new relays
            
            successful = self.db.query(Redemption)\
                .filter(Redemption.redeemed_at > week_ago)\
                .filter(Redemption.winner == True)\
                .count()
            
            # Calculate success rate
            reputation = successful / total
            return round(reputation, 2)
        except Exception:
            return 0.75  # Default fallback
    
    def get_budget_utilization(self) -> float:
        """
        Calculate proportion of daily budget spent (0.0 to 1.0)
        
        Production implementation would:
        - Query winning tickets redeemed today
        - Sum total payouts
        - Compare against daily budget limit
        - Factor in pending settlements
        
        Returns:
            float: Budget usage between 0.0 (none spent) and 1.0 (budget exhausted)
        """
        if self.simulation_mode:
            return self._simulate_budget_utilization()
        
        # Production code (example):
        try:
            DAILY_BUDGET = 10000  # $10,000 daily limit (configure as needed)
            
            today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Sum value of all winning tickets redeemed today
            spent = self.db.query(func.sum(Ticket.value))\
                .join(Redemption)\
                .filter(Ticket.created_at >= today_start)\
                .filter(Redemption.winner == True)\
                .scalar() or 0
            
            utilization = min(spent / DAILY_BUDGET, 1.0)
            return round(utilization, 2)
        except Exception:
            return 0.2  # Default fallback
    
    def get_recent_win_rate(self) -> float:
        """
        Calculate win rate from recent ticket redemptions (0.0 to 1.0)
        
        Production implementation would:
        - Query last N tickets (e.g., 100)
        - Count winners vs losers
        - Use rolling window
        - Weight by time/value
        
        Returns:
            float: Win rate between 0.0 (no winners) and 1.0 (all winners)
        """
        if self.simulation_mode:
            return self._simulate_win_rate()
        
        # Production code (example):
        try:
            WINDOW_SIZE = 100  # Last 100 tickets
            
            recent_redemptions = self.db.query(Redemption)\
                .order_by(Redemption.redeemed_at.desc())\
                .limit(WINDOW_SIZE)\
                .all()
            
            if not recent_redemptions:
                return 0.01  # Default 1% if no history
            
            winners = sum(1 for r in recent_redemptions if r.winner)
            win_rate = winners / len(recent_redemptions)
            
            return round(win_rate, 3)
        except Exception:
            return 0.01  # Default fallback
    
    # ========== SIMULATION METHODS (for development/demo) ==========
    
    def _simulate_network_load(self) -> float:
        """
        Simulate realistic network traffic patterns
        - Higher during business hours (9 AM - 5 PM)
        - Lower during night/early morning
        - Random fluctuations
        """
        current_hour = datetime.utcnow().hour
        
        # Base load by time of day
        if 9 <= current_hour <= 17:  # Business hours
            base_load = random.uniform(0.5, 0.9)
        elif 6 <= current_hour < 9 or 17 < current_hour <= 22:  # Peak edges
            base_load = random.uniform(0.3, 0.6)
        else:  # Night
            base_load = random.uniform(0.1, 0.4)
        
        # Add some randomness
        noise = random.uniform(-0.1, 0.1)
        load = max(0.0, min(1.0, base_load + noise))
        
        return round(load, 2)
    
    def _simulate_relay_reputation(self) -> float:
        """
        Simulate relay reputation scores
        - Most relays are good (0.7-0.95)
        - Some occasional poor performers
        """
        # 80% chance of good reputation
        if random.random() < 0.8:
            reputation = random.uniform(0.75, 0.95)
        else:
            reputation = random.uniform(0.5, 0.75)
        
        return round(reputation, 2)
    
    def _simulate_budget_utilization(self) -> float:
        """
        Simulate budget usage throughout the day
        - Starts low in morning
        - Increases throughout day
        - Resets at midnight
        """
        elapsed_seconds = (datetime.utcnow() - self.start_time).total_seconds()
        
        # Simulate daily cycle (24 hours)
        daily_progress = (elapsed_seconds % 86400) / 86400
        
        # Add growth curve (spending accelerates during day)
        base_utilization = daily_progress ** 1.5
        
        # Add randomness
        noise = random.uniform(-0.05, 0.15)
        utilization = max(0.0, min(1.0, base_utilization + noise))
        
        return round(utilization, 2)
    
    def _simulate_win_rate(self) -> float:
        """
        Simulate recent win rate with realistic feedback
        - Generally stays around expected probability
        - Has variance and clusters
        - Adjusts based on "system state"
        """
        # Base win rate around 1-2%
        base_rate = random.uniform(0.005, 0.025)
        
        # Add clustering effect (winners come in streaks)
        if random.random() < 0.2:  # 20% chance of streak
            base_rate *= random.uniform(1.5, 2.5)
        
        win_rate = min(base_rate, 0.5)  # Cap at 50%
        return round(win_rate, 3)
    
    def get_telemetry_summary(self) -> str:
        """Get human-readable telemetry summary"""
        metrics = self.get_all_metrics()
        
        return f"""
Telemetry Snapshot:
  Network Load: {metrics['network_load']:.1%} {'🔴 HIGH' if metrics['network_load'] > 0.7 else '🟢 NORMAL'}
  Relay Reputation: {metrics['relay_reputation']:.1%} {'⭐ EXCELLENT' if metrics['relay_reputation'] > 0.85 else '✓ GOOD'}
  Budget Used: {metrics['budget_utilization']:.1%} {'⚠️  CRITICAL' if metrics['budget_utilization'] > 0.8 else '✓ OK'}
  Recent Winners: {metrics['recent_win_rate']:.1%} {'📈 HIGH' if metrics['recent_win_rate'] > 0.02 else '📉 LOW'}
        """.strip()


# Singleton instance (optional, for caching)
_telemetry_instance = None

def get_telemetry(db: Session) -> TelemetryService:
    """Factory function to get telemetry service"""
    return TelemetryService(db, simulation_mode=True)  # Set to False for production