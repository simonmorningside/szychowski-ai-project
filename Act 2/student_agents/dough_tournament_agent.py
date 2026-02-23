#!/usr/bin/env python
"""
Tournament Agent: King kong
Student: Simon Zychowski
Generated: 2026-02-12 17:40:17

Evolution Details:
- Generations: 100
- Final Fitness: N/A
- Trained against: Grim Trigger, Random (0.3), Always Invest, Tit-for-Two-Tats, Random (0.5)...

Strategy: climb to the top of the empire state building
"""

from agents import Agent, INVEST, UNDERCUT
import random


class DougDahlAgent(Agent):
    """
    King kong

    climb to the top of the empire state building

    Evolved Genes: [1.0, 0.0, 0.9562391288988988, 0.0, 0.5658161479289543, 0.7625331808588661, 0.4507713495605262, 0.37186986151263907]
    """

    def __init__(self):
        # These genes were evolved through 100 generations
        self.genes = [1.0, 0.0, 0.9562391288988988, 0.0, 0.5658161479289543, 0.7625331808588661, 0.4507713495605262, 0.37186986151263907]

        # Required for tournament compatibility
        self.student_name = "Doug Dahl"

        super().__init__(
            name="King kong",
            description="climb to the top of the empire state building"
        )

    def choose_action(self) -> bool:

    # --- Early cooperation phase ---
        if self.round_num < 3:
            return random.random() < (0.8 * self.genes[0] + 0.2)

        # --- Memory window ---
        memory_length = max(2, int(self.genes[4] * 8) + 2)
        recent_history = self.history[-memory_length:]
        cooperation_rate = sum(recent_history) / len(recent_history)

        last_opponent_move = self.history[-1]

        # --- Immediate retaliation ---
        if last_opponent_move == UNDERCUT:
            # Retaliate strongly
            if random.random() < (0.7 + 0.3 * self.genes[2]):
                return UNDERCUT
            # Occasional forgiveness
            return INVEST

        # --- Highly cooperative opponent (exploit zone) ---
        if cooperation_rate > 0.85:
            # Small exploitation chance
            if random.random() < self.genes[3] * 0.25:
                return UNDERCUT
            return INVEST

        # --- Moderately cooperative opponent ---
        if cooperation_rate >= 0.5:
            return random.random() < self.genes[1]

        # --- Mostly defecting opponent ---
        return UNDERCUT

        return move



# Convenience function for tournament loading
def get_agent():
    """Return an instance of this agent for tournament use"""
    return DougDahlAgent()


if __name__ == "__main__":
    # Test that the agent can be instantiated
    agent = get_agent()
    print(f"✅ Agent loaded successfully: {agent.name}")
    print(f"   Genes: {agent.genes}")
    print(f"   Description: {agent.description}")
