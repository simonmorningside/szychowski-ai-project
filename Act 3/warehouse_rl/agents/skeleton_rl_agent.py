#!/usr/bin/env python3
"""
Skeleton Optimization Agent - Template for students to implement optimization algorithms

IMPORTANT: This is NOT an RL agent - it's a template for 
students to implement their own optimization algorithms (greedy, Hungarian, etc.)

This agent currently makes terrible decisions:
- Random staffing decisions
- No layout optimization 
- Ignores order priorities
- No intelligent assignment logic

Students should replace the naive methods with proper optimization algorithms.
"""

from matplotlib.pyplot import grid
import numpy as np
from typing import Dict, Optional
from .standardized_agents import BaselineAgent

class SkeletonOptimizationAgent(BaselineAgent):
    """
    Template for students to implement their own optimization algorithms.
    
    Students should replace the naive methods with:
    - Economic models for staffing decisions
    - Hungarian algorithm for order assignment  
    - Greedy search for layout optimization
    - Multi-objective optimization techniques
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.name = "StudentOptimization"

        # Enable unlimited hiring for true economic optimization
        self.env._unlimited_hiring = True
        
        # Students can add their own state tracking here
        self.performance_metrics = []
        self.decision_history = []
        
        # Students can add algorithm parameters here
        # Example: Hungarian algorithm matrices, greedy search parameters, etc.
        self.staffing_parameters = {
            'hire_threshold_ratio': 3.0,
            'fire_threshold_ratio': 2.0,
            'profit_threshold': 0
        }
        
        # Students can implement adaptive parameters that change based on performance
        self.adaptive_optimization_enabled = False
        
    def reset(self):
        """Reset agent state - students should expand this"""
        self.action_history = []
        self.reward_history = []
        # TODO: Reset any neural network states, replay buffers, etc.
    
    def get_action(self, observation: Dict) -> Dict:
        """
        Generate action based on observation with performance tracking.
        Suppresses per-step debug prints unless a layout swap occurs.
        """

        current_timestep = observation['time'][0]
        financial_state = observation['financial']
        queue_info = observation['order_queue']
        employee_info = observation['employees']

        layout_action = self._get_naive_layout_action(current_timestep)

        action = {
            'staffing_action': self._get_naive_staffing_action(financial_state, employee_info),
            'layout_swap': layout_action,
            'order_assignments': self._get_naive_order_assignments(queue_info, employee_info)
        }

        # Record action history
        if not hasattr(self, 'action_history'):
            self.action_history = []
        self.action_history.append(action.copy())

        # Optional debug: only print if a swap happened
        if layout_action != [0, 0]:
            print(f"[Action] Timestep {current_timestep}, layout swap executed: {layout_action}")

        # Track performance every 100 timesteps
        if current_timestep % 100 == 0:
            self.track_layout_performance()

        return action
    
    def _get_naive_staffing_action(self, financial_state, employee_info) -> int:
        """
        WEEK 2 STEP 1: Staffing decisions - students should improve this!
        
        Current problems:
        - Ignores queue length and workload
        - Random decisions regardless of profit
        - No consideration of employee efficiency
        """
        
        # TODO WEEK 2 STEP 1: Students should implement intelligent staffing logic
        # Current approach: Make random decisions based on "vibes"
        
        current_profit = financial_state[0]
        num_employees = np.sum(employee_info[:, 0] > 0)  # Count active employees
        
        # Terrible logic: Random decisions with slight bias
        # BUT: Hire managers more frequently so layout optimization can work
        has_manager = np.any(employee_info[:, 5] == 1)  # Check if we have a manager
        
        if np.random.random() < 0.3:  # 30% chance to hire
            if num_employees < 20:  # Don't go completely overboard
                return 1  # Hire worker
        elif np.random.random() < 0.1:  # 10% chance to fire
            if num_employees > 1:  # Don't fire everyone
                return 2  # Fire worker
        elif np.random.random() < 0.2:  # 20% chance to hire manager (increased from 5%)
            if not has_manager or num_employees > 10:  # Hire manager if we don't have one, or if we have lots of workers
                return 3  # Hire manager
        
        return 0  # No action
    
    def _get_naive_layout_action(self, current_timestep):

        if current_timestep % 100 != 0:
            return [0, 0]

        grid = self.env.warehouse_grid
        width = grid.width

        frequencies = np.array(grid.item_access_frequency)
        active_indices = np.where(frequencies > 0)[0]

        if len(active_indices) == 0:
            return [0, 0]

        # =========================================================
        # 1️⃣ PRIORITY: Co-occurrence clustering
        # =========================================================
        co_swap = self._find_cooccurrence_swap()
        if co_swap:
            print("[Smart Greedy] Executing co-occurrence swap")
            return co_swap
        


        swap = self._find_hot_cold_swap()
        if swap:
            print("[Smart Greedy] Executing global hot-cold swap")
            return swap
    def _find_hot_cold_swap(self):
        """
        Improved hot-cold swap logic using global efficiency gain.
        
        Selects swaps that maximize:
            gain = (freq_hot - freq_cold) * (dist_cold - dist_hot)

        Only executes swap if gain > threshold.
        """
        grid = self.env.warehouse_grid
        width = grid.width
        delivery_positions = grid.truck_bay_positions

        frequencies = np.array(grid.item_access_frequency)
        active_items = np.where(frequencies > 0)[0]

        if len(active_items) < 2:
            return None

        item_data = []

        # Collect (item, freq, location, distance_to_bay)
        for item in active_items:
            locs = grid.find_item_locations(item)
            if not locs:
                continue

            pos = locs[0]
            dist = min(
                grid.manhattan_distance(pos, bay)
                for bay in delivery_positions
            )

            item_data.append((item, frequencies[item], pos, dist))

        if len(item_data) < 2:
            return None

        best_gain = 0
        best_swap = None

        # Evaluate all pairs
        for i in range(len(item_data)):
            for j in range(i + 1, len(item_data)):

                item1, freq1, pos1, dist1 = item_data[i]
                item2, freq2, pos2, dist2 = item_data[j]

                # Only consider meaningful hot-cold contrast
                if freq1 == freq2:
                    continue

                # Define hot and cold
                if freq1 > freq2:
                    hot_item, cold_item = item1, item2
                    hot_freq, cold_freq = freq1, freq2
                    hot_pos, cold_pos = pos1, pos2
                    hot_dist, cold_dist = dist1, dist2
                else:
                    hot_item, cold_item = item2, item1
                    hot_freq, cold_freq = freq2, freq1
                    hot_pos, cold_pos = pos2, pos1
                    hot_dist, cold_dist = dist2, dist1

                # Only swap if cold is closer to bay than hot
                if cold_dist >= hot_dist:
                    continue

                # Calculate weighted gain
                gain = (hot_freq - cold_freq) * (hot_dist - cold_dist)

                if gain > best_gain:
                    best_gain = gain
                    best_swap = (hot_pos, cold_pos)

        # Threshold prevents micro-optimizations
        MIN_GAIN_THRESHOLD = 5

        if best_gain > MIN_GAIN_THRESHOLD and best_swap:
            hot_pos, cold_pos = best_swap
            print(f"[Hot-Cold] Gain={best_gain:.2f}")
            return [
                hot_pos[1] * width + hot_pos[0],
                cold_pos[1] * width + cold_pos[0]
            ]

        return None
    
    def _find_cooccurrence_swap(self):
        """
        Greedy clustering algorithm for association-based spatial optimization.

        Scans all item pairs, calculates benefit (frequency × distance),
        and returns the adjacency swap with highest benefit.
        """
        grid = self.env.warehouse_grid
        cooccurrence = grid.item_cooccurrence  # co-occurrence matrix
        width, height = grid.width, grid.height

        min_cooccurrence = 3
        min_distance = 4
        best_benefit = 0
        best_swap = None

        num_items = cooccurrence.shape[0]

        # Scan all unique item pairs (item1 < item2)
        for item1 in range(num_items):
            for item2 in range(item1 + 1, num_items):
                freq = cooccurrence[item1, item2]

                # Skip pairs with low co-occurrence
                if freq < min_cooccurrence:
                    continue

                # Get all locations of both items
                locs1 = grid.find_item_locations(item1)
                locs2 = grid.find_item_locations(item2)
                if not locs1 or not locs2:
                    continue

                # Compute minimum distance between any pair of locations
                current_dist = min(
                    grid.manhattan_distance(l1, l2) for l1 in locs1 for l2 in locs2
                )

                # Skip pairs that are already close
                if current_dist < min_distance:
                    continue

                # Benefit function: frequency × distance
                benefit = freq * current_dist
                if benefit > best_benefit:
                    best_benefit = benefit
                    best_pair = (item1, item2)
                    best_pair_locs = (locs1[0], locs2[0])  # take first occurrence

        if best_benefit > 0:
            item1, item2 = best_pair
            loc1, loc2 = best_pair_locs

            # Find adjacency swap: move item2 next to item1
            target_positions = [
                (loc1[0] + dx, loc1[1] + dy)
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]
                if 0 <= loc1[0] + dx < width and 0 <= loc1[1] + dy < height
            ]

            for pos in target_positions:
                # Only swap into empty or accessible storage
                if grid.cell_types[pos[1], pos[0]] == 1 and pos in grid.empty_storage_positions:
                    print(f"[Phase 2] Clustering swap: Move item {item2} from {loc2} to {pos} next to item {item1}")
                    return [loc2[1]*width + loc2[0], pos[1]*width + pos[0]]

        print("[Phase 2] No beneficial co-occurrence swap found.")
        return None   
    
    def _find_closer_position(self, current_pos, delivery_positions):
        """
        Greedy neighborhood search for better item placement.

        Returns [current_index, target_index] if beneficial swap found, else None.
        """
        grid = self.env.warehouse_grid
        width, height = grid.width, grid.height

        # Current distance to nearest delivery bay
        current_dist = min(grid.manhattan_distance(current_pos, delivery_pos) 
                        for delivery_pos in delivery_positions)

        best_pos = None
        best_dist = current_dist

        # Loop through all grid positions
        for y in range(height):
            for x in range(width):
                candidate_pos = (x, y)

                # Only consider storage cells (1 = storage)
                if grid.cell_types[y, x] != 1:
                    continue

                # Skip current position
                if candidate_pos == current_pos:
                    continue

                # Must be empty to move into (optional: you can also swap)
                if candidate_pos not in grid.empty_storage_positions:
                    continue

                # Distance from candidate to nearest delivery bay
                candidate_dist = min(grid.manhattan_distance(candidate_pos, delivery_pos) 
                                    for delivery_pos in delivery_positions)

                # Greedy selection: pick position that improves distance the most
                if candidate_dist < best_dist - 1:  # Minimum improvement >1 step
                    best_dist = candidate_dist
                    best_pos = candidate_pos

        if best_pos:
            current_index = current_pos[1] * width + current_pos[0]
            target_index = best_pos[1] * width + best_pos[0]
            return [current_index, target_index]

        return None  # No better position found
    
    def track_layout_performance(self):
        """
        Objective function evaluation for layout quality.
        
        Efficiency = 1.0 - (frequency-weighted average distance / max_possible_distance)
        Higher efficiency (closer to 1.0) = better layout
        """
        grid = self.env.warehouse_grid
        delivery_positions = grid.truck_bay_positions

        frequencies = np.array(grid.item_access_frequency)
        item_ids = np.arange(len(frequencies))

        # Filter items that have been accessed
        active_indices = np.where(frequencies > 0)[0]
        if len(active_indices) == 0:
            print("[Efficiency] No active items, returning neutral score 0.5")
            return 0.5  # Neutral efficiency if no active items

        active_frequencies = frequencies[active_indices]
        active_item_ids = item_ids[active_indices]

        weighted_distances = []
        total_frequency = np.sum(active_frequencies)

        # Calculate weighted distance for each active item
        for item_id, freq in zip(active_item_ids, active_frequencies):
            locations = grid.find_item_locations(item_id)
            if not locations:
                continue

            # Distance to nearest delivery bay
            min_dist = min(
                grid.manhattan_distance(loc, bay) for loc in locations for bay in delivery_positions
            )

            weighted_distances.append(freq * min_dist)
            print(f"[Efficiency] Item {item_id}, freq={freq}, min_dist={min_dist}, weighted={freq * min_dist}")

        # Weighted average distance
        weighted_avg_distance = sum(weighted_distances) / total_frequency

        # Maximum possible distance (diagonal across grid)
        max_distance = grid.width + grid.height - 2  # Manhattan distance across grid
        efficiency = 1.0 - (weighted_avg_distance / max_distance)
        efficiency = max(0.0, min(1.0, efficiency))  # Clamp 0-1

        print(f"[Efficiency] Weighted avg distance: {weighted_avg_distance:.2f}, Efficiency: {efficiency:.3f}")
        return efficiency
                
    def _get_naive_order_assignments(self, queue_info, employee_info) -> list:
        """
        WEEK 2 STEP 2: Order assignment - students should improve this!
        
        Current problems:
        - Ignores employee locations
        - No consideration of order priority/value
        - Random assignments regardless of efficiency
        - Doesn't check if employees are actually available
        """
        
        # TODO WEEK 2 STEP 2: Students should implement intelligent order assignment
        # Current approach: Random assignments
        
        assignments = [0] * 20  # No assignments by default
        
        # Count how many employees we have (very naive)
        num_employees = int(np.sum(employee_info[:, 0] > 0))
        
        # Randomly assign first few orders to first few employees
        if num_employees > 0:
            for i in range(min(3, num_employees)):  # Only assign 3 orders max
                if np.random.random() < 0.6:  # 60% chance to assign
                    assignments[i] = (i % num_employees) + 1  # Random employee
        
        return assignments
    
    def record_reward(self, reward: float):
        """
        WEEK 3 STEP 1: Reward tracking and learning - students should expand this
        
        TODO WEEK 3 STEP 1: Students should implement:
        - Proper reward tracking
        - Performance analysis
        - Adaptive parameter adjustment
        - Multi-objective optimization
        """
        self.reward_history.append(reward)
        
        # Skeleton optimization - doesn't actually learn anything useful
        if len(self.reward_history) > 10:
            # "Update" weights randomly (this doesn't actually improve performance)
            if reward > 0:
                self.staffing_weights += np.random.randn(4) * 0.01
                self.layout_weights += np.random.randn(3) * 0.01
    
    def should_update_policy(self) -> bool:
        """
        WEEK 3 STEP 2: Policy updates and adaptation - students should improve this
        
        TODO WEEK 3 STEP 2: Students should implement proper update schedules
        """
        return len(self.action_history) % 50 == 0  # Arbitrary update frequency
    
    def get_performance_metrics(self) -> Dict:
        """
        Get agent performance metrics for analysis
        Students can use this to debug their improvements
        """
        if not self.reward_history:
            return {"avg_reward": 0, "total_actions": 0}
        
        return {
            "avg_reward": np.mean(self.reward_history[-100:]),  # Last 100 rewards
            "total_actions": len(self.action_history),
            "exploration_rate": self.exploration_rate,
            "recent_performance": np.mean(self.reward_history[-10:]) if len(self.reward_history) >= 10 else 0
        }


def create_skeleton_optimization_agent(env) -> SkeletonOptimizationAgent:
    """Factory function to create skeleton Optimization agent"""
    return SkeletonOptimizationAgent(env)


# TODO: Students should implement these advanced components:

class StudentOptimizationAgent(SkeletonOptimizationAgent):
    """
    Template for students to implement their improved Optimization agent
    
    Suggested improvements:
    1. Replace random staffing with demand-based hiring
    2. Implement proper layout optimization using item frequencies
    3. Add distance-based order assignment
    4. Implement basic Q-optimization or policy gradients
    5. Add proper exploration vs exploitation balance
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.name = "StudentOptimization"
        
        # TODO: Students implement these
        # self.q_table = {}
        # self.policy_network = SimpleNeuralNetwork()
        # self.experience_buffer = []
        # self.target_network = None
        
    def _get_improved_staffing_action(self, financial_state, employee_info, queue_info):
        """
        WEEK 2 STEP 1: Students implement intelligent staffing:
        - Hire when queue is growing
        - Fire when queue is empty for extended periods
        - Consider profit margins before hiring
        - Balance managers vs workers
        """
        pass
    
    def _get_improved_layout_action(self, observation):
        """
        WEEK 1 STEP 1: Students implement smart layout optimization:
        - Move frequently accessed items closer to delivery
        - Group items that are often ordered together
        - Only optimize when queue is manageable
        - Track swap effectiveness
        """
        pass
    
    def _get_improved_order_assignments(self, queue_info, employee_info):
        """
        WEEK 2 STEP 2: Students implement efficient order assignment:
        - Assign orders to closest available employees
        - Prioritize high-value or urgent orders
        - Consider employee current locations
        - Balance workload across employees
        """
        pass
    
    def learn_from_experience(self, state, action, reward, next_state, done):
        """
        WEEK 3 STEP 3: Students implement multi-objective optimization:
        - Performance trend analysis
        - Adaptive parameter tuning
        - Multi-objective trade-off handling
        - Robust optimization techniques
        """
        pass