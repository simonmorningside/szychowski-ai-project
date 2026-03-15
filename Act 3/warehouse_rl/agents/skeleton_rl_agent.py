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
        self.env._unlimited_hiring = False     
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
        """Reset agent state between episodes/tests."""
        self.action_history = []
        self.reward_history = []
        self.performance_metrics = []
        self.decision_history = []
        self.profit_history = []
        self.integrated_metrics = []
    
    def get_action(self, observation: Dict) -> Dict:
        """
        Week 2: Integrated optimization across all decision areas
        """
        current_timestep = observation['time'][0]
        financial_state = observation['financial']
        queue_info = observation['order_queue']
        employee_info = observation['employees']

        action = {
            'staffing_action': self._get_naive_staffing_action(financial_state, employee_info),
            'layout_swap': self._get_naive_layout_action(current_timestep),
            'order_assignments': self._get_naive_order_assignments(queue_info, employee_info)
        }

        # Record action for optimization analysis
        if not hasattr(self, 'action_history'):
            self.action_history = []
        self.action_history.append(action.copy())

        # Optional debug
        if action['layout_swap'] != [0, 0]:
            print(f"[Action] Timestep {current_timestep}, layout swap executed: {action['layout_swap']}")

        # Track integrated performance every 100 steps
        if current_timestep % 100 == 0:
            self.track_integrated_performance()

        return action
    
    def track_integrated_performance(self):
        """
        Performance analysis for integrated optimization system.

        Measures:
        - Economic efficiency (profit per employee)
        - Assignment quality proxy (queue pressure / assignment usage)
        - Layout effectiveness
        - Overall system performance
        """
        if not hasattr(self, 'integrated_metrics'):
            self.integrated_metrics = []

        current_profit = self.env.cumulative_profit
        num_employees = len(self.env.employees)
        queue_length = len(self.env.order_queue.orders)

        # Economic efficiency
        profit_per_employee = current_profit / max(1, num_employees)
        queue_pressure = queue_length / max(1, num_employees)

        # Assignment quality proxy:
        # count how many non-zero order assignments were made in the most recent action
        assignment_count = 0
        staffing_action = 0
        layout_swap_made = 0

        if hasattr(self, 'action_history') and len(self.action_history) > 0:
            latest_action = self.action_history[-1]
            assignment_count = sum(1 for a in latest_action['order_assignments'] if a != 0)
            staffing_action = latest_action['staffing_action']
            layout_swap_made = 1 if latest_action['layout_swap'] != [0, 0] else 0

        # Layout efficiency from Week 1 logic
        layout_efficiency = self.track_layout_performance()

        self.integrated_metrics.append({
            'timestep': self.env.current_timestep,
            'profit_per_employee': profit_per_employee,
            'queue_pressure': queue_pressure,
            'assignment_count': assignment_count,
            'staffing_action': staffing_action,
            'layout_swap_made': layout_swap_made,
            'total_decisions': len([
                a for a in self.action_history
                if (
                    a['staffing_action'] != 0 or
                    a['layout_swap'] != [0, 0] or
                    any(x != 0 for x in a['order_assignments'])
                )
            ]),
            'layout_efficiency': layout_efficiency
        })

        # Print integrated progress every 1000 steps
        if self.env.current_timestep % 1000 == 0:
            recent_metrics = self.integrated_metrics[-10:]
            avg_profit_per_emp = np.mean([m['profit_per_employee'] for m in recent_metrics])
            avg_queue_pressure = np.mean([m['queue_pressure'] for m in recent_metrics])
            avg_layout_eff = np.mean([m['layout_efficiency'] for m in recent_metrics])
            avg_assignments = np.mean([m['assignment_count'] for m in recent_metrics])

            print(
                f"[Integrated Performance] "
                f"${avg_profit_per_emp:.2f}/employee, "
                f"Queue pressure: {avg_queue_pressure:.2f}, "
                f"Layout efficiency: {avg_layout_eff:.3f}, "
                f"Assignments: {avg_assignments:.2f}"
            )

    def _get_naive_staffing_action(self, financial_state, employee_info) -> int:
        current_profit = financial_state[0]
        revenue = financial_state[1]
        costs = financial_state[2]
        burn_rate = financial_state[3]
        
        num_employees = int(np.sum(employee_info[:, 0] > 0))
        queue_length = len(self.env.order_queue.orders)
        has_manager = np.any(employee_info[:, 5] == 1)

        queue_pressure = queue_length / max(1, num_employees)
        profit_per_employee = current_profit / max(1, num_employees)

        if not hasattr(self, "profit_history"):
            self.profit_history = []
        self.profit_history.append(current_profit)
        if len(self.profit_history) > 8:
            self.profit_history.pop(0)

        if len(self.profit_history) >= 4:
            midpoint = len(self.profit_history) // 2
            older_avg = np.mean(self.profit_history[:midpoint])
            recent_avg = np.mean(self.profit_history[midpoint:])
            profit_trend = recent_avg - older_avg
        else:
            profit_trend = 0.0

        hire_threshold = 2.0
        fire_threshold = 0.25
        min_staff = 2

        # Hard safety cap
        max_staff = 20

        # Do not hire aggressively once workforce gets large
        large_staff_threshold = 12

        # Main hire logic
        if num_employees < max_staff:
            # Small/medium workforce: hire based mostly on backlog
            if num_employees < large_staff_threshold:
                if queue_pressure > hire_threshold:
                    return 1

            # Large workforce: require backlog AND financial health
            else:
                if (
                    queue_pressure > 2.5
                    and current_profit > 0
                    and profit_per_employee > 0
                    and profit_trend >= 0
                ):
                    return 1

        # Conservative firing
        if (
            queue_pressure < fire_threshold
            and queue_length == 0
            and current_profit < 0
            and num_employees > min_staff
        ):
            return 2

        # Manager only when worth trying
        if (
            not has_manager
            and num_employees >= 8
            and num_employees <= 15
            and queue_length >= 8
            and current_profit > 0
            and profit_trend >= 0
        ):
            return 3

        return 0
    
    def _get_naive_layout_action(self, current_timestep):

        if current_timestep % 100 != 0:
            return [0, 0]

        grid = self.env.warehouse_grid
        frequencies = np.array(grid.item_access_frequency)
        active_indices = np.where(frequencies > 0)[0]

        if len(active_indices) == 0:
            return [0, 0]

        # 1. Highest priority: enforce hot items near the front
        priority_swap = self._find_frequency_priority_swap()
        if priority_swap:
            print("[Smart Greedy] Executing frequency-priority swap")
            return priority_swap

        # 2. Secondary: co-occurrence clustering
        co_swap = self._find_cooccurrence_swap()
        if co_swap:
            print("[Smart Greedy] Executing co-occurrence swap")
            return co_swap

        # 3. Fallback: generic hot-cold swap
        swap = self._find_hot_cold_swap()
        if swap:
            print("[Smart Greedy] Executing global hot-cold swap")
            return swap

        return [0, 0]
    
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
                
    def _calculate_order_distance(self, worker_pos, order):
        """
        Calculate minimum distance from worker to any item needed for this order.
        
        Algorithm: Find the closest required item location and return that distance.
        """
        grid = self.env.warehouse_grid
        min_distance = float('inf')
        
        # Assumes order.items contains item IDs needed for the order
        for item_id in order.items:
            item_locations = grid.find_item_locations(item_id)
            
            if not item_locations:
                continue
            
            for location in item_locations:
                distance = grid.manhattan_distance(worker_pos, location)
                if distance < min_distance:
                    min_distance = distance
        
        return min_distance if min_distance != float('inf') else 0


    def _get_idle_workers(self, employee_info):
        """
        Identify workers available for order assignment.
        
        Returns list of (worker_index, worker_position) for available workers.
        
        employee_info format:
        [x, y, state, has_order, items_collected, is_manager]
        """
        idle_workers = []
        
        for worker_idx, worker in enumerate(employee_info):
            x = int(worker[0])
            y = int(worker[1])
            state = int(worker[2])
            has_order = int(worker[3])
            is_manager = int(worker[5])
            
            # Treat valid coordinates as active/present in the warehouse
            is_active = x >= 0 and y >= 0
            is_idle = state == 0
            not_busy_with_order = has_order == 0
            not_manager = is_manager == 0
            
            if is_active and is_idle and not_busy_with_order and not_manager:
                idle_workers.append((worker_idx, (x, y)))
        
        return idle_workers

    def _get_storage_positions_by_bay_distance(self):
        """
        Return all occupied storage positions sorted by closeness to nearest truck bay.
        Closest positions come first.
        """
        grid = self.env.warehouse_grid
        delivery_positions = grid.truck_bay_positions
        positions = []

        for y in range(grid.height):
            for x in range(grid.width):
                if grid.cell_types[y, x] != 1:
                    continue

                pos = (x, y)
                dist = min(
                    grid.manhattan_distance(pos, bay)
                    for bay in delivery_positions
                )
                positions.append((pos, dist))

        positions.sort(key=lambda x: x[1])
        return [pos for pos, _ in positions]


    def _get_item_primary_positions(self):
        """
        Build a mapping from item_id -> one representative location.
        Assumes one primary useful location per item for swap planning.
        """
        grid = self.env.warehouse_grid
        frequencies = np.array(grid.item_access_frequency)

        item_positions = {}
        for item_id in np.where(frequencies > 0)[0]:
            locs = grid.find_item_locations(item_id)
            if locs:
                item_positions[item_id] = locs[0]

        return item_positions


    def _find_frequency_priority_swap(self):
        """
        Stronger global layout rule:
        make sure closest positions are occupied by hottest items.

        Returns a swap [from_idx, to_idx] if a beneficial swap exists.
        """
        grid = self.env.warehouse_grid
        width = grid.width
        frequencies = np.array(grid.item_access_frequency)

        # Active items sorted hottest -> coldest
        active_items = [item for item in np.where(frequencies > 0)[0]]
        if len(active_items) < 2:
            return None

        active_items.sort(key=lambda item: frequencies[item], reverse=True)

        # Closest storage positions sorted front -> back
        ranked_positions = self._get_storage_positions_by_bay_distance()

        # Current item locations
        item_positions = self._get_item_primary_positions()
        if len(item_positions) < 2:
            return None

        # Reverse mapping: position -> item
        position_to_item = {pos: item for item, pos in item_positions.items()}

        # Only consider positions that currently hold active items
        occupied_ranked_positions = [pos for pos in ranked_positions if pos in position_to_item]
        if len(occupied_ranked_positions) < 2:
            return None

        # Ideal assignment:
        # hottest item should be in closest occupied position, etc.
        ideal_pairs = list(zip(active_items[:len(occupied_ranked_positions)], occupied_ranked_positions))

        best_gain = 0
        best_swap = None

        for ideal_item, target_pos in ideal_pairs:
            current_pos = item_positions.get(ideal_item)
            if current_pos is None:
                continue

            # Already in the desired spot
            if current_pos == target_pos:
                continue

            occupying_item = position_to_item.get(target_pos)
            if occupying_item is None:
                continue

            hot_freq = frequencies[ideal_item]
            cold_freq = frequencies[occupying_item]

            # Only swap if hotter item is being blocked by colder one
            if hot_freq <= cold_freq:
                continue

            # Compute benefit based on frequency gap and positional improvement
            current_dist = min(
                grid.manhattan_distance(current_pos, bay)
                for bay in grid.truck_bay_positions
            )
            target_dist = min(
                grid.manhattan_distance(target_pos, bay)
                for bay in grid.truck_bay_positions
            )

            if target_dist >= current_dist:
                continue

            gain = (hot_freq - cold_freq) * (current_dist - target_dist)

            if gain > best_gain:
                best_gain = gain
                best_swap = (current_pos, target_pos)

        MIN_GAIN_THRESHOLD = 3

        if best_swap and best_gain > MIN_GAIN_THRESHOLD:
            from_pos, to_pos = best_swap
            print(f"[Priority Layout] Gain={best_gain:.2f} swapping {from_pos} <-> {to_pos}")
            return [
                from_pos[1] * width + from_pos[0],
                to_pos[1] * width + to_pos[0]
            ]

        return None
    
    def _get_naive_order_assignments(self, queue_info, employee_info) -> list:
        """
        WEEK 2 STEP 2: Worker-to-order matching optimization
        
        Greedy assignment using:
        - Distance Weight: 70%
        - Value Weight: 30%
        - Maximum Distance: grid width + grid height
        - Assignment Method: Greedy
        """
        
        assignments = [0] * 20
        
        # Get pending orders limited to action space size
        pending_orders = list(self.env.order_queue.orders[:20])
        if len(pending_orders) == 0:
            return assignments
        
        # Find available workers
        idle_workers = self._get_idle_workers(employee_info)
        if len(idle_workers) == 0:
            return assignments
        
        grid = self.env.warehouse_grid
        max_distance = grid.width + grid.height
        distance_weight = 0.7
        value_weight = 0.3
        
        def get_order_value(order):
            """
            Try common order value fields. Default to 1.0 if none found.
            """
            for attr in ["total_value", "value", "reward", "priority"]:
                if hasattr(order, attr):
                    try:
                        return float(getattr(order, attr))
                    except Exception:
                        pass
            return 1.0
        
        # Normalize order values
        order_values = [get_order_value(order) for order in pending_orders]
        max_order_value = max(order_values) if order_values else 1.0
        if max_order_value <= 0:
            max_order_value = 1.0
        
        # Calculate assignment scores for all worker-order pairs
        scored_pairs = []
        
        for order_idx, order in enumerate(pending_orders):
            order_value = get_order_value(order)
            value_score = order_value / max_order_value
            
            for worker_idx, worker_pos in idle_workers:
                min_dist = self._calculate_order_distance(worker_pos, order)
                
                # Distance score = 1 / (1 + min_distance)
                distance_score = 1.0 / (1.0 + min_dist)
                
                # Combined score
                combined_score = (
                    distance_weight * distance_score
                    + value_weight * value_score
                )
                
                scored_pairs.append((combined_score, order_idx, worker_idx))
        
        # Greedy assignment algorithm
        scored_pairs.sort(reverse=True, key=lambda x: x[0])
        
        assigned_workers = set()
        assigned_orders = set()
        
        for score, order_idx, worker_idx in scored_pairs:
            if order_idx in assigned_orders:
                continue
            if worker_idx in assigned_workers:
                continue
            
            # action format uses 1-indexed worker IDs
            assignments[order_idx] = worker_idx + 1
            assigned_orders.add(order_idx)
            assigned_workers.add(worker_idx)
            
            if len(assigned_workers) == len(idle_workers):
                break
        
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