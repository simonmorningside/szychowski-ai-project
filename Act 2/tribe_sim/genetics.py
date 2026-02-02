import random
import math
from entities import Gatherer
from config import *

class GeneticAlgorithm:
    def __init__(self):
        self.generation = 1
        self.fitness_history = []
        self.trait_history = []  # Track trait averages over generations
    
    def create_initial_population(self):
        population = []
        for _ in range(INITIAL_POPULATION):
            gatherer = Gatherer()
            population.append(gatherer)
        return population
    
    def evaluate_fitness(self, population):
        fitness_scores = []
        for gatherer in population:
            fitness = gatherer.calculate_fitness()
            fitness_scores.append((gatherer, fitness))
        
        # Sort by fitness (highest first)
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        return fitness_scores
    
    def select_survivors(self, fitness_scores):
        # STUDENT ASSIGNMENT 3: Implement a better selection mechanism
        # Current version just takes top 50% - very simple!
        #
        # Available information:
        # - fitness_scores: list of (gatherer, fitness) tuples, sorted by fitness (best first)
        # - SURVIVAL_RATE: currently 0.05 (top 5% survive)
        # - len(fitness_scores): total population size
        #
        # Alternative selection strategies to consider:
        # 1. Tournament selection: pick random groups, take best from each
        # 2. Roulette wheel: probability proportional to fitness
        # 3. Rank-based: select based on rank, not raw fitness values
        # 4. Elite + random: guarantee best survive, then random selection
        # 5. Fitness-proportionate with scaling (linear/exponential)
        # 6. Hybrid approaches: combine multiple strategies
        #
        # Strategy hints:
        # - Pure elitism (current) can cause premature convergence
        # - Pure randomness loses good solutions
        # - Tournament selection often works well (simple + effective)
        # - Consider selection pressure: too high = less diversity, too low = slow evolution
        #
        # Remember: Selection determines which traits get passed to next generation!
        
        # Minimal version: just take top 50% of population
        group_size = 5
        shuffled_scores = fitness_scores[:]
        random.shuffle(shuffled_scores) 
        grouped_scores = [shuffled_scores[i:i + group_size] for i in range(0, len(shuffled_scores), group_size)]  # Top 50%
        top_picks = []
        for group in grouped_scores:
            top_picks.append(sorted(group, key=lambda x: x[1], reverse=True))  # Sort each group by fitness # Select the best from each group
        survivors = [gatherer for group in top_picks for gatherer, fitness in group[:len(group) // 2]]
        return survivors
    
    def crossover(self, parent1, parent2):
        child_genes = {}
        for gene_name in parent1.genes:
            # 50% chance to inherit from each parent
            if random.random() < 0.5:
                child_genes[gene_name] = parent1.genes[gene_name]
            else:
                child_genes[gene_name] = parent2.genes[gene_name]
        
        child = Gatherer(genes=child_genes)
        return child
    
    def mutate(self, gatherer):
        """
        Adaptive Gaussian mutation with cooperation protection.
        - Strong exploration early
        - Fine-tuning later
        - Prevents cooperation collapse
        """

        for gene_name, value in gatherer.genes.items():
            if random.random() < MUTATION_RATE:
                min_val, max_val = GENE_RANGES[gene_name]
                gene_range = max_val - min_val

                # Adaptive mutation scaling (decays over generations)
                generation_factor = max(0.1, 1.0 - self.generation / 100)

                # Base mutation strength
                sigma = gene_range * MUTATION_STRENGTH * generation_factor

                # Cooperation protection: mutate less aggressively
                if gene_name == "cooperation":
                    sigma *= 0.5

                # Gaussian mutation
                mutation_amount = random.gauss(0, sigma)

                # Apply mutation
                new_value = value + mutation_amount

                # Clamp to valid range
                gatherer.genes[gene_name] = max(
                    min_val,
                    min(max_val, new_value)
                )

    
    def create_next_generation(self, population):
        # Evaluate fitness
        fitness_scores = self.evaluate_fitness(population)
        
        # Record statistics
        if fitness_scores:
            best_fitness = fitness_scores[0][1]
            avg_fitness = sum(fitness for _, fitness in fitness_scores) / len(fitness_scores)
            self.fitness_history.append({
                'generation': self.generation,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness
            })
            
            # Record trait averages
            all_gatherers = [gatherer for gatherer, _ in fitness_scores]
            trait_averages = {
                'generation': self.generation,
                'avg_speed': sum(g.genes['speed'] for g in all_gatherers) / len(all_gatherers),
                'avg_caution': sum(g.genes['caution'] for g in all_gatherers) / len(all_gatherers),
                'avg_search_pattern': sum(g.genes['search_pattern'] for g in all_gatherers) / len(all_gatherers),
                'avg_efficiency': sum(g.genes['efficiency'] for g in all_gatherers) / len(all_gatherers),
                'avg_cooperation': sum(g.genes['cooperation'] for g in all_gatherers) / len(all_gatherers)
            }
            self.trait_history.append(trait_averages)
        
        # Select survivors
        survivors = self.select_survivors(fitness_scores)
        
        # Create new population
        new_population = []
        
        # Add survivors (reset their state)
        for survivor in survivors:
            new_gatherer = Gatherer(genes=survivor.genes)
            new_population.append(new_gatherer)
        
        # Create offspring to fill remaining slots
        offspring_count = INITIAL_POPULATION - len(survivors)
        for _ in range(offspring_count):
            parent1 = random.choice(survivors)
            parent2 = random.choice(survivors)
            child = self.crossover(parent1, parent2)
            self.mutate(child)
            new_population.append(child)
        
        self.generation += 1
        return new_population
    
    def get_population_stats(self, population):
        if not population:
            return {
                'alive_count': 0,
                'total_count': 0,
                'avg_fitness': 0,
                'best_fitness': 0,
                'avg_speed': 0,
                'avg_caution': 0,
                'avg_cooperation': 0
            }
        
        alive_gatherers = [g for g in population if g.alive]
        alive_count = len(alive_gatherers)
        total_count = len(population)
        
        if alive_gatherers:
            fitness_scores = [g.calculate_fitness() for g in alive_gatherers]
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            best_fitness = max(fitness_scores)
            avg_speed = sum(g.genes['speed'] for g in alive_gatherers) / len(alive_gatherers)
            avg_caution = sum(g.genes['caution'] for g in alive_gatherers) / len(alive_gatherers)
            avg_cooperation = sum(g.genes['cooperation'] for g in alive_gatherers) / len(alive_gatherers)
        else:
            # Check all gatherers if none alive
            fitness_scores = [g.calculate_fitness() for g in population]
            avg_fitness = sum(fitness_scores) / len(fitness_scores) if fitness_scores else 0
            best_fitness = max(fitness_scores) if fitness_scores else 0
            avg_speed = sum(g.genes['speed'] for g in population) / len(population)
            avg_caution = sum(g.genes['caution'] for g in population) / len(population)
            avg_cooperation = sum(g.genes['cooperation'] for g in population) / len(population)
        
        return {
            'alive_count': alive_count,
            'total_count': total_count,
            'avg_fitness': avg_fitness,
            'best_fitness': best_fitness,
            'avg_speed': avg_speed,
            'avg_caution': avg_caution,
            'avg_cooperation': avg_cooperation
        }
    
    def reset(self):
        self.generation = 1
        self.fitness_history = []
        self.trait_history = []