import numpy as np
import pandas as pd
from collections import defaultdict
import random
from cbwo_metrics import CBWOMetricsCalculator

class CBWOLoadBalancer:
    """
    Chaos-Based Black Widow Optimization (CBWO) Load Balancer
    
    CBWO is a metaheuristic optimization algorithm inspired by the black widow spider's
    mating behavior. It uses chaos theory for better exploration and exploitation balance.
    """
    
    def __init__(self, df):
        self.df = df.copy()
        self.vm_names = df['vm_name'].unique()
        self.n_vms = len(self.vm_names)
        self.simulation_results = []
        
        # CBWO parameters
        self.population_size = 10
        self.max_iterations = 50
        self.procreate_rate = 0.6
        self.cannibalism_rate = 0.44
        self.mutation_rate = 0.4
        self.chaos_constant = 4.0
        
        # Initialize population (black widows)
        self.population = self._initialize_population()
    
    def _initialize_population(self):
        """Initialize population of black widows (solutions)"""
        population = []
        for _ in range(self.population_size):
            # Each black widow is a weight vector for VMs
            weights = np.random.uniform(0.1, 1.0, self.n_vms)
            weights = weights / weights.sum()  # Normalize
            population.append({
                'weights': weights,
                'fitness': 0.0
            })
        return population
    
    def _chaos_map(self, x):
        """Logistic chaos map for chaotic behavior"""
        return self.chaos_constant * x * (1 - x)
    
    def _calculate_fitness(self, weights, time_data):
        """Calculate fitness of a solution based on load balancing objectives"""
        if len(time_data) == 0:
            return 0.0
        
        fitness = 0.0
        vm_metrics = {}
        
        for idx, vm_name in enumerate(self.vm_names):
            vm_data = time_data[time_data['vm_name'] == vm_name].tail(1)
            if len(vm_data) > 0:
                latest = vm_data.iloc[0]
                cpu_usage = latest['cpu_usage']
                mem_usage_norm = latest['mem_usage'] / latest['max_mem'] if latest['max_mem'] > 0 else 0
                bw_usage_norm = latest['bw_usage'] / latest['max_bw'] if latest['max_bw'] > 0 else 0
                priority_norm = 1.0 - (latest['priority'] / 4.0)
                
                # Combined resource usage (lower is better)
                resource_usage = (cpu_usage + mem_usage_norm + bw_usage_norm) / 3.0
                
                # Fitness considers weight, resource availability, and priority
                vm_fitness = weights[idx] * (1.0 - resource_usage) * priority_norm
                vm_metrics[vm_name] = {
                    'usage': resource_usage,
                    'fitness': vm_fitness
                }
                fitness += vm_fitness
        
        # Add load balance component (penalize uneven distribution)
        if len(vm_metrics) > 0:
            usages = [vm['usage'] for vm in vm_metrics.values()]
            variance = np.var(usages)
            fitness -= variance * 0.5  # Penalty for imbalance
        
        return fitness / self.n_vms if self.n_vms > 0 else 0.0
    
    def _select_vm_cbwo(self, weights, time_data):
        """Select VM using CBWO optimized weights"""
        # Calculate selection probabilities based on weights and current state
        probabilities = np.zeros(self.n_vms)
        
        for idx, vm_name in enumerate(self.vm_names):
            vm_data = time_data[time_data['vm_name'] == vm_name].tail(1)
            if len(vm_data) > 0:
                latest = vm_data.iloc[0]
                cpu_usage = latest['cpu_usage']
                mem_usage_norm = latest['mem_usage'] / latest['max_mem'] if latest['max_mem'] > 0 else 0
                resource_usage = (cpu_usage + mem_usage_norm) / 2.0
                
                # Higher weight and lower usage = higher probability
                probabilities[idx] = weights[idx] * (1.0 - resource_usage)
            else:
                probabilities[idx] = weights[idx] * 0.5
        
        # Normalize probabilities
        if probabilities.sum() > 0:
            probabilities = probabilities / probabilities.sum()
        else:
            probabilities = np.ones(self.n_vms) / self.n_vms
        
        # Select VM based on probabilities
        selected_idx = np.random.choice(self.n_vms, p=probabilities)
        return self.vm_names[selected_idx]
    
    def _optimize_weights(self, time_data):
        """Optimize weights using CBWO algorithm"""
        # Evaluate current population
        for widow in self.population:
            widow['fitness'] = self._calculate_fitness(widow['weights'], time_data)
        
        # Sort by fitness (descending)
        self.population.sort(key=lambda x: x['fitness'], reverse=True)
        
        # Main CBWO loop
        for iteration in range(self.max_iterations):
            # Procreate (crossover)
            new_population = []
            num_offspring = int(self.population_size * self.procreate_rate)
            
            for _ in range(num_offspring):
                # Select parents (better fitness = higher chance)
                parent1_idx = np.random.choice(min(5, len(self.population)))
                parent2_idx = np.random.choice(min(5, len(self.population)))
                
                parent1 = self.population[parent1_idx]['weights']
                parent2 = self.population[parent2_idx]['weights']
                
                # Crossover with chaos
                chaos_value = random.random()
                for _ in range(3):  # Apply chaos map
                    chaos_value = self._chaos_map(chaos_value)
                
                alpha = chaos_value
                offspring_weights = alpha * parent1 + (1 - alpha) * parent2
                offspring_weights = offspring_weights / offspring_weights.sum()
                
                # Mutation
                if random.random() < self.mutation_rate:
                    mutation_idx = np.random.randint(0, self.n_vms)
                    mutation_value = random.uniform(-0.2, 0.2)
                    offspring_weights[mutation_idx] = np.clip(
                        offspring_weights[mutation_idx] + mutation_value, 0.1, 1.0
                    )
                    offspring_weights = offspring_weights / offspring_weights.sum()
                
                new_population.append({
                    'weights': offspring_weights,
                    'fitness': self._calculate_fitness(offspring_weights, time_data)
                })
            
            # Cannibalism (remove worst solutions)
            self.population.extend(new_population)
            self.population.sort(key=lambda x: x['fitness'], reverse=True)
            
            num_cannibalize = int(len(self.population) * self.cannibalism_rate)
            self.population = self.population[:-num_cannibalize] if num_cannibalize > 0 else self.population
            
            # Maintain population size
            while len(self.population) < self.population_size:
                # Add random solution
                weights = np.random.uniform(0.1, 1.0, self.n_vms)
                weights = weights / weights.sum()
                self.population.append({
                    'weights': weights,
                    'fitness': self._calculate_fitness(weights, time_data)
                })
            
            self.population = self.population[:self.population_size]
        
        # Return best solution
        self.population.sort(key=lambda x: x['fitness'], reverse=True)
        return self.population[0]['weights']
    
    def simulate_load_balancing(self, start_time=None, end_time=None):
        """Simulate load balancing over a time period"""
        if start_time is None:
            start_time = self.df['timestamp'].min()
        if end_time is None:
            end_time = self.df['timestamp'].max()
        
        # Reset state
        self.simulation_results = []
        self.population = self._initialize_population()
        
        # Get request timestamps
        requests = self.df[
            (self.df['timestamp'] >= start_time) &
            (self.df['timestamp'] <= end_time) &
            (self.df['update'] == 1)
        ].copy()
        
        if len(requests) == 0:
            return []
        
        # Optimize weights periodically
        optimization_interval = max(10, len(requests) // 5)
        current_weights = np.ones(self.n_vms) / self.n_vms  # Start with equal weights
        
        results = []
        
        for idx, (_, row) in enumerate(requests.iterrows()):
            timestamp = row['timestamp']
            
            # Re-optimize weights periodically
            if idx % optimization_interval == 0:
                time_data = self.df[self.df['timestamp'] <= timestamp].tail(1000)
                current_weights = self._optimize_weights(time_data)
            
            # Get current VM states
            time_data = self.df[self.df['timestamp'] <= timestamp].tail(1000)
            selected_vm = self._select_vm_cbwo(current_weights, time_data)
            
            # Get actual VM state
            vm_data = self.df[
                (self.df['vm_name'] == selected_vm) &
                (self.df['timestamp'] <= timestamp)
            ].tail(1)
            
            if len(vm_data) > 0:
                vm_state = vm_data.iloc[0]
                reward = self._calculate_reward(vm_state)
            else:
                reward = 0.0
            
            results.append({
                'timestamp': timestamp,
                'selected_vm': selected_vm,
                'reward': reward,
                'cpu_usage': vm_state['cpu_usage'] if len(vm_data) > 0 else 0,
                'mem_usage': vm_state['mem_usage'] / vm_state['max_mem'] if len(vm_data) > 0 and vm_state['max_mem'] > 0 else 0,
                'bw_usage': vm_state['bw_usage'] / vm_state['max_bw'] if len(vm_data) > 0 and vm_state['max_bw'] > 0 else 0,
                'score': vm_state['score'] if len(vm_data) > 0 else 0,
                'priority': vm_state['priority'] if len(vm_data) > 0 else 0
            })
        
        self.simulation_results = results
        return results
    
    def _calculate_reward(self, vm_state):
        """Calculate reward for selecting a VM"""
        cpu_reward = (1.0 - vm_state['cpu_usage']) * 0.3
        mem_usage_norm = vm_state['mem_usage'] / vm_state['max_mem'] if vm_state['max_mem'] > 0 else 0
        mem_reward = (1.0 - mem_usage_norm) * 0.2
        bw_usage_norm = vm_state['bw_usage'] / vm_state['max_bw'] if vm_state['max_bw'] > 0 else 0
        bw_reward = (1.0 - bw_usage_norm) * 0.2
        priority_reward = (1.0 - vm_state['priority'] / 4.0) * 0.2
        return cpu_reward + mem_reward + bw_reward + priority_reward
    
    def get_performance_metrics(self, power_coefficient=50.0):
        """Calculate CBWO paper-compliant performance metrics"""
        if not self.simulation_results:
            return {}
        
        # Use CBWO metrics calculator with CBWO algorithm type
        metrics_calc = CBWOMetricsCalculator(self.df, self.simulation_results)
        metrics = metrics_calc.get_all_metrics(power_coefficient, algorithm_type="cbwo")
        
        # Add VM distribution for visualization
        vm_distribution = defaultdict(int)
        for req in self.simulation_results:
            vm_distribution[req['selected_vm']] += 1
        
        metrics['vm_distribution'] = dict(vm_distribution)
        metrics['total_requests'] = len(self.simulation_results)
        
        return metrics

