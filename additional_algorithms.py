import numpy as np
import pandas as pd
from collections import defaultdict
from cbwo_metrics import CBWOMetricsCalculator


class RoundRobinLoadBalancer:
    """Round Robin Load Balancer - Cycles through VMs in order"""

    def __init__(self, df):
        self.df = df.copy()
        self.vm_names = df['vm_name'].unique()
        self.n_vms = len(self.vm_names)
        self.current_index = 0
        self.simulation_results = []

    def select_vm(self):
        """Select next VM in round-robin fashion"""
        selected_vm = self.vm_names[self.current_index]
        self.current_index = (self.current_index + 1) % self.n_vms
        return selected_vm

    def simulate_load_balancing(self, start_time=None, end_time=None):
        """Simulate load balancing over a time period"""
        if start_time is None:
            start_time = self.df['timestamp'].min()
        if end_time is None:
            end_time = self.df['timestamp'].max()

        # Reset state
        self.current_index = 0
        self.simulation_results = []

        # Get request timestamps
        requests = self.df[
            (self.df['timestamp'] >= start_time) &
            (self.df['timestamp'] <= end_time) &
            (self.df['update'] == 1)
        ].copy()

        if len(requests) == 0:
            return []

        results = []

        for idx, row in requests.iterrows():
            timestamp = row['timestamp']
            selected_vm = self.select_vm()

            # Get actual VM state at this time
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
        
        # Use CBWO metrics calculator with Round Robin algorithm type
        metrics_calc = CBWOMetricsCalculator(self.df, self.simulation_results)
        metrics = metrics_calc.get_all_metrics(power_coefficient, algorithm_type="round_robin")
        
        # Add VM distribution for visualization
        vm_distribution = defaultdict(int)
        for req in self.simulation_results:
            vm_distribution[req['selected_vm']] += 1
        
        metrics['vm_distribution'] = dict(vm_distribution)
        metrics['total_requests'] = len(self.simulation_results)
        
        return metrics


class ChaosBlackWidowLoadBalancer:
    """Chaos-based Black Widow Optimization (BWO) inspired load balancer.

    This is a lightweight, heuristic implementation suitable for selecting a VM
    at each request timestamp. It uses a simple chaotic map (logistic map)
    to introduce diversity and a small generational loop inspired by BWO
    (mating and selection). For small action spaces (few VMs) this behaves
    similarly to selecting the VM with the highest reward but includes
    controllable stochasticity via chaos_factor.
    """

    def __init__(self, df, pop_size=10, generations=5, chaos_factor=0.2, seed=None):
        self.df = df.copy()
        self.vm_names = df['vm_name'].unique()
        self.n_vms = len(self.vm_names)
        self.pop_size = max(2, int(pop_size))
        self.generations = max(1, int(generations))
        self.chaos_factor = float(chaos_factor)
        self.simulation_results = []

        if seed is not None:
            np.random.seed(seed)

    def _logistic_map(self, x, mu=4.0):
        return mu * x * (1 - x)

    def _initial_population(self):
        # Initialize population using chaotic logistic map
        pop = []
        x = np.random.rand()
        for _ in range(self.pop_size):
            x = self._logistic_map(x)
            idx = int((x % 1.0) * self.n_vms)
            pop.append(idx)
        return pop

    def _calculate_reward_from_state(self, vm_state):
        # Reuse same reward formula as other balancers
        cpu_reward = (1.0 - vm_state['cpu_usage']) * 0.3
        mem_usage_norm = vm_state['mem_usage'] / vm_state['max_mem'] if vm_state['max_mem'] > 0 else 0
        mem_reward = (1.0 - mem_usage_norm) * 0.2
        bw_usage_norm = vm_state['bw_usage'] / vm_state['max_bw'] if vm_state['max_bw'] > 0 else 0
        bw_reward = (1.0 - bw_usage_norm) * 0.2
        priority_reward = (1.0 - vm_state['priority'] / 4.0) * 0.2
        return cpu_reward + mem_reward + bw_reward + priority_reward

    def _evaluate_vm(self, vm_name, time_data):
        # Use the latest state for this VM
        vm_data = time_data[time_data['vm_name'] == vm_name].tail(1)
        if len(vm_data) == 0:
            return -1.0
        vm_state = vm_data.iloc[0]
        return self._calculate_reward_from_state(vm_state)

    def _optimize_choice(self, time_data):
        # Very small generational loop: start with chaotic population of VM indices
        population = self._initial_population()

        for gen in range(self.generations):
            # Evaluate population
            fitness = [self._evaluate_vm(self.vm_names[i], time_data) for i in population]

            # Introduce chaotic perturbation to fitness to increase exploration
            chaos_seq = np.random.rand(len(fitness)) if self.chaos_factor > 0 else np.zeros(len(fitness))
            fitness = [f + self.chaos_factor * (s - 0.5) for f, s in zip(fitness, chaos_seq)]

            # Select top half as parents
            parents_idx = np.argsort(fitness)[-max(1, len(fitness)//2):]
            parents = [population[i] for i in parents_idx]

            # Create offspring by randomly sampling parents and applying small mutation (chaos)
            offspring = []
            while len(offspring) < self.pop_size - len(parents):
                a, b = np.random.choice(parents, 2, replace=True)
                # crossover-like choice: randomly pick one parent's index
                child = a if np.random.rand() < 0.5 else b
                # mutation: small chance to switch to a nearby vm index
                if np.random.rand() < 0.2:
                    child = (child + np.random.randint(-1, 2)) % self.n_vms
                offspring.append(int(child))

            # New population = parents + offspring
            population = list(parents) + offspring

            # If population shrinks/rare, refill with chaotic picks
            while len(population) < self.pop_size:
                x = np.random.rand()
                idx = int(self._logistic_map(x) * self.n_vms) % self.n_vms
                population.append(idx)

        # Final evaluation and pick best
        final_fitness = [self._evaluate_vm(self.vm_names[i], time_data) for i in population]
        best_idx = int(np.argmax(final_fitness))
        return self.vm_names[population[best_idx]]

    def simulate_load_balancing(self, start_time=None, end_time=None):
        if start_time is None:
            start_time = self.df['timestamp'].min()
        if end_time is None:
            end_time = self.df['timestamp'].max()

        self.simulation_results = []

        requests = self.df[
            (self.df['timestamp'] >= start_time) &
            (self.df['timestamp'] <= end_time) &
            (self.df['update'] == 1)
        ].copy()

        if len(requests) == 0:
            return []

        results = []

        for idx, row in requests.iterrows():
            timestamp = row['timestamp']

            # Use recent time window data
            time_data = self.df[(self.df['timestamp'] <= timestamp)].tail(1000)
            if len(time_data) == 0:
                continue

            selected_vm = self._optimize_choice(time_data)

            vm_data = time_data[time_data['vm_name'] == selected_vm].tail(1)
            if len(vm_data) > 0:
                vm_state = vm_data.iloc[0]
                reward = self._calculate_reward_from_state(vm_state)
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

    def get_performance_metrics(self):
        if not self.simulation_results:
            return {}

        results_df = pd.DataFrame(self.simulation_results)

        vm_distribution = defaultdict(int)
        for req in self.simulation_results:
            vm_distribution[req['selected_vm']] += 1

        return {
            'total_requests': len(self.simulation_results),
            'avg_reward': results_df['reward'].mean(),
            'std_reward': results_df['reward'].std(),
            'avg_cpu_usage': results_df['cpu_usage'].mean(),
            'avg_mem_usage': results_df['mem_usage'].mean(),
            'avg_bw_usage': results_df['bw_usage'].mean(),
            'avg_score': results_df['score'].mean(),
            'vm_distribution': dict(vm_distribution),
            'load_balance_index': self._calculate_load_balance_index(results_df)
        }

    def _calculate_load_balance_index(self, results_df):
        """Calculate load balance index similar to RoundRobin implementation."""
        if results_df is None or len(results_df) == 0:
            return 0.0

        # group by selected_vm and compute mean usage
        try:
            vm_cpu_means = results_df.groupby('selected_vm')['cpu_usage'].mean()
            vm_mem_means = results_df.groupby('selected_vm')['mem_usage'].mean()
            cpu_variance = vm_cpu_means.var() if len(vm_cpu_means) > 0 else 0.0
            mem_variance = vm_mem_means.var() if len(vm_mem_means) > 0 else 0.0
            return float((cpu_variance + mem_variance) / 2.0)
        except Exception:
            return 0.0




