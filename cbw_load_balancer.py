import numpy as np
import pandas as pd
from collections import defaultdict
import math

class CBWLoadBalancer:
    """
    Connection-Based Weighted (CBW) Load Balancer
    
    CBW distributes requests based on:
    1. Current connection count
    2. Server weights (CPU, Memory, Bandwidth)
    3. Response time/score
    4. Priority levels
    """
    
    def __init__(self, df):
        self.df = df.copy()
        self.vm_names = df['vm_name'].unique()
        self.n_vms = len(self.vm_names)
        
        # Initialize connection tracking
        self.connections = {vm: 0 for vm in self.vm_names}
        self.request_history = []
        self.simulation_results = []  # Store simulation results
        
        # Weight factors for different metrics
        self.weights = {
            'cpu': 0.4,
            'memory': 0.3,
            'bandwidth': 0.2,
            'priority': 0.1
        }
    
    def calculate_vm_weight(self, vm_data):
        """Calculate weight for a VM based on current state"""
        if len(vm_data) == 0:
            return 0.0
        
        latest = vm_data.iloc[-1]
        
        # Normalize metrics
        cpu_usage = latest['cpu_usage']
        mem_usage_norm = latest['mem_usage'] / latest['max_mem'] if latest['max_mem'] > 0 else 0
        bw_usage_norm = latest['bw_usage'] / latest['max_bw'] if latest['max_bw'] > 0 else 0
        priority_norm = 1.0 - (latest['priority'] / 4.0)  # Lower priority number = higher weight
        
        # Calculate weighted score (lower usage = higher weight)
        weight = (
            (1.0 - cpu_usage) * self.weights['cpu'] +
            (1.0 - mem_usage_norm) * self.weights['memory'] +
            (1.0 - bw_usage_norm) * self.weights['bandwidth'] +
            priority_norm * self.weights['priority']
        )
        
        # Adjust for current connections (fewer connections = higher weight)
        connection_factor = 1.0 / (1.0 + self.connections[latest['vm_name']] * 0.1)
        weight *= connection_factor
        
        return weight
    
    def select_vm(self, timestamp):
        """Select VM using CBW algorithm"""
        # Get current state data
        time_data = self.df[
            (self.df['timestamp'] <= timestamp)
        ].tail(1000)  # Use recent data
        
        vm_weights = {}
        
        for vm_name in self.vm_names:
            vm_data = time_data[time_data['vm_name'] == vm_name]
            weight = self.calculate_vm_weight(vm_data)
            vm_weights[vm_name] = weight
        
        # Handle case where no weights calculated
        if not vm_weights or all(w == 0 for w in vm_weights.values()):
            # Fallback to round-robin
            vm_name = min(self.connections.items(), key=lambda x: x[1])[0]
        else:
            # Select VM with highest weight (using weighted random selection for better distribution)
            total_weight = sum(vm_weights.values())
            if total_weight > 0:
                probabilities = {vm: w / total_weight for vm, w in vm_weights.items()}
                vm_name = np.random.choice(
                    list(probabilities.keys()),
                    p=list(probabilities.values())
                )
            else:
                vm_name = np.random.choice(self.vm_names)
        
        # Update connection count
        self.connections[vm_name] += 1
        
        # Store selection
        self.request_history.append({
            'timestamp': timestamp,
            'selected_vm': vm_name,
            'weights': vm_weights.copy(),
            'connections': self.connections.copy()
        })
        
        return vm_name
    
    def simulate_load_balancing(self, start_time=None, end_time=None):
        """Simulate load balancing over a time period"""
        if start_time is None:
            start_time = self.df['timestamp'].min()
        if end_time is None:
            end_time = self.df['timestamp'].max()
        
        # Reset state
        self.connections = {vm: 0 for vm in self.vm_names}
        self.request_history = []
        self.simulation_results = []
        
        # Get request timestamps (where update == 1)
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
            selected_vm = self.select_vm(timestamp)
            
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
            
            result_item = {
                'timestamp': timestamp,
                'selected_vm': selected_vm,
                'reward': reward,
                'cpu_usage': vm_state['cpu_usage'] if len(vm_data) > 0 else 0,
                'mem_usage': vm_state['mem_usage'] / vm_state['max_mem'] if len(vm_data) > 0 and vm_state['max_mem'] > 0 else 0,
                'bw_usage': vm_state['bw_usage'] / vm_state['max_bw'] if len(vm_data) > 0 and vm_state['max_bw'] > 0 else 0,
                'score': vm_state['score'] if len(vm_data) > 0 else 0,
                'priority': vm_state['priority'] if len(vm_data) > 0 else 0
            }
            results.append(result_item)
        
        # Store results for get_performance_metrics
        self.simulation_results = results
        
        return results
    
    def _calculate_reward(self, vm_state):
        """Calculate reward for selecting a VM (same as PPO for fair comparison)"""
        cpu_reward = (1.0 - vm_state['cpu_usage']) * 0.3
        mem_usage_norm = vm_state['mem_usage'] / vm_state['max_mem'] if vm_state['max_mem'] > 0 else 0
        mem_reward = (1.0 - mem_usage_norm) * 0.2
        bw_usage_norm = vm_state['bw_usage'] / vm_state['max_bw'] if vm_state['max_bw'] > 0 else 0
        bw_reward = (1.0 - bw_usage_norm) * 0.2
        priority_reward = (1.0 - vm_state['priority'] / 4.0) * 0.2
        
        total_reward = cpu_reward + mem_reward + bw_reward + priority_reward
        return total_reward
    
    def get_vm_distribution(self):
        """Get distribution of requests across VMs"""
        # Use simulation_results if available, otherwise use request_history
        if self.simulation_results:
            distribution = defaultdict(int)
            for req in self.simulation_results:
                distribution[req['selected_vm']] += 1
            return dict(distribution)
        elif self.request_history:
            distribution = defaultdict(int)
            for req in self.request_history:
                distribution[req['selected_vm']] += 1
            return dict(distribution)
        else:
            return {vm: 0 for vm in self.vm_names}
    
    def get_performance_metrics(self):
        """Calculate performance metrics"""
        # Use simulation_results if available
        if not self.simulation_results:
            return {}
        
        results_df = pd.DataFrame(self.simulation_results)
        
        metrics = {
            'total_requests': len(self.simulation_results),
            'avg_reward': results_df['reward'].mean(),
            'std_reward': results_df['reward'].std(),
            'avg_cpu_usage': results_df['cpu_usage'].mean(),
            'avg_mem_usage': results_df['mem_usage'].mean(),
            'avg_bw_usage': results_df['bw_usage'].mean(),
            'avg_score': results_df['score'].mean(),
            'vm_distribution': self.get_vm_distribution(),
            'load_balance_index': self._calculate_load_balance_index(results_df)
        }
        
        return metrics
    
    def _calculate_load_balance_index(self, results_df):
        """Calculate load balance index (lower is better)"""
        vm_cpu_means = results_df.groupby('selected_vm')['cpu_usage'].mean()
        vm_mem_means = results_df.groupby('selected_vm')['mem_usage'].mean()
        
        cpu_variance = vm_cpu_means.var()
        mem_variance = vm_mem_means.var()
        
        return (cpu_variance + mem_variance) / 2.0

class ComparisonAnalyzer:
    """Compare load balancing algorithms using CBWO paper-compliant metrics"""
    
    def __init__(self, df):
        self.df = df
    
    def compare_algorithms(self, algo1_results, algo2_results, algo1_name="Algorithm 1", algo2_name="Algorithm 2"):
        """Compare two algorithms using CBWO paper-compliant metrics"""
        from cbwo_metrics import CBWOMetricsCalculator
        
        # Calculate CBWO metrics for both algorithms
        algo1_metrics_calc = CBWOMetricsCalculator(self.df, algo1_results)
        algo2_metrics_calc = CBWOMetricsCalculator(self.df, algo2_results)
        
        algo1_metrics = algo1_metrics_calc.get_all_metrics(power_coefficient=50.0)
        algo2_metrics = algo2_metrics_calc.get_all_metrics(power_coefficient=50.0)
        
        comparison = {
            'metrics': {
                algo1_name: {
                    'Makespan': algo1_metrics.get('Makespan', 0),
                    'Task Completion Time': algo1_metrics.get('Task Completion Time', 0),
                    'Resource Utilization': algo1_metrics.get('Resource Utilization', 0),
                    'Degree of Imbalance': algo1_metrics.get('Degree of Imbalance', 0),
                    'Energy Consumption': algo1_metrics.get('Energy Consumption', 0),
                    'Execution Time': algo1_metrics.get('Execution Time', 0)
                },
                algo2_name: {
                    'Makespan': algo2_metrics.get('Makespan', 0),
                    'Task Completion Time': algo2_metrics.get('Task Completion Time', 0),
                    'Resource Utilization': algo2_metrics.get('Resource Utilization', 0),
                    'Degree of Imbalance': algo2_metrics.get('Degree of Imbalance', 0),
                    'Energy Consumption': algo2_metrics.get('Energy Consumption', 0),
                    'Execution Time': algo2_metrics.get('Execution Time', 0)
                }
            },
            'improvement': {}
        }
        
        # Calculate improvements (PPO vs baseline)
        # Lower is better for: Makespan, Task Completion Time, Degree of Imbalance, Energy Consumption, Execution Time
        # Higher is better for: Resource Utilization (but we want lower for efficiency)
        
        for metric in ['Makespan', 'Task Completion Time', 'Degree of Imbalance', 'Energy Consumption', 'Execution Time']:
            algo1_val = comparison['metrics'][algo1_name][metric]
            algo2_val = comparison['metrics'][algo2_name][metric]
            
            if algo1_val > 0:
                # Lower is better - calculate reduction percentage
                improvement = ((algo1_val - algo2_val) / algo1_val) * 100
            else:
                improvement = 0
            
            comparison['improvement'][metric] = improvement
        
        # Resource Utilization - lower is better (more efficient)
        algo1_util = comparison['metrics'][algo1_name]['Resource Utilization']
        algo2_util = comparison['metrics'][algo2_name]['Resource Utilization']
        if algo1_util > 0:
            improvement = ((algo1_util - algo2_util) / algo1_util) * 100
        else:
            improvement = 0
        comparison['improvement']['Resource Utilization'] = improvement
        
        return comparison
    
    def get_vm_distribution_comparison(self, algo1_dist, algo2_dist, algo1_name="Algorithm 1", algo2_name="Algorithm 2"):
        """Compare VM distribution between algorithms"""
        all_vms = set(list(algo1_dist.keys()) + list(algo2_dist.keys()))
        
        comparison_data = []
        for vm in all_vms:
            comparison_data.append({
                'VM': vm,
                algo1_name: algo1_dist.get(vm, 0),
                algo2_name: algo2_dist.get(vm, 0)
            })
        
        return pd.DataFrame(comparison_data)

