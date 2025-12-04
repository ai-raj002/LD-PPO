import pandas as pd
import numpy as np
from scipy import stats

class DataAnalyzer:
    """Comprehensive data analysis for load balancing data"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.prepare_data()
    
    def prepare_data(self):
        """Prepare data for analysis"""
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.df['timestamp']):
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        # Calculate normalized metrics
        self.df['cpu_usage_norm'] = self.df['cpu_usage'] / self.df['max_cpu']
        self.df['mem_usage_norm'] = self.df['mem_usage'] / self.df['max_mem']
        self.df['bw_usage_norm'] = self.df['bw_usage'] / self.df['max_bw']
        
        # Calculate resource utilization score
        self.df['resource_util'] = (
            self.df['cpu_usage_norm'] * 0.4 +
            self.df['mem_usage_norm'] * 0.3 +
            self.df['bw_usage_norm'] * 0.3
        )
    
    def calculate_efficiency_metrics(self):
        """Calculate load balancing efficiency metrics"""
        metrics = {}
        
        # Load Balance Index (LBI): measures how evenly load is distributed
        # Lower variance = better balance
        cpu_by_vm = self.df.groupby('vm_name')['cpu_usage'].mean()
        mem_by_vm = self.df.groupby('vm_name')['mem_usage'].mean()
        
        cpu_variance = cpu_by_vm.var()
        mem_variance = mem_by_vm.var()
        
        # Normalized balance index (0 = perfect balance, 1 = worst)
        balance_index = (cpu_variance + mem_variance) / 2.0
        
        metrics['balance_index'] = balance_index
        metrics['cpu_variance'] = cpu_variance
        metrics['mem_variance'] = mem_variance
        
        # Overall efficiency: inverse of balance index, normalized
        max_possible_variance = 0.25  # Assuming max variance for 0-1 range
        efficiency = 1.0 - min(balance_index / max_possible_variance, 1.0)
        metrics['overall_efficiency'] = efficiency
        
        # Resource utilization rate
        metrics['avg_cpu_util'] = self.df['cpu_usage'].mean()
        metrics['avg_mem_util'] = self.df['mem_usage_norm'].mean()
        metrics['avg_bw_util'] = self.df['bw_usage_norm'].mean()
        
        return metrics
    
    def analyze_priority_distribution(self):
        """Analyze priority distribution across VMs"""
        priority_stats = self.df.groupby(['vm_name', 'priority']).size().reset_index(name='count')
        return priority_stats
    
    def detect_anomalies(self, method='iqr', threshold=2.0):
        """Detect anomalies in resource usage"""
        anomalies = []
        
        if method == 'iqr':
            # Interquartile Range method
            for metric in ['cpu_usage', 'mem_usage', 'bw_usage']:
                Q1 = self.df[metric].quantile(0.25)
                Q3 = self.df[metric].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                metric_anomalies = self.df[
                    (self.df[metric] < lower_bound) | (self.df[metric] > upper_bound)
                ]
                
                if len(metric_anomalies) > 0:
                    anomalies.append(metric_anomalies)
        
        elif method == 'zscore':
            # Z-score method
            for metric in ['cpu_usage', 'mem_usage', 'bw_usage']:
                z_scores = np.abs(stats.zscore(self.df[metric]))
                metric_anomalies = self.df[z_scores > threshold]
                
                if len(metric_anomalies) > 0:
                    anomalies.append(metric_anomalies)
        
        if anomalies:
            combined_anomalies = pd.concat(anomalies).drop_duplicates()
            return combined_anomalies.sort_values('timestamp')
        else:
            return pd.DataFrame()
    
    def get_summary_statistics(self):
        """Get comprehensive summary statistics by VM"""
        summary = []
        
        for vm_name in self.df['vm_name'].unique():
            vm_data = self.df[self.df['vm_name'] == vm_name]
            
            stats_dict = {
                'VM': vm_name,
                'Records': len(vm_data),
                'Avg CPU Usage': vm_data['cpu_usage'].mean(),
                'Std CPU Usage': vm_data['cpu_usage'].std(),
                'Max CPU Usage': vm_data['cpu_usage'].max(),
                'Avg Memory Usage': vm_data['mem_usage_norm'].mean(),
                'Std Memory Usage': vm_data['mem_usage_norm'].std(),
                'Avg Bandwidth Usage': vm_data['bw_usage_norm'].mean(),
                'Avg Score': vm_data['score'].mean(),
                'Avg Priority': vm_data['priority'].mean(),
                'Total Network In': vm_data['cum_netin'].iloc[-1] if len(vm_data) > 0 else 0,
                'Total Network Out': vm_data['cum_netout'].iloc[-1] if len(vm_data) > 0 else 0,
            }
            
            summary.append(stats_dict)
        
        return pd.DataFrame(summary)
    
    def calculate_load_distribution(self):
        """Calculate how load is distributed across VMs"""
        distribution = {}
        
        for vm_name in self.df['vm_name'].unique():
            vm_data = self.df[self.df['vm_name'] == vm_name]
            
            distribution[vm_name] = {
                'cpu_load': vm_data['cpu_usage'].mean(),
                'mem_load': vm_data['mem_usage_norm'].mean(),
                'bw_load': vm_data['bw_usage_norm'].mean(),
                'total_load': vm_data['resource_util'].mean(),
                'request_count': len(vm_data[vm_data['update'] == 1])
            }
        
        return distribution
    
    def identify_bottlenecks(self):
        """Identify potential bottlenecks"""
        bottlenecks = []
        
        # High CPU usage
        high_cpu = self.df[self.df['cpu_usage'] > 0.8]
        if len(high_cpu) > 0:
            bottlenecks.append({
                'type': 'High CPU Usage',
                'count': len(high_cpu),
                'vms': high_cpu['vm_name'].unique().tolist()
            })
        
        # High memory usage
        high_mem = self.df[self.df['mem_usage_norm'] > 0.8]
        if len(high_mem) > 0:
            bottlenecks.append({
                'type': 'High Memory Usage',
                'count': len(high_mem),
                'vms': high_mem['vm_name'].unique().tolist()
            })
        
        # High bandwidth usage
        high_bw = self.df[self.df['bw_usage_norm'] > 0.8]
        if len(high_bw) > 0:
            bottlenecks.append({
                'type': 'High Bandwidth Usage',
                'count': len(high_bw),
                'vms': high_bw['vm_name'].unique().tolist()
            })
        
        return bottlenecks
    
    def calculate_performance_metrics(self):
        """Calculate performance metrics"""
        metrics = {}
        
        # Response time proxy (using score as indicator)
        metrics['avg_response_score'] = self.df['score'].mean()
        metrics['min_response_score'] = self.df['score'].min()
        metrics['max_response_score'] = self.df['score'].max()
        
        # Throughput (requests per time unit)
        time_span = (self.df['timestamp'].max() - self.df['timestamp'].min()).total_seconds()
        total_requests = len(self.df[self.df['update'] == 1])
        metrics['throughput'] = total_requests / time_span if time_span > 0 else 0
        
        # Resource utilization efficiency
        metrics['resource_efficiency'] = self.df['resource_util'].mean()
        
        # Load balancing fairness (using coefficient of variation)
        cpu_by_vm = self.df.groupby('vm_name')['cpu_usage'].mean()
        metrics['cpu_fairness'] = 1.0 / (1.0 + cpu_by_vm.std() / (cpu_by_vm.mean() + 1e-8))
        
        return metrics

