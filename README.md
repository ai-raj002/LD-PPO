# Load Balancing in Cloud Computing Using Deep Reinforcement Learning (PPO)

A comprehensive Streamlit application for analyzing load balancing in cloud computing environments, comparing traditional algorithms (Round Robin, CBWO) with **Proximal Policy Optimization (PPO)** - a state-of-the-art deep reinforcement learning algorithm for intelligent load distribution.

## 🎯 Features

### 📊 Data Overview
- Interactive dataset exploration
- Statistical summaries
- VM distribution analysis
- Real-time data preview

### 📈 Performance Analysis
- Time series visualization of resource metrics
- Resource utilization comparisons across VMs
- Heatmaps for resource usage patterns
- Load balancing efficiency metrics

### 🤖 PPO Load Balancer
- Trainable PPO model for intelligent load balancing
- Configurable hyperparameters:
  - Learning rate
  - Training steps
  - Batch size
  - Epochs per update
  - Discount factor (γ)
  - Clip range
- Real-time training visualization
- Model evaluation and performance metrics
- VM selection distribution analysis

### ⚙️ CBW Load Balancer
- Connection-Based Weighted load balancing simulation
- Rule-based algorithm considering:
  - Current connection count
  - CPU, Memory, and Bandwidth usage
  - Priority levels
  - Resource availability
- No training required - instant results
- Performance metrics and visualization

### 🔄 Algorithm Comparison
- **Side-by-side comparison** of CBW vs PPO
- Performance metrics comparison:
  - Average reward
  - CPU/Memory/Bandwidth usage
  - Load balance index
  - VM distribution
- Improvement percentage calculations
- Reward distribution comparison
- Key insights and recommendations

### 🔍 Deep Insights
- Correlation analysis between resources
- Priority distribution analysis
- Network traffic pattern visualization
- Anomaly detection
- Comprehensive summary statistics

## 🚀 Installation

1. **Clone or download this repository**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure your CSV data files are in the same directory:**
   - `LeastConn_22-06-2025_10VUs_30m.csv`
   - `LeastConn_23-06-2025_50VUs_1h.csv`

## 📖 Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## 📁 Project Structure

```
.
├── app.py                      # Main Streamlit application
├── ppo_load_balancer.py        # PPO implementation and environment
├── data_analysis.py            # Data analysis utilities
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── LeastConn_22-06-2025_10VUs_30m.csv  # Dataset 1
└── LeastConn_23-06-2025_50VUs_1h.csv    # Dataset 2
```

## 🔬 Load Balancing Algorithms

### Connection-Based Weighted (CBW)

CBW is a traditional rule-based load balancing algorithm that:
- **Distributes requests** based on weighted scores
- **Considers multiple factors**:
  - Current connection count (fewer = higher priority)
  - CPU usage (lower = higher weight)
  - Memory usage (lower = higher weight)
  - Bandwidth usage (lower = higher weight)
  - Priority levels (lower number = higher priority)
- **No training required** - works immediately
- **Deterministic** - same inputs produce same outputs

### Proximal Policy Optimization (PPO)

PPO is a policy gradient method that:
1. **Learns a policy** to select which VM to route requests to
2. **Optimizes resource utilization** by balancing load across VMs
3. **Uses clipped objective** to prevent large policy updates
4. **Maximizes reward** based on:
   - Low CPU usage
   - Low memory usage
   - Low bandwidth usage
   - High priority (lower priority number)
   - Balanced load distribution

### Environment

The load balancing environment:
- **State Space**: Normalized resource metrics (CPU, memory, bandwidth, score, priority, time) for each VM
- **Action Space**: Discrete selection of which VM to route a request to
- **Reward Function**: Combines multiple objectives:
  - Resource availability (lower usage = higher reward)
  - Priority consideration
  - Load balancing fairness

### Training Process

1. The agent observes the current state of all VMs
2. Selects an action (VM) based on the current policy
3. Receives a reward based on the selected VM's state
4. Updates the policy using PPO algorithm
5. Repeats to learn optimal load balancing strategy

## 📊 Data Format

The CSV files should contain the following columns:
- `no`: Record number
- `fetch`: Fetch number
- `update`: Update flag (0 or 1)
- `vm_id`: Virtual machine ID
- `vm_name`: VM name
- `cpu_usage`: CPU usage (0-1)
- `max_cpu`: Maximum CPU capacity
- `mem_usage`: Memory usage (bytes)
- `max_mem`: Maximum memory capacity
- `cum_netin`: Cumulative network input
- `cum_netout`: Cumulative network output
- `rate_netin`: Network input rate
- `rate_netout`: Network output rate
- `bw_usage`: Bandwidth usage
- `max_bw`: Maximum bandwidth
- `score`: Load balancing score
- `priority`: Priority level (1-4)
- `unix_timestamp`: Unix timestamp
- `timestamp`: Human-readable timestamp

## 🎛️ Configuration

### PPO Hyperparameters

- **Learning Rate**: Controls how fast the model learns (typically 0.0001-0.01)
- **Training Steps**: Number of training iterations
- **Batch Size**: Number of samples per update (typically 32-512)
- **Epochs per Update**: Number of times to iterate over the batch
- **Discount Factor (γ)**: How much future rewards are valued (0.9-0.99)
- **Clip Range**: Clipping parameter for PPO (typically 0.1-0.3)

## 📈 Metrics Explained

### Load Balance Index
Measures how evenly load is distributed across VMs. Lower values indicate better balance.

### Resource Variance
Variance in resource usage across VMs. Lower variance means more balanced load.

### Overall Efficiency
Normalized measure of load balancing efficiency (0-100%).

### Throughput
Number of requests processed per second.

### Resource Fairness
Coefficient of variation-based measure of fairness in resource distribution.

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. **File Not Found**: Ensure CSV files are in the same directory as `app.py`

3. **Memory Issues**: For large datasets, consider:
   - Reducing training steps
   - Using smaller batch sizes
   - Processing data in chunks

4. **Training Takes Too Long**: 
   - Reduce number of training steps
   - Use smaller batch sizes
   - Reduce number of epochs per update

## 📊 Comparison Results

The application provides comprehensive comparison metrics:

- **Reward Improvement**: How much better PPO performs vs CBW
- **Resource Usage Reduction**: CPU, Memory, and Bandwidth efficiency gains
- **Load Balance Index**: Measure of how evenly load is distributed
- **VM Distribution**: Visual comparison of request routing patterns

### When to Use Each Algorithm

**CBW (Connection-Based Weighted)**:
- ✅ Simple, interpretable rules
- ✅ No training time required
- ✅ Consistent performance
- ✅ Easy to implement and maintain

**PPO (Deep Reinforcement Learning)**:
- ✅ Learns optimal strategies from data
- ✅ Adapts to changing patterns
- ✅ Can discover non-obvious optimizations
- ✅ Potentially better long-term performance

## 🔮 Future Enhancements

- [ ] Support for more RL algorithms (A3C, DQN, etc.)
- [ ] Real-time data streaming
- [ ] Model persistence and loading
- [ ] Advanced visualization options
- [ ] Export analysis reports
- [ ] Multi-objective optimization
- [ ] Additional traditional algorithms (Round-Robin, Least Connections, etc.)

## 📝 License

This project is provided as-is for educational and research purposes.

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📧 Contact

For questions or suggestions, please open an issue in the repository.

---

**Note**: This application is designed for analyzing load balancing data and demonstrating DRL concepts. For production use, additional considerations such as model validation, robustness testing, and deployment infrastructure should be addressed.

