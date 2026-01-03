import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from ppo_load_balancer import PPOLoadBalancer, create_environment
from data_analysis import DataAnalyzer
from cbw_load_balancer import ComparisonAnalyzer
from additional_algorithms import (
    RoundRobinLoadBalancer
)
from cbwo_load_balancer import CBWOLoadBalancer
from cbwo_metrics import CBWOMetricsCalculator
from export_utils import (
    export_results_to_csv,
    export_comparison_to_csv,
    generate_comparison_report,
    create_download_link
)

# Page configuration
st.set_page_config(
    page_title="Load Balancing in Cloud Computing Using Deep Reinforcement Learning (PPO)",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

def load_data():
    """Load CSV data files"""
    data_files = {
        "10 VUs - 30 minutes": "LeastConn_22-06-2025_10VUs_30m.csv",
        "50 VUs - 1 hour": "LeastConn_23-06-2025_50VUs_1h.csv"
    }
    
    loaded_data = {}
    for name, filename in data_files.items():
        filepath = os.path.join(os.path.dirname(__file__), filename)
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                loaded_data[name] = df
            except Exception as e:
                st.error(f"Error loading {filename}: {str(e)}")
    
    return loaded_data

def main():
    st.markdown('<h1 class="main-header">⚖️ Load Balancing in Cloud Computing Using Deep Reinforcement Learning (PPO)</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["📊 Data Overview", "📈 Performance Analysis", "🤖 PPO Load Balancer", "🔄 Algorithm Comparison", "📊 Multi-Algorithm Comparison", "🔍 Deep Insights"]
    )
    
    # Load data
    data_dict = load_data()
    
    if not data_dict:
        st.error("No data files found. Please ensure CSV files are in the same directory.")
        return
    
    # Data Overview Page
    if page == "📊 Data Overview":
        show_data_overview(data_dict)
    
    # Performance Analysis Page
    elif page == "📈 Performance Analysis":
        show_performance_analysis(data_dict)
    
    # PPO Load Balancer Page
    elif page == "🤖 PPO Load Balancer":
        show_ppo_balancer(data_dict)
    
    # Algorithm Comparison Page
    elif page == "🔄 Algorithm Comparison":
        show_algorithm_comparison(data_dict)
    
    # Multi-Algorithm Comparison Page
    elif page == "📊 Multi-Algorithm Comparison":
        show_multi_algorithm_comparison(data_dict)
    
    # Deep Insights Page
    elif page == "🔍 Deep Insights":
        show_deep_insights(data_dict)

def show_data_overview(data_dict):
    st.header("📊 Data Overview")
    
    # Dataset selection
    selected_dataset = st.selectbox("Select Dataset", list(data_dict.keys()))
    df = data_dict[selected_dataset]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Unique VMs", df['vm_id'].nunique())
    with col3:
        st.metric("Time Range", f"{df['timestamp'].min().strftime('%Y-%m-%d %H:%M')} to {df['timestamp'].max().strftime('%Y-%m-%d %H:%M')}")
    with col4:
        st.metric("Total Duration", f"{(df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 60:.1f} minutes")
    
    st.subheader("Dataset Preview")
    st.dataframe(df.head(100), use_container_width=True)
    
    st.subheader("Data Statistics")
    numeric_cols = ['cpu_usage', 'mem_usage', 'rate_netin', 'rate_netout', 'bw_usage', 'score', 'priority']
    st.dataframe(df[numeric_cols].describe(), use_container_width=True)
    
    st.subheader("VM Distribution")
    vm_counts = df['vm_name'].value_counts()
    fig = px.bar(
        x=vm_counts.index,
        y=vm_counts.values,
        labels={'x': 'Virtual Machine', 'y': 'Record Count'},
        title="Records per VM"
    )
    st.plotly_chart(fig, use_container_width=True)

def show_performance_analysis(data_dict):
    st.header("📈 Performance Analysis")
    
    selected_dataset = st.selectbox("Select Dataset", list(data_dict.keys()), key="perf_dataset")
    df = data_dict[selected_dataset]
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        selected_vms = st.multiselect("Select VMs", df['vm_name'].unique(), default=df['vm_name'].unique())
    with col2:
        metric_type = st.selectbox("Select Metric", ["Resource Utilization (CPU)", "Score", "Priority"])
    
    df_filtered = df[df['vm_name'].isin(selected_vms)]
    
    # Time series plots
    st.subheader("Time Series Analysis")
    
    metric_map = {
        "Resource Utilization (CPU)": "cpu_usage",
        "Score": "score",
        "Priority": "priority"
    }
    
    metric_col = metric_map[metric_type]
    
    # Convert CPU usage to percentage for Resource Utilization
    if metric_type == "Resource Utilization (CPU)":
        y_values = df_filtered[metric_col] * 100
        y_label = "Resource Utilization (%)"
    else:
        y_values = df_filtered[metric_col]
        y_label = metric_type
    
    fig = px.line(
        df_filtered,
        x='timestamp',
        y=y_values,
        color='vm_name',
        title=f"{metric_type} Over Time",
        labels={'y': y_label, 'timestamp': 'Time'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Resource utilization comparison
    st.subheader("Resource Utilization Comparison")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_cpu = df_filtered.groupby('vm_name')['cpu_usage'].mean()
        fig_cpu = px.bar(
            x=avg_cpu.index,
            y=avg_cpu.values,
            labels={'x': 'VM', 'y': 'Average CPU Usage'},
            title="Average CPU Usage by VM"
        )
        st.plotly_chart(fig_cpu, use_container_width=True)
    
    with col2:
        # Resource Utilization (CPU-based)
        avg_cpu_pct = df_filtered.groupby('vm_name')['cpu_usage'].mean() * 100
        fig_resource = px.bar(
            x=avg_cpu_pct.index,
            y=avg_cpu_pct.values,
            labels={'x': 'VM', 'y': 'Resource Utilization (%)'},
            title="Resource Utilization by VM"
        )
        st.plotly_chart(fig_resource, use_container_width=True)
    
    with col3:
        # Calculate Degree of Imbalance
        cpu_by_vm = df_filtered.groupby('vm_name')['cpu_usage'].mean()
        mean_cpu = cpu_by_vm.mean()
        std_cpu = cpu_by_vm.std()
        doi = std_cpu / mean_cpu if mean_cpu > 0 else 0
        
        st.metric("Degree of Imbalance", f"{doi:.4f}")
        st.info("Lower DoI indicates better load distribution")
    
    # Resource Utilization Heatmap (CPU only)
    st.subheader("Resource Utilization Heatmap")
    df_pivot = df_filtered.pivot_table(
        values=['cpu_usage'],
        index='vm_name',
        aggfunc='mean'
    )
    df_pivot = df_pivot * 100  # Convert to percentage
    fig_heat = px.imshow(
        df_pivot.T,
        labels=dict(x="VM", y="Resource", color="Utilization (%)"),
        title="Resource Utilization Heatmap",
        aspect="auto",
        color_continuous_scale="RdYlGn_r"
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    
    # Load balancing efficiency - CBWO metrics
    st.subheader("Load Balancing Efficiency Metrics")
    analyzer = DataAnalyzer(df_filtered)
    
    # Calculate Degree of Imbalance
    cpu_by_vm = df_filtered.groupby('vm_name')['cpu_usage'].mean()
    mean_cpu = cpu_by_vm.mean()
    std_cpu = cpu_by_vm.std()
    doi = std_cpu / mean_cpu if mean_cpu > 0 else 0
    
    # Resource Utilization
    resource_util = df_filtered['cpu_usage'].mean() * 100
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Resource Utilization", f"{resource_util:.2f}%")
    with col2:
        st.metric("Degree of Imbalance", f"{doi:.4f}")
    with col3:
        st.metric("CPU Variance", f"{cpu_by_vm.var():.4f}")

def show_ppo_balancer(data_dict):
    st.header("🤖 PPO-Based Load Balancer")
    
    st.markdown("""
    ### Proximal Policy Optimization (PPO) for Load Balancing
    
    PPO is a policy gradient method that uses a clipped objective function to prevent 
    large policy updates. In load balancing, PPO learns to distribute requests across 
    VMs to optimize resource utilization and minimize response time.
    """)
    
    selected_dataset = st.selectbox("Select Dataset", list(data_dict.keys()), key="ppo_dataset")
    df = data_dict[selected_dataset]
    
    # PPO Configuration
    st.subheader("PPO Configuration")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        learning_rate = st.slider("Learning Rate", 0.0001, 0.01, 0.001, 0.0001)
        n_steps = st.slider("Training Steps", 100, 5000, 1000, 100)
    with col2:
        batch_size = st.slider("Batch Size", 32, 512, 128, 32)
        n_epochs = st.slider("Epochs per Update", 1, 10, 4)
    with col3:
        gamma = st.slider("Discount Factor (γ)", 0.9, 0.99, 0.95, 0.01)
        clip_range = st.slider("Clip Range", 0.1, 0.3, 0.2, 0.05)
    
    if st.button("🚀 Train PPO Model", type="primary"):
        with st.spinner("Training PPO model..."):
            # Prepare environment
            env = create_environment(df)
            
            # Initialize PPO
            ppo_balancer = PPOLoadBalancer(
                state_dim=env.observation_space.shape[0],
                action_dim=env.action_space.n,
                learning_rate=learning_rate,
                gamma=gamma,
                clip_range=clip_range
            )
            
            # Training progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            training_history = []
            for step in range(n_steps):
                state = env.reset()
                episode_reward = 0
                done = False
                
                while not done:
                    action = ppo_balancer.select_action(state)
                    next_state, reward, done, info = env.step(action)
                    ppo_balancer.store_transition(state, action, reward, next_state, done)
                    state = next_state
                    episode_reward += reward
                
                if len(ppo_balancer.buffer) >= batch_size:
                    loss = ppo_balancer.update(n_epochs=n_epochs, batch_size=batch_size)
                    training_history.append({
                        'step': step,
                        'reward': episode_reward,
                        'loss': loss
                    })
                
                progress = (step + 1) / n_steps
                progress_bar.progress(progress)
                status_text.text(f"Training step {step + 1}/{n_steps} | Reward: {episode_reward:.2f}")
            
            st.success("Training completed!")
            
            # Store in session state
            st.session_state['ppo_model'] = ppo_balancer
            st.session_state['training_history'] = training_history
            st.session_state['env'] = env
    
    # Display training results
    if 'training_history' in st.session_state and st.session_state['training_history']:
        st.subheader("Training Results")
        
        history_df = pd.DataFrame(st.session_state['training_history'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_loss = px.line(
                history_df,
                x='step',
                y='loss',
                title="PPO Training Loss Over Time",
                labels={'step': 'Training Step', 'loss': 'Loss'}
            )
            st.plotly_chart(fig_loss, use_container_width=True)
        
        with col2:
            st.info("""
            **Training Progress:**
            - Lower loss indicates better policy convergence
            - The model learns optimal VM selection strategies
            - Training metrics are internal to the learning process
            """)
        
        # Evaluate model
        if st.button("📊 Evaluate Model Performance"):
            if 'ppo_model' in st.session_state and 'env' in st.session_state:
                ppo_model = st.session_state['ppo_model']
                env = st.session_state['env']
                
                # Run evaluation and collect results
                eval_results = []
                vm_selections = {vm: 0 for vm in df['vm_name'].unique()}
                
                # Get request timestamps for evaluation
                requests = df[df['update'] == 1].head(100)
                
                for idx, row in requests.iterrows():
                    timestamp = row['timestamp']
                    time_data = df[df['timestamp'] <= timestamp].tail(1000)
                    
                    if len(time_data) == 0:
                        continue
                    
                    # Build state vector
                    state_vector = []
                    for vm_name in df['vm_name'].unique():
                        vm_data = time_data[time_data['vm_name'] == vm_name].tail(1)
                        if len(vm_data) > 0:
                            latest = vm_data.iloc[0]
                            state_vector.extend([
                                latest['cpu_usage'],
                                latest['mem_usage'] / latest['max_mem'] if latest['max_mem'] > 0 else 0,
                                latest['bw_usage'] / latest['max_bw'] if latest['max_bw'] > 0 else 0,
                                latest['score'] / 10.0,
                                latest['priority'] / 4.0,
                                0.5
                            ])
                        else:
                            state_vector.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                    
                    state_array = np.array(state_vector, dtype=np.float32)
                    action = ppo_model.select_action(state_array, deterministic=True)
                    selected_vm = df['vm_name'].unique()[action]
                    vm_selections[selected_vm] += 1
                    
                    vm_data = time_data[time_data['vm_name'] == selected_vm].tail(1)
                    if len(vm_data) > 0:
                        vm_state = vm_data.iloc[0]
                        eval_results.append({
                            'timestamp': timestamp,
                            'selected_vm': selected_vm,
                            'cpu_usage': vm_state['cpu_usage'],
                            'score': vm_state['score']
                        })
                
                # Calculate CBWO-compliant metrics
                if eval_results:
                    metrics_calc = CBWOMetricsCalculator(df, eval_results)
                    metrics = metrics_calc.get_all_metrics(power_coefficient=50.0, algorithm_type="ppo")
                    
                    st.subheader("Evaluation Results - CBWO Paper-Compliant Metrics")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Makespan", f"{metrics.get('Makespan', 0):.2f} s")
                        st.metric("Task Completion Time", f"{metrics.get('Task Completion Time', 0):.2f} s")
                    with col2:
                        st.metric("Resource Utilization", f"{metrics.get('Resource Utilization', 0):.2f}%")
                        st.metric("Degree of Imbalance", f"{metrics.get('Degree of Imbalance', 0):.4f}")
                    with col3:
                        st.metric("Energy Consumption", f"{metrics.get('Energy Consumption', 0):.2f} J")
                        st.metric("Execution Time", f"{metrics.get('Execution Time', 0):.2f} s")
                    
                    # VM Distribution
                    fig_dist = px.bar(
                        x=list(vm_selections.keys()),
                        y=list(vm_selections.values()),
                        labels={'x': 'VM', 'y': 'Selection Count'},
                        title="VM Selection Distribution (PPO Policy)"
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)

def show_deep_insights(data_dict):
    st.header("🔍 Deep Insights")
    
    selected_dataset = st.selectbox("Select Dataset", list(data_dict.keys()), key="insights_dataset")
    df = data_dict[selected_dataset]
    
    analyzer = DataAnalyzer(df)
    
    # Correlation analysis
    st.subheader("Resource Correlation Analysis")
    numeric_cols = ['cpu_usage', 'mem_usage', 'rate_netin', 'rate_netout', 'bw_usage', 'score', 'priority']
    corr_matrix = df[numeric_cols].corr()
    
    fig_corr = px.imshow(
        corr_matrix,
        labels=dict(color="Correlation"),
        title="Resource Correlation Matrix",
        aspect="auto",
        color_continuous_scale="RdBu"
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Priority analysis
    st.subheader("Priority Distribution Analysis")
    priority_analysis = analyzer.analyze_priority_distribution()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_priority = px.histogram(
            df,
            x='priority',
            color='vm_name',
            title="Priority Distribution by VM",
            labels={'priority': 'Priority Level', 'count': 'Frequency'}
        )
        st.plotly_chart(fig_priority, use_container_width=True)
    
    with col2:
        priority_stats = df.groupby('vm_name')['priority'].agg(['mean', 'std', 'min', 'max'])
        st.dataframe(priority_stats, use_container_width=True)
    
    # Network analysis
    st.subheader("Network Traffic Analysis")
    
    fig_network = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Network Input Rate', 'Network Output Rate'),
        vertical_spacing=0.1
    )
    
    for vm in df['vm_name'].unique():
        vm_data = df[df['vm_name'] == vm]
        fig_network.add_trace(
            go.Scatter(
                x=vm_data['timestamp'],
                y=vm_data['rate_netin'],
                name=f"{vm} (In)",
                mode='lines'
            ),
            row=1, col=1
        )
        fig_network.add_trace(
            go.Scatter(
                x=vm_data['timestamp'],
                y=vm_data['rate_netout'],
                name=f"{vm} (Out)",
                mode='lines'
            ),
            row=2, col=1
        )
    
    fig_network.update_xaxes(title_text="Time", row=2, col=1)
    fig_network.update_yaxes(title_text="Rate (bytes/s)", row=1, col=1)
    fig_network.update_yaxes(title_text="Rate (bytes/s)", row=2, col=1)
    fig_network.update_layout(height=600, title_text="Network Traffic Patterns")
    
    st.plotly_chart(fig_network, use_container_width=True)
    
    # Anomaly detection
    st.subheader("Anomaly Detection")
    
    anomalies = analyzer.detect_anomalies()
    
    if len(anomalies) > 0:
        st.warning(f"Found {len(anomalies)} potential anomalies")
        st.dataframe(anomalies.head(20), use_container_width=True)
        
        # Visualize anomalies
        fig_anomaly = px.scatter(
            df,
            x='timestamp',
            y='cpu_usage',
            color='vm_name',
            size='score',
            title="CPU Usage with Anomalies Highlighted",
            hover_data=['priority', 'mem_usage']
        )
        st.plotly_chart(fig_anomaly, use_container_width=True)
    else:
        st.success("No significant anomalies detected")
    
    # Summary statistics
    st.subheader("Summary Statistics by VM")
    summary_stats = analyzer.get_summary_statistics()
    st.dataframe(summary_stats, use_container_width=True)

def show_algorithm_comparison(data_dict):
    st.header("🔄 PPO vs Round Robin Algorithm Comparison")
    
    st.markdown("""
    ### Compare Deep Reinforcement Learning (PPO) vs Round Robin
    
    This page allows you to compare the performance of both load balancing algorithms
    side-by-side using the same dataset and time period.
    """)
    
    selected_dataset = st.selectbox("Select Dataset", list(data_dict.keys()), key="comp_dataset")
    df = data_dict[selected_dataset]
    
    # Configuration
    st.subheader("Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Time Range**")
        start_time = st.date_input("Start Date", value=df['timestamp'].min().date(), key="comp_start")
        start_datetime = pd.Timestamp.combine(start_time, df['timestamp'].min().time())
        end_time = st.date_input("End Date", value=df['timestamp'].max().date(), key="comp_end")
        end_datetime = pd.Timestamp.combine(end_time, df['timestamp'].max().time())
    
    with col2:
        st.markdown("**PPO Training Parameters**")
        ppo_learning_rate = st.slider("Learning Rate", 0.0001, 0.01, 0.001, 0.0001, key="comp_lr")
        ppo_steps = st.slider("Training Steps", 100, 2000, 500, 100, key="comp_steps")
        ppo_batch_size = st.slider("Batch Size", 32, 256, 64, 32, key="comp_batch")
    
    if st.button("🚀 Run Comparison", type="primary"):
        try:
            # Run Round Robin
            rr_status = st.empty()
            rr_status.info("🔄 Running Round Robin simulation...")
            
            rr_balancer = RoundRobinLoadBalancer(df)
            rr_results = rr_balancer.simulate_load_balancing(start_datetime, end_datetime)
            rr_metrics = rr_balancer.get_performance_metrics(power_coefficient=50.0)
            
            rr_status.success(f"✅ Round Robin completed: {len(rr_results)} requests processed")
            
            if not rr_results or len(rr_results) == 0:
                st.error("❌ No Round Robin results generated. Please check the time range and ensure there are requests (update=1) in the selected period.")
                st.stop()
            
            # Run PPO
            ppo_status = st.empty()
            ppo_status.info("🔄 Training PPO model...")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            env = create_environment(df)
            ppo_balancer = PPOLoadBalancer(
                state_dim=env.observation_space.shape[0],
                action_dim=env.action_space.n,
                learning_rate=ppo_learning_rate
            )
            
            # Train PPO with progress tracking
            max_steps_per_episode = 50  # Limit steps per episode to prevent infinite loops
            for step in range(ppo_steps):
                state = env.reset()
                done = False
                episode_steps = 0
                
                while not done and episode_steps < max_steps_per_episode:
                    action = ppo_balancer.select_action(state)
                    next_state, reward, done, info = env.step(action)
                    ppo_balancer.store_transition(state, action, reward, next_state, done)
                    state = next_state
                    episode_steps += 1
                
                if len(ppo_balancer.buffer) >= ppo_batch_size:
                    ppo_balancer.update(n_epochs=4, batch_size=ppo_batch_size)
                
                # Update progress
                progress = (step + 1) / ppo_steps
                progress_bar.progress(progress)
                status_text.text(f"Training PPO: Step {step + 1}/{ppo_steps}")
            
            progress_bar.empty()
            status_text.empty()
            ppo_status.success("✅ PPO training completed")
                
            # Evaluate PPO on actual request timestamps
            eval_status = st.empty()
            eval_status.info("🔄 Evaluating PPO model on request timestamps...")
            
            ppo_results = []
            vm_selections = {vm: 0 for vm in df['vm_name'].unique()}
            
            # Get request timestamps from Round Robin results
            request_timestamps = [r['timestamp'] for r in rr_results]
            
            # Limit to reasonable number for performance
            max_eval_requests = min(100, len(request_timestamps))
            
            eval_progress = st.progress(0)
            eval_status_detail = st.empty()
            
            for i, timestamp in enumerate(request_timestamps[:max_eval_requests]):
                eval_progress.progress((i + 1) / max_eval_requests)
                eval_status_detail.text(f"Evaluating PPO: {i + 1}/{max_eval_requests}")
                try:
                    # Get VM state at this timestamp
                    time_data = df[df['timestamp'] <= timestamp].tail(1000)
                    
                    if len(time_data) == 0:
                        continue
                    
                    # Build state vector similar to environment
                    state_vector = []
                    for vm_name in df['vm_name'].unique():
                        vm_data = time_data[time_data['vm_name'] == vm_name].tail(1)
                        if len(vm_data) > 0:
                            latest = vm_data.iloc[0]
                            state_vector.extend([
                                latest['cpu_usage'],
                                latest['mem_usage'] / latest['max_mem'] if latest['max_mem'] > 0 else 0,
                                latest['bw_usage'] / latest['max_bw'] if latest['max_bw'] > 0 else 0,
                                latest['score'] / 10.0,
                                latest['priority'] / 4.0,
                                (timestamp - df['timestamp'].min()).total_seconds() / (df['timestamp'].max() - df['timestamp'].min()).total_seconds() if (df['timestamp'].max() - df['timestamp'].min()).total_seconds() > 0 else 0
                            ])
                        else:
                            state_vector.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                    
                    state_array = np.array(state_vector, dtype=np.float32)
                    
                    # Use PPO to select VM
                    action = ppo_balancer.select_action(state_array, deterministic=True)
                    selected_vm = df['vm_name'].unique()[action]
                    vm_selections[selected_vm] += 1
                    
                    # Get actual VM state
                    vm_data = time_data[time_data['vm_name'] == selected_vm].tail(1)
                    
                    if len(vm_data) > 0:
                        vm_state = vm_data.iloc[0]
                        # Calculate reward using same formula as Round Robin
                        cpu_reward = (1.0 - vm_state['cpu_usage']) * 0.3
                        mem_usage_norm = vm_state['mem_usage'] / vm_state['max_mem'] if vm_state['max_mem'] > 0 else 0
                        mem_reward = (1.0 - mem_usage_norm) * 0.2
                        bw_usage_norm = vm_state['bw_usage'] / vm_state['max_bw'] if vm_state['max_bw'] > 0 else 0
                        bw_reward = (1.0 - bw_usage_norm) * 0.2
                        priority_reward = (1.0 - vm_state['priority'] / 4.0) * 0.2
                        reward = cpu_reward + mem_reward + bw_reward + priority_reward
                        
                        ppo_results.append({
                            'timestamp': timestamp,
                            'selected_vm': selected_vm,
                            'reward': reward,
                            'cpu_usage': vm_state['cpu_usage'],
                            'mem_usage': mem_usage_norm,
                            'bw_usage': bw_usage_norm,
                            'score': vm_state['score'],
                            'priority': vm_state['priority']
                        })
                except Exception as e:
                    # Skip this timestamp if there's an error
                    continue
                
                eval_progress.empty()
                eval_status_detail.empty()
                eval_status.success(f"✅ PPO evaluation completed: {len(ppo_results)} requests evaluated")
            
            # Compare
            comp_status = st.empty()
            comp_status.info("🔄 Comparing algorithms...")
            
            if rr_results and len(rr_results) > 0 and ppo_results and len(ppo_results) > 0:
                try:
                    # Calculate PPO metrics using CBWO calculator
                    from cbwo_metrics import CBWOMetricsCalculator
                    ppo_metrics_calc = CBWOMetricsCalculator(df, ppo_results)
                    ppo_metrics = ppo_metrics_calc.get_all_metrics(power_coefficient=50.0, algorithm_type="ppo")
                    
                    comparison_analyzer = ComparisonAnalyzer(df)
                    comparison = comparison_analyzer.compare_algorithms(
                        rr_results, ppo_results, 
                        algo1_name="Round Robin", 
                        algo2_name="PPO (Proposed)"
                    )
                    
                    st.session_state['comparison'] = comparison
                    st.session_state['rr_results'] = rr_results
                    st.session_state['ppo_results'] = ppo_results
                    st.session_state['rr_metrics'] = rr_metrics
                    st.session_state['ppo_metrics'] = ppo_metrics
                    st.session_state['vm_selections'] = vm_selections
                    
                    comp_status.success(f"✅ Comparison completed! Round Robin: {len(rr_results)} requests, PPO: {len(ppo_results)} requests.")
                except Exception as e:
                    comp_status.error(f"❌ Error during comparison: {str(e)}")
                    st.exception(e)
            else:
                error_msg = []
                if not rr_results or len(rr_results) == 0:
                    error_msg.append("❌ No Round Robin results generated. Please check the time range.")
                if not ppo_results or len(ppo_results) == 0:
                    error_msg.append("❌ No PPO results generated. This might be due to:")
                    error_msg.append("  - No requests found in the selected time range")
                    error_msg.append("  - PPO evaluation failed")
                    error_msg.append("  - Try reducing the time range or increasing training steps")
                
                for msg in error_msg:
                    st.warning(msg)
        except Exception as e:
            st.error(f"❌ Error running comparison: {str(e)}")
            st.exception(e)
    
    # Display comparison results
    if 'comparison' in st.session_state and 'rr_metrics' in st.session_state:
        try:
            comparison = st.session_state['comparison']
            rr_metrics = st.session_state['rr_metrics']
            vm_selections = st.session_state.get('vm_selections', {})
            
            st.subheader("📊 Performance Comparison - CBWO Paper-Compliant Metrics")
            
            # Metrics comparison table
            metrics_df = pd.DataFrame(comparison['metrics']).T
            # Format the metrics for better display
            formatted_metrics = metrics_df.copy()
            for col in formatted_metrics.columns:
                if 'Makespan' in col or 'Time' in col or 'Energy' in col or 'Execution' in col:
                    formatted_metrics[col] = formatted_metrics[col].apply(lambda x: f"{x:.2f}")
                elif 'Resource Utilization' in col:
                    formatted_metrics[col] = formatted_metrics[col].apply(lambda x: f"{x:.2f}%")
                elif 'Degree of Imbalance' in col:
                    formatted_metrics[col] = formatted_metrics[col].apply(lambda x: f"{x:.4f}")
            
            st.dataframe(formatted_metrics, use_container_width=True)
            
            # Improvement metrics
            st.subheader("📈 PPO (Proposed) Improvement over Round Robin")
            improvement = comparison.get('improvement', {})
        
            col1, col2, col3 = st.columns(3)
            with col1:
                makespan_improvement = improvement.get('Makespan', 0)
                st.metric(
                    "Makespan Reduction",
                    f"{makespan_improvement:+.2f}%",
                    delta=f"{makespan_improvement:+.2f}%"
                )
            with col2:
                energy_improvement = improvement.get('Energy Consumption', 0)
                st.metric(
                    "Energy Consumption Reduction",
                    f"{energy_improvement:+.2f}%",
                    delta=f"{energy_improvement:+.2f}%"
                )
            with col3:
                doi_improvement = improvement.get('Degree of Imbalance', 0)
                st.metric(
                    "Degree of Imbalance Reduction",
                    f"{doi_improvement:+.2f}%",
                    delta=f"{doi_improvement:+.2f}%"
                )
            
            col4, col5 = st.columns(2)
            with col4:
                task_time_improvement = improvement.get('Task Completion Time', 0)
                st.metric(
                    "Task Completion Time Reduction",
                    f"{task_time_improvement:+.2f}%",
                    delta=f"{task_time_improvement:+.2f}%"
                )
            with col5:
                resource_improvement = improvement.get('Resource Utilization', 0)
                st.metric(
                    "Resource Utilization Reduction",
                    f"{resource_improvement:+.2f}%",
                    delta=f"{resource_improvement:+.2f}%"
                )
        
            # Side-by-side VM distribution
            st.subheader("VM Distribution Comparison")
            try:
                comparison_analyzer = ComparisonAnalyzer(df)
                rr_dist = rr_metrics.get('vm_distribution', {})
                if rr_dist and vm_selections:
                    dist_comparison = comparison_analyzer.get_vm_distribution_comparison(
                        rr_dist,
                        vm_selections,
                        algo1_name="Round Robin",
                        algo2_name="PPO (Proposed)"
                    )
                    
                    if len(dist_comparison) > 0:
                        fig_dist_comp = go.Figure()
                        fig_dist_comp.add_trace(go.Bar(
                            x=dist_comparison['VM'],
                            y=dist_comparison['Round Robin'],
                            name='Round Robin',
                            marker_color='lightgreen'
                        ))
                        fig_dist_comp.add_trace(go.Bar(
                            x=dist_comparison['VM'],
                            y=dist_comparison['PPO'],
                            name='PPO',
                            marker_color='lightcoral'
                        ))
                        fig_dist_comp.update_layout(
                            title="VM Request Distribution: Round Robin vs PPO",
                            xaxis_title="Virtual Machine",
                            yaxis_title="Request Count",
                            barmode='group'
                        )
                        st.plotly_chart(fig_dist_comp, use_container_width=True)
                    else:
                        st.warning("No VM distribution data available for comparison.")
                else:
                    st.warning("Missing VM distribution data.")
            except Exception as e:
                st.warning(f"Could not generate VM distribution comparison: {str(e)}")
        
            # CBWO Paper-Compliant Graphs
            st.subheader("📈 Performance Metrics Visualization")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Makespan Comparison
                makespan_rr = rr_metrics.get('Makespan', 0)
                makespan_ppo = st.session_state.get('ppo_metrics', {}).get('Makespan', 0)
                fig_makespan = go.Figure()
                fig_makespan.add_trace(go.Bar(
                    x=['Round Robin', 'PPO (Proposed)'],
                    y=[makespan_rr, makespan_ppo],
                    marker_color=['lightgreen', 'lightcoral'],
                    text=[f"{makespan_rr:.2f}", f"{makespan_ppo:.2f}"],
                    textposition='auto'
                ))
                fig_makespan.update_layout(
                    title="Makespan vs Algorithm",
                    xaxis_title="Algorithm",
                    yaxis_title="Makespan (seconds)"
                )
                st.plotly_chart(fig_makespan, use_container_width=True)
            
            with col2:
                # Energy Consumption Comparison
                energy_rr = rr_metrics.get('Energy Consumption', 0)
                energy_ppo = st.session_state.get('ppo_metrics', {}).get('Energy Consumption', 0)
                fig_energy = go.Figure()
                fig_energy.add_trace(go.Bar(
                    x=['Round Robin', 'PPO (Proposed)'],
                    y=[energy_rr, energy_ppo],
                    marker_color=['lightgreen', 'lightcoral'],
                    text=[f"{energy_rr:.2f}", f"{energy_ppo:.2f}"],
                    textposition='auto'
                ))
                fig_energy.update_layout(
                    title="Energy Consumption vs Algorithm",
                    xaxis_title="Algorithm",
                    yaxis_title="Energy Consumption (Joules)"
                )
                st.plotly_chart(fig_energy, use_container_width=True)
            
            col3, col4 = st.columns(2)
            
            with col3:
                # Resource Utilization Comparison
                util_rr = rr_metrics.get('Resource Utilization', 0)
                util_ppo = st.session_state.get('ppo_metrics', {}).get('Resource Utilization', 0)
                fig_resource = go.Figure()
                fig_resource.add_trace(go.Bar(
                    x=['Round Robin', 'PPO (Proposed)'],
                    y=[util_rr, util_ppo],
                    marker_color=['lightgreen', 'lightcoral'],
                    text=[f"{util_rr:.2f}%", f"{util_ppo:.2f}%"],
                    textposition='auto'
                ))
                fig_resource.update_layout(
                    title="Resource Utilization vs Algorithm",
                    xaxis_title="Algorithm",
                    yaxis_title="Resource Utilization (%)"
                )
                st.plotly_chart(fig_resource, use_container_width=True)
            
            with col4:
                # Degree of Imbalance Comparison
                doi_rr = rr_metrics.get('Degree of Imbalance', 0)
                doi_ppo = st.session_state.get('ppo_metrics', {}).get('Degree of Imbalance', 0)
                fig_doi = go.Figure()
                fig_doi.add_trace(go.Bar(
                    x=['Round Robin', 'PPO (Proposed)'],
                    y=[doi_rr, doi_ppo],
                    marker_color=['lightgreen', 'lightcoral'],
                    text=[f"{doi_rr:.4f}", f"{doi_ppo:.4f}"],
                    textposition='auto'
                ))
                fig_doi.update_layout(
                    title="Degree of Imbalance vs Algorithm",
                    xaxis_title="Algorithm",
                    yaxis_title="Degree of Imbalance"
                )
                st.plotly_chart(fig_doi, use_container_width=True)
            
            # Task Completion Time Comparison
            task_rr = rr_metrics.get('Task Completion Time', 0)
            task_ppo = st.session_state.get('ppo_metrics', {}).get('Task Completion Time', 0)
            fig_task = go.Figure()
            fig_task.add_trace(go.Bar(
                x=['Round Robin', 'PPO (Proposed)'],
                y=[task_rr, task_ppo],
                marker_color=['lightgreen', 'lightcoral'],
                text=[f"{task_rr:.2f}", f"{task_ppo:.2f}"],
                textposition='auto'
            ))
            fig_task.update_layout(
                title="Task Completion Time vs Algorithm",
                xaxis_title="Algorithm",
                yaxis_title="Task Completion Time (seconds)"
            )
            st.plotly_chart(fig_task, use_container_width=True)
            
            # Summary insights
            st.subheader("💡 Key Insights")
            
            improvement = comparison.get('improvement', {})
            insights = []
            
            if 'Makespan' in improvement:
                if improvement['Makespan'] > 0:
                    insights.append(f"✅ PPO (Proposed) reduces Makespan by {improvement['Makespan']:.2f}%")
                else:
                    insights.append(f"⚠️ Round Robin achieves {-improvement['Makespan']:.2f}% lower Makespan")
            
            if 'Energy Consumption' in improvement:
                if improvement['Energy Consumption'] > 0:
                    insights.append(f"✅ PPO (Proposed) reduces Energy Consumption by {improvement['Energy Consumption']:.2f}%")
                else:
                    insights.append(f"⚠️ Round Robin achieves {-improvement['Energy Consumption']:.2f}% lower Energy Consumption")
            
            if 'Degree of Imbalance' in improvement:
                if improvement['Degree of Imbalance'] > 0:
                    insights.append(f"✅ PPO (Proposed) improves load balance (reduces DoI by {improvement['Degree of Imbalance']:.2f}%)")
                else:
                    insights.append(f"⚠️ Round Robin achieves better load balance")
            
            if 'Task Completion Time' in improvement:
                if improvement['Task Completion Time'] > 0:
                    insights.append(f"✅ PPO (Proposed) reduces Task Completion Time by {improvement['Task Completion Time']:.2f}%")
                else:
                    insights.append(f"⚠️ Round Robin achieves {-improvement['Task Completion Time']:.2f}% lower Task Completion Time")
            
            if 'Resource Utilization' in improvement:
                if improvement['Resource Utilization'] > 0:
                    insights.append(f"✅ PPO (Proposed) reduces Resource Utilization by {improvement['Resource Utilization']:.2f}% (more efficient)")
                else:
                    insights.append(f"⚠️ Round Robin achieves {-improvement['Resource Utilization']:.2f}% lower Resource Utilization")
            
            for insight in insights:
                st.markdown(f"- {insight}")
            
            # Export Section
            st.subheader("📥 Export Results")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📄 Export CSV"):
                    try:
                        csv_data = export_comparison_to_csv(comparison)
                        st.download_button(
                            label="Download Comparison CSV",
                            data=csv_data,
                            file_name=f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    except Exception as e:
                        st.error(f"Error exporting CSV: {str(e)}")
            
            with col2:
                if st.button("📊 Export PDF Report"):
                    try:
                        algorithms_metrics = {
                            'Round Robin': rr_metrics,
                            'PPO (Proposed)': st.session_state.get('ppo_metrics', {})
                        }
                        pdf_buffer = generate_comparison_report(comparison, algorithms_metrics, 'pdf')
                        st.download_button(
                            label="Download PDF Report",
                            data=pdf_buffer.getvalue(),
                            file_name=f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )
                    except Exception as e:
                        st.error(f"Error exporting PDF: {str(e)}")
            
            with col3:
                if st.button("🌐 Export HTML Report"):
                    try:
                        algorithms_metrics = {
                            'Round Robin': rr_metrics,
                            'PPO (Proposed)': st.session_state.get('ppo_metrics', {})
                        }
                        html_content = generate_comparison_report(comparison, algorithms_metrics, 'html')
                        st.download_button(
                            label="Download HTML Report",
                            data=html_content,
                            file_name=f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                            mime="text/html"
                        )
                    except Exception as e:
                        st.error(f"Error exporting HTML: {str(e)}")
                        
        except Exception as e:
            st.error(f"Error displaying comparison results: {str(e)}")
            st.exception(e)

def show_multi_algorithm_comparison(data_dict):
    st.header("📊 Multi-Algorithm Comparison")
    
    st.markdown("""
    ### Compare All Load Balancing Algorithms
    
    This page allows you to compare multiple load balancing algorithms simultaneously:
    - **PPO** (Proximal Policy Optimization - DRL)
    - **Round Robin** (Cycles through VMs)
    - **CBWO** (Chaos-Based Black Widow Optimization)
    """)
    
    selected_dataset = st.selectbox("Select Dataset", list(data_dict.keys()), key="multi_comp_dataset")
    df = data_dict[selected_dataset]
    
    # Algorithm selection
    st.subheader("Algorithm Selection")
    selected_algorithms = st.multiselect(
        "Select algorithms to compare",
        ["PPO", "Round Robin", "CBWO"],
        default=["PPO", "Round Robin", "CBWO"],
        key="multi_algo_select"
    )
    
    if not selected_algorithms:
        st.warning("Please select at least one algorithm to compare.")
        return
    
    # Configuration
    st.subheader("Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Time Range**")
        start_time = st.date_input("Start Date", value=df['timestamp'].min().date(), key="multi_start")
        start_datetime = pd.Timestamp.combine(start_time, df['timestamp'].min().time())
        end_time = st.date_input("End Date", value=df['timestamp'].max().date(), key="multi_end")
        end_datetime = pd.Timestamp.combine(end_time, df['timestamp'].max().time())
    
    with col2:
        st.markdown("**PPO Training Parameters** (if PPO is selected)")
        ppo_learning_rate = st.slider("Learning Rate", 0.0001, 0.01, 0.001, 0.0001, key="multi_lr")
        ppo_steps = st.slider("Training Steps", 100, 1000, 300, 100, key="multi_steps")
        ppo_batch_size = st.slider("Batch Size", 32, 128, 64, 32, key="multi_batch")
    
    if st.button("🚀 Run Multi-Algorithm Comparison", type="primary"):
        results_dict = {}
        metrics_dict = {}
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_algorithms = len(selected_algorithms)
        
        try:
            for idx, algo_name in enumerate(selected_algorithms):
                progress = (idx + 1) / total_algorithms
                progress_bar.progress(progress)
                status_text.text(f"Running {algo_name}... ({idx + 1}/{total_algorithms})")
                
                if algo_name == "PPO":
                    env = create_environment(df)
                    ppo_balancer = PPOLoadBalancer(
                        state_dim=env.observation_space.shape[0],
                        action_dim=env.action_space.n,
                        learning_rate=ppo_learning_rate
                    )
                    
                    # Quick training
                    for step in range(min(ppo_steps, 200)):
                        state = env.reset()
                        done = False
                        episode_steps = 0
                        while not done and episode_steps < 50:
                            action = ppo_balancer.select_action(state)
                            next_state, reward, done, _ = env.step(action)
                            ppo_balancer.store_transition(state, action, reward, next_state, done)
                            state = next_state
                            episode_steps += 1
                        
                        if len(ppo_balancer.buffer) >= ppo_batch_size:
                            ppo_balancer.update(n_epochs=4, batch_size=ppo_batch_size)
                    
                    # Evaluate
                    requests = df[
                        (df['timestamp'] >= start_datetime) &
                        (df['timestamp'] <= end_datetime) &
                        (df['update'] == 1)
                    ]
                    
                    if len(requests) > 0:
                        request_timestamps = requests['timestamp'].unique()[:50].tolist()
                        results = []
                        for timestamp in request_timestamps:
                            time_data = df[df['timestamp'] <= timestamp].tail(1000)
                            state_vector = []
                            for vm_name in df['vm_name'].unique():
                                vm_data = time_data[time_data['vm_name'] == vm_name].tail(1)
                                if len(vm_data) > 0:
                                    latest = vm_data.iloc[0]
                                    state_vector.extend([
                                        latest['cpu_usage'],
                                        latest['mem_usage'] / latest['max_mem'] if latest['max_mem'] > 0 else 0,
                                        latest['bw_usage'] / latest['max_bw'] if latest['max_bw'] > 0 else 0,
                                        latest['score'] / 10.0,
                                        latest['priority'] / 4.0,
                                        0.5
                                    ])
                                else:
                                    state_vector.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                            
                            state_array = np.array(state_vector, dtype=np.float32)
                            action = ppo_balancer.select_action(state_array, deterministic=True)
                            selected_vm = df['vm_name'].unique()[action]
                            
                            vm_data = time_data[time_data['vm_name'] == selected_vm].tail(1)
                            if len(vm_data) > 0:
                                vm_state = vm_data.iloc[0]
                                cpu_reward = (1.0 - vm_state['cpu_usage']) * 0.3
                                mem_usage_norm = vm_state['mem_usage'] / vm_state['max_mem'] if vm_state['max_mem'] > 0 else 0
                                mem_reward = (1.0 - mem_usage_norm) * 0.2
                                bw_usage_norm = vm_state['bw_usage'] / vm_state['max_bw'] if vm_state['max_bw'] > 0 else 0
                                bw_reward = (1.0 - bw_usage_norm) * 0.2
                                priority_reward = (1.0 - vm_state['priority'] / 4.0) * 0.2
                                reward = cpu_reward + mem_reward + bw_reward + priority_reward
                                
                                results.append({
                                    'timestamp': timestamp,
                                    'selected_vm': selected_vm,
                                    'reward': reward,
                                    'cpu_usage': vm_state['cpu_usage'],
                                    'mem_usage': mem_usage_norm,
                                    'bw_usage': bw_usage_norm,
                                    'score': vm_state['score'],
                                    'priority': vm_state['priority']
                                })
                        
                        # Use CBWO metrics calculator for PPO
                        metrics_calc = CBWOMetricsCalculator(df, results)
                        metrics = metrics_calc.get_all_metrics(power_coefficient=50.0, algorithm_type="ppo")
                        # Create DataFrame for VM distribution calculation
                        results_df = pd.DataFrame(results)
                        if len(results_df) > 0 and 'selected_vm' in results_df.columns:
                            metrics['vm_distribution'] = dict(results_df['selected_vm'].value_counts())
                        else:
                            metrics['vm_distribution'] = {}
                        metrics['total_requests'] = len(results)
                    else:
                        results = []
                        metrics = {}
                        
                elif algo_name == "Round Robin":
                    balancer = RoundRobinLoadBalancer(df)
                    results = balancer.simulate_load_balancing(start_datetime, end_datetime)
                    metrics = balancer.get_performance_metrics(power_coefficient=50.0)
                
                elif algo_name == "CBWO":
                    balancer = CBWOLoadBalancer(df)
                    results = balancer.simulate_load_balancing(start_datetime, end_datetime)
                    metrics = balancer.get_performance_metrics(power_coefficient=50.0)
                
                results_dict[algo_name] = results
                metrics_dict[algo_name] = metrics
            
            progress_bar.empty()
            status_text.empty()
            st.success(f"✅ Comparison completed for {len(selected_algorithms)} algorithms!")
            
            # Store in session state
            st.session_state['multi_results'] = results_dict
            st.session_state['multi_metrics'] = metrics_dict
            
        except Exception as e:
            st.error(f"❌ Error running comparison: {str(e)}")
            st.exception(e)
    
    # Display results
    if 'multi_results' in st.session_state and 'multi_metrics' in st.session_state:
        results_dict = st.session_state['multi_results']
        metrics_dict = st.session_state['multi_metrics']
        
        st.subheader("📊 Performance Comparison")
        
        # Create comparison DataFrame with CBWO paper-compliant metrics
        comparison_data = []
        for algo_name, metrics in metrics_dict.items():
            comparison_data.append({
                'Algorithm': algo_name,
                'Makespan (s)': f"{metrics.get('Makespan', 0):.2f}",
                'Task Completion Time (s)': f"{metrics.get('Task Completion Time', 0):.2f}",
                'Resource Utilization (%)': f"{metrics.get('Resource Utilization', 0):.2f}",
                'Degree of Imbalance': f"{metrics.get('Degree of Imbalance', 0):.4f}",
                'Energy Consumption (J)': f"{metrics.get('Energy Consumption', 0):.2f}",
                'Execution Time (s)': f"{metrics.get('Execution Time', 0):.2f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Visualizations - CBWO Paper-Compliant Graphs Only
        st.subheader("📈 Performance Metrics Visualization")
        
        algo_names = list(metrics_dict.keys())
        
        # Graph 1: Makespan vs Algorithm
        col1, col2 = st.columns(2)
        with col1:
            makespans = [metrics_dict[algo].get('Makespan', 0) for algo in algo_names]
            fig_makespan = go.Figure()
            fig_makespan.add_trace(go.Bar(
                x=algo_names,
                y=makespans,
                marker_color='lightblue',
                text=[f"{m:.2f}" for m in makespans],
                textposition='auto'
            ))
            fig_makespan.update_layout(
                title="Makespan vs Algorithm",
                xaxis_title="Algorithm",
                yaxis_title="Makespan (seconds)"
            )
            st.plotly_chart(fig_makespan, use_container_width=True)
        
        # Graph 2: Energy Consumption vs Algorithm
        with col2:
            energy = [metrics_dict[algo].get('Energy Consumption', 0) for algo in algo_names]
            fig_energy = go.Figure()
            fig_energy.add_trace(go.Bar(
                x=algo_names,
                y=energy,
                marker_color='lightcoral',
                text=[f"{e:.2f}" for e in energy],
                textposition='auto'
            ))
            fig_energy.update_layout(
                title="Energy Consumption vs Algorithm",
                xaxis_title="Algorithm",
                yaxis_title="Energy Consumption (Joules)"
            )
            st.plotly_chart(fig_energy, use_container_width=True)
        
        # Graph 3: Resource Utilization vs Algorithm
        col3, col4 = st.columns(2)
        with col3:
            resource_util = [metrics_dict[algo].get('Resource Utilization', 0) for algo in algo_names]
            fig_resource = go.Figure()
            fig_resource.add_trace(go.Bar(
                x=algo_names,
                y=resource_util,
                marker_color='lightgreen',
                text=[f"{r:.2f}%" for r in resource_util],
                textposition='auto'
            ))
            fig_resource.update_layout(
                title="Resource Utilization vs Algorithm",
                xaxis_title="Algorithm",
                yaxis_title="Resource Utilization (%)"
            )
            st.plotly_chart(fig_resource, use_container_width=True)
        
        # Graph 4: Degree of Imbalance vs Algorithm
        with col4:
            doi = [metrics_dict[algo].get('Degree of Imbalance', 0) for algo in algo_names]
            fig_doi = go.Figure()
            fig_doi.add_trace(go.Bar(
                x=algo_names,
                y=doi,
                marker_color='lightyellow',
                text=[f"{d:.4f}" for d in doi],
                textposition='auto'
            ))
            fig_doi.update_layout(
                title="Degree of Imbalance vs Algorithm",
                xaxis_title="Algorithm",
                yaxis_title="Degree of Imbalance"
            )
            st.plotly_chart(fig_doi, use_container_width=True)
        
        # Graph 5: Task Completion Time vs Algorithm
        st.subheader("Task Completion Time Comparison")
        task_times = [metrics_dict[algo].get('Task Completion Time', 0) for algo in algo_names]
        fig_task = go.Figure()
        fig_task.add_trace(go.Bar(
            x=algo_names,
            y=task_times,
            marker_color='lightpink',
            text=[f"{t:.2f}" for t in task_times],
            textposition='auto'
        ))
        fig_task.update_layout(
            title="Task Completion Time vs Algorithm",
            xaxis_title="Algorithm",
            yaxis_title="Task Completion Time (seconds)"
        )
        st.plotly_chart(fig_task, use_container_width=True)
        
        # VM Distribution Comparison
        st.subheader("VM Distribution Comparison")
        fig_dist = go.Figure()
        
        colors_map = {
            'PPO': 'lightcoral',
            'Round Robin': 'lightgreen',
            'CBWO': 'lightblue'
        }
        
        for algo_name in algo_names:
            vm_dist = metrics_dict[algo_name].get('vm_distribution', {})
            if vm_dist:
                fig_dist.add_trace(go.Bar(
                    x=list(vm_dist.keys()),
                    y=list(vm_dist.values()),
                    name=algo_name,
                    marker_color=colors_map.get(algo_name, 'lightblue')
                ))
        
        fig_dist.update_layout(
            title="VM Request Distribution by Algorithm",
            xaxis_title="Virtual Machine",
            yaxis_title="Request Count",
            barmode='group'
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Export Section
        st.subheader("📥 Export Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_data = comparison_df.to_csv(index=False)
            st.download_button(
                label="📄 Download CSV",
                data=csv_data,
                file_name=f"multi_algorithm_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            try:
                # Create comparison data structure for PDF
                comparison_for_pdf = {
                    'metrics': {algo: {
                        'Makespan': metrics_dict[algo].get('Makespan', 0),
                        'Task Completion Time': metrics_dict[algo].get('Task Completion Time', 0),
                        'Resource Utilization': metrics_dict[algo].get('Resource Utilization', 0),
                        'Degree of Imbalance': metrics_dict[algo].get('Degree of Imbalance', 0),
                        'Energy Consumption': metrics_dict[algo].get('Energy Consumption', 0),
                        'Execution Time': metrics_dict[algo].get('Execution Time', 0)
                    } for algo in algo_names}
                }
                
                pdf_buffer = generate_comparison_report(comparison_for_pdf, metrics_dict, 'pdf')
                st.download_button(
                    label="📊 Download PDF Report",
                    data=pdf_buffer.getvalue(),
                    file_name=f"multi_algorithm_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
            except Exception as e:
                st.error(f"PDF export error: {str(e)}")
        
        with col3:
            try:
                comparison_for_html = {
                    'metrics': {algo: {
                        'Makespan': metrics_dict[algo].get('Makespan', 0),
                        'Task Completion Time': metrics_dict[algo].get('Task Completion Time', 0),
                        'Resource Utilization': metrics_dict[algo].get('Resource Utilization', 0),
                        'Degree of Imbalance': metrics_dict[algo].get('Degree of Imbalance', 0),
                        'Energy Consumption': metrics_dict[algo].get('Energy Consumption', 0),
                        'Execution Time': metrics_dict[algo].get('Execution Time', 0)
                    } for algo in algo_names}
                }
                
                html_content = generate_comparison_report(comparison_for_html, metrics_dict, 'html')
                st.download_button(
                    label="🌐 Download HTML Report",
                    data=html_content,
                    file_name=f"multi_algorithm_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime="text/html"
                )
            except Exception as e:
                st.error(f"HTML export error: {str(e)}")

if __name__ == "__main__":
    main()

