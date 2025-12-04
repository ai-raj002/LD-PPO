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

# Page configuration
st.set_page_config(
    page_title="Load Balancing with DRL (PPO)",
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
    st.markdown('<h1 class="main-header">⚖️ Load Balancing Analysis with DRL (PPO)</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["📊 Data Overview", "📈 Performance Analysis", "🤖 PPO Load Balancer", "🔍 Deep Insights"]
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
        metric_type = st.selectbox("Select Metric", ["CPU Usage", "Memory Usage", "Network Bandwidth", "Score", "Priority"])
    
    df_filtered = df[df['vm_name'].isin(selected_vms)]
    
    # Time series plots
    st.subheader("Time Series Analysis")
    
    metric_map = {
        "CPU Usage": "cpu_usage",
        "Memory Usage": "mem_usage",
        "Network Bandwidth": "bw_usage",
        "Score": "score",
        "Priority": "priority"
    }
    
    metric_col = metric_map[metric_type]
    
    fig = px.line(
        df_filtered,
        x='timestamp',
        y=metric_col,
        color='vm_name',
        title=f"{metric_type} Over Time",
        labels={metric_col: metric_type, 'timestamp': 'Time'}
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
        avg_mem = df_filtered.groupby('vm_name')['mem_usage'].mean()
        fig_mem = px.bar(
            x=avg_mem.index,
            y=avg_mem.values,
            labels={'x': 'VM', 'y': 'Average Memory Usage'},
            title="Average Memory Usage by VM"
        )
        st.plotly_chart(fig_mem, use_container_width=True)
    
    with col3:
        avg_score = df_filtered.groupby('vm_name')['score'].mean()
        fig_score = px.bar(
            x=avg_score.index,
            y=avg_score.values,
            labels={'x': 'VM', 'y': 'Average Score'},
            title="Average Load Balancing Score by VM"
        )
        st.plotly_chart(fig_score, use_container_width=True)
    
    # Heatmap
    st.subheader("Resource Usage Heatmap")
    df_pivot = df_filtered.pivot_table(
        values=['cpu_usage', 'mem_usage', 'bw_usage'],
        index='vm_name',
        aggfunc='mean'
    )
    fig_heat = px.imshow(
        df_pivot.T,
        labels=dict(x="VM", y="Resource", color="Usage"),
        title="Resource Usage Heatmap",
        aspect="auto"
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    
    # Load balancing efficiency
    st.subheader("Load Balancing Efficiency Metrics")
    analyzer = DataAnalyzer(df_filtered)
    efficiency_metrics = analyzer.calculate_efficiency_metrics()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Load Balance Index", f"{efficiency_metrics['balance_index']:.3f}")
    with col2:
        st.metric("CPU Variance", f"{efficiency_metrics['cpu_variance']:.4f}")
    with col3:
        st.metric("Memory Variance", f"{efficiency_metrics['mem_variance']:.4f}")
    with col4:
        st.metric("Overall Efficiency", f"{efficiency_metrics['overall_efficiency']:.2%}")

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
            fig_reward = px.line(
                history_df,
                x='step',
                y='reward',
                title="Training Reward Over Time",
                labels={'step': 'Training Step', 'reward': 'Episode Reward'}
            )
            st.plotly_chart(fig_reward, use_container_width=True)
        
        with col2:
            fig_loss = px.line(
                history_df,
                x='step',
                y='loss',
                title="PPO Loss Over Time",
                labels={'step': 'Training Step', 'loss': 'Loss'}
            )
            st.plotly_chart(fig_loss, use_container_width=True)
        
        # Evaluate model
        if st.button("📊 Evaluate Model Performance"):
            if 'ppo_model' in st.session_state and 'env' in st.session_state:
                ppo_model = st.session_state['ppo_model']
                env = st.session_state['env']
                
                # Run evaluation
                eval_rewards = []
                vm_selections = {vm: 0 for vm in df['vm_name'].unique()}
                
                for _ in range(100):
                    state = env.reset()
                    done = False
                    episode_reward = 0
                    
                    while not done:
                        action = ppo_model.select_action(state, deterministic=True)
                        vm_name = df['vm_name'].unique()[action]
                        vm_selections[vm_name] += 1
                        next_state, reward, done, _ = env.step(action)
                        state = next_state
                        episode_reward += reward
                    
                    eval_rewards.append(episode_reward)
                
                st.subheader("Evaluation Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Average Reward", f"{np.mean(eval_rewards):.2f}")
                    st.metric("Std Deviation", f"{np.std(eval_rewards):.2f}")
                
                with col2:
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

if __name__ == "__main__":
    main()

