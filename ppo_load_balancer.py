import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque

# Handle both gym and gymnasium
try:
    import gym
    from gym import spaces
except ImportError:
    try:
        import gymnasium as gym
        from gymnasium import spaces
    except ImportError:
        raise ImportError("Please install either 'gym' or 'gymnasium' package")

class PPONetwork(nn.Module):
    """Neural network for PPO policy and value estimation"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PPONetwork, self).__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Policy head
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Value head
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state):
        shared_out = self.shared(state)
        policy_out = self.policy(shared_out)
        value_out = self.value(shared_out)
        return policy_out, value_out

class PPOLoadBalancer:
    """PPO-based Load Balancer"""
    
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.95, clip_range=0.2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_range = clip_range
        
        # Initialize networks
        self.network = PPONetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.buffer = deque(maxlen=10000)
        
        # Training statistics
        self.training_losses = []
    
    def select_action(self, state, deterministic=False):
        """Select action using current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            policy, _ = self.network(state_tensor)
            dist = Categorical(policy)
            
            if deterministic:
                action = torch.argmax(policy, dim=1).item()
            else:
                action = dist.sample().item()
        
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })
    
    def compute_returns(self, rewards, dones):
        """Compute discounted returns"""
        returns = []
        G = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                G = 0
            G = reward + self.gamma * G
            returns.insert(0, G)
        return returns
    
    def update(self, n_epochs=4, batch_size=128):
        """Update policy using PPO algorithm"""
        if len(self.buffer) < batch_size:
            return 0.0
        
        # Sample batch
        batch_indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in batch_indices]
        
        states = torch.FloatTensor([t['state'] for t in batch])
        actions = torch.LongTensor([t['action'] for t in batch])
        rewards = [t['reward'] for t in batch]
        next_states = torch.FloatTensor([t['next_state'] for t in batch])
        dones = [t['done'] for t in batch]
        
        # Compute returns
        returns = torch.FloatTensor(self.compute_returns(rewards, dones))
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # Normalize
        
        total_loss = 0.0
        
        for epoch in range(n_epochs):
            # Get current policy and value
            policy, values = self.network(states)
            dist = Categorical(policy)
            
            # Compute old log probabilities (for clipping)
            old_log_probs = dist.log_prob(actions).detach()
            
            # Compute advantages
            advantages = returns - values.squeeze()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Compute new log probabilities
            new_log_probs = dist.log_prob(actions)
            
            # Compute ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Compute clipped objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = nn.MSELoss()(values.squeeze(), returns)
            
            # Entropy bonus (encourages exploration)
            entropy = dist.entropy().mean()
            
            # Total loss
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / n_epochs
        self.training_losses.append(avg_loss)
        
        return avg_loss

class LoadBalancingEnv(gym.Env):
    """Gym environment for load balancing"""
    
    def __init__(self, df):
        super(LoadBalancingEnv, self).__init__()
        
        self.df = df.copy()
        self.vm_names = df['vm_name'].unique()
        self.n_vms = len(self.vm_names)
        
        # Normalize timestamps
        self.df['timestamp_norm'] = (self.df['timestamp'] - self.df['timestamp'].min()).dt.total_seconds()
        self.max_time = self.df['timestamp_norm'].max()
        
        # Action space: select which VM to route request to
        self.action_space = spaces.Discrete(self.n_vms)
        
        # State space: normalized resource metrics
        # [cpu_usage, mem_usage, bw_usage, score, priority, time_normalized]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_vms * 6,),  # 6 features per VM
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self):
        """Reset environment to initial state"""
        self.current_time_idx = 0
        self.current_time = 0.0
        
        # Get initial state
        self.current_state = self._get_state()
        
        return self.current_state
    
    def _get_state(self):
        """Get current state vector"""
        # Get data at current time
        time_data = self.df[
            (self.df['timestamp_norm'] >= self.current_time) &
            (self.df['timestamp_norm'] < self.current_time + 1)
        ]
        
        if len(time_data) == 0:
            # Use last available data
            time_data = self.df[self.df['timestamp_norm'] <= self.current_time].tail(self.n_vms)
        
        state = []
        
        for vm_name in self.vm_names:
            vm_data = time_data[time_data['vm_name'] == vm_name]
            
            if len(vm_data) > 0:
                # Use latest data for this VM
                latest = vm_data.iloc[-1]
                state.extend([
                    latest['cpu_usage'],
                    latest['mem_usage'] / latest['max_mem'] if latest['max_mem'] > 0 else 0,
                    latest['bw_usage'] / latest['max_bw'] if latest['max_bw'] > 0 else 0,
                    latest['score'] / 10.0,  # Normalize score
                    latest['priority'] / 4.0,  # Normalize priority
                    self.current_time / self.max_time if self.max_time > 0 else 0
                ])
            else:
                # Default state if no data
                state.extend([0.0, 0.0, 0.0, 0.0, 0.0, self.current_time / self.max_time if self.max_time > 0 else 0])
        
        return np.array(state, dtype=np.float32)
    
    def step(self, action):
        """Execute one step in the environment"""
        # Get VM name for selected action
        selected_vm = self.vm_names[action]
        
        # Get current VM states
        time_data = self.df[
            (self.df['timestamp_norm'] >= self.current_time) &
            (self.df['timestamp_norm'] < self.current_time + 1)
        ]
        
        if len(time_data) == 0:
            time_data = self.df[self.df['timestamp_norm'] <= self.current_time].tail(self.n_vms)
        
        # Calculate reward based on load balancing objectives
        reward = self._calculate_reward(selected_vm, time_data)
        
        # Update time
        self.current_time += 1.0
        self.current_time_idx += 1
        
        # Check if done
        done = self.current_time >= self.max_time
        
        # Get next state
        next_state = self._get_state()
        self.current_state = next_state
        
        info = {'selected_vm': selected_vm}
        
        return next_state, reward, done, info
    
    def _calculate_reward(self, selected_vm, time_data):
        """Calculate reward for selecting a VM"""
        vm_data = time_data[time_data['vm_name'] == selected_vm]
        
        if len(vm_data) == 0:
            return -1.0  # Penalty for selecting unavailable VM
        
        latest = vm_data.iloc[-1]
        
        # Reward components
        # 1. Prefer lower CPU usage
        cpu_reward = (1.0 - latest['cpu_usage']) * 0.3
        
        # 2. Prefer lower memory usage
        mem_usage_norm = latest['mem_usage'] / latest['max_mem'] if latest['max_mem'] > 0 else 0
        mem_reward = (1.0 - mem_usage_norm) * 0.2
        
        # 3. Prefer lower bandwidth usage
        bw_usage_norm = latest['bw_usage'] / latest['max_bw'] if latest['max_bw'] > 0 else 0
        bw_reward = (1.0 - bw_usage_norm) * 0.2
        
        # 4. Prefer higher priority (lower priority number = higher priority)
        priority_reward = (1.0 - latest['priority'] / 4.0) * 0.2
        
        # 5. Load balancing: penalize if this VM is overloaded compared to others
        if len(time_data) > 0:
            avg_cpu = time_data['cpu_usage'].mean()
            cpu_balance_penalty = abs(latest['cpu_usage'] - avg_cpu) * 0.1
        else:
            cpu_balance_penalty = 0.0
        
        total_reward = cpu_reward + mem_reward + bw_reward + priority_reward - cpu_balance_penalty
        
        return total_reward

def create_environment(df):
    """Create and return a load balancing environment"""
    return LoadBalancingEnv(df)

