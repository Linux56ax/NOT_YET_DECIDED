import networkx as nx
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque, namedtuple

class TrainPosition(Enum):
    AT_NODE = "at_node"
    ON_EDGE = "on_edge"

@dataclass
class TrainState:
    train_id: str
    position_type: TrainPosition
    
    # Position data
    current_node: Optional[str] = None  # If at node
    current_edge: Optional[Tuple[str, str]] = None  # If on edge (from, to)
    position_on_edge: float = 0.0  # meters from edge start
    
    # Train properties
    length: float = 200.0  # meters
    speed_kmh: float = 40.0  # kilometers per hour
    time_step_seconds: float = 5.0  # seconds per simulation step
    
    @property
    def speed_ms(self) -> float:
        """Speed in meters per second"""
        return self.speed_kmh / 3.6
    
    @property
    def distance_per_step(self) -> float:
        """Distance covered per simulation step in meters"""
        return self.speed_ms * self.time_step_seconds
    
    # Journey data
    destination: Optional[str] = None
    planned_route: List[str] = field(default_factory=list)
    route_index: int = 0
    
    def get_position_info(self) -> str:
        """Get readable position information"""
        if self.position_type == TrainPosition.AT_NODE:
            return f"At station/junction: {self.current_node}"
        else:
            from_node, to_node = self.current_edge
            return f"On track {from_node} → {to_node} at {self.position_on_edge:.1f}m"

@dataclass
class TrainOnEdge:
    """Represents a train positioned on an edge"""
    train_id: str
    position: float  # meters from edge start
    length: float    # train length in meters
    is_halted: bool = False
    halt_reason: str = ""
    
    @property
    def front_position(self) -> float:
        return self.position
    
    @property
    def rear_position(self) -> float:
        return max(0, self.position - self.length)

class GlobalStateManager:
    """Manages global state of all trains and edges in the railway system"""
    
    def __init__(self, network: nx.DiGraph):
        self.network = network
        
        # Safety constraints
        self.MIN_TRAIN_DISTANCE = 100.0  # meters minimum distance between trains
        self.NODE_SAFETY_DISTANCE = 25.0  # meters safety distance from nodes
        
        # Global state storage
        self.edge_occupancy: Dict[Tuple[str, str], List[TrainOnEdge]] = {}
        self.node_occupancy: Dict[str, List[str]] = {}  # node -> list of train_ids
        self.halted_trains: Dict[str, Dict] = {}  # train_id -> halt info
        
        self._initialize_state()
    
    def _initialize_state(self):
        """Initialize state for all edges and nodes"""
        for edge in self.network.edges():
            edge_key = (edge[0], edge[1])
            self.edge_occupancy[edge_key] = []
        
        for node in self.network.nodes():
            self.node_occupancy[node] = []
    
    def add_train_to_node(self, train_id: str, node: str):
        """Add train to a node"""
        if train_id not in self.node_occupancy[node]:
            self.node_occupancy[node].append(train_id)
    
    def remove_train_from_node(self, train_id: str, node: str):
        """Remove train from a node"""
        if train_id in self.node_occupancy[node]:
            self.node_occupancy[node].remove(train_id)
    
    def add_train_to_edge(self, train_id: str, edge: Tuple[str, str], position: float, length: float):
        """Add train to an edge"""
        train_on_edge = TrainOnEdge(train_id, position, length)
        self.edge_occupancy[edge].append(train_on_edge)
        self._sort_trains_on_edge(edge)
    
    def remove_train_from_edge(self, train_id: str, edge: Tuple[str, str]):
        """Remove train from an edge"""
        self.edge_occupancy[edge] = [
            train for train in self.edge_occupancy[edge] 
            if train.train_id != train_id
        ]
        if train_id in self.halted_trains:
            del self.halted_trains[train_id]
    
    def update_train_position_on_edge(self, train_id: str, edge: Tuple[str, str], new_position: float):
        """Update train's position on an edge"""
        for train in self.edge_occupancy[edge]:
            if train.train_id == train_id:
                train.position = new_position
                train.is_halted = False
                train.halt_reason = ""
                break
        # If the train was previously marked halted, clear it when it moves
        if train_id in self.halted_trains:
            del self.halted_trains[train_id]
        self._sort_trains_on_edge(edge)
    
    def _sort_trains_on_edge(self, edge: Tuple[str, str]):
        """Sort trains on edge by position (ascending)"""
        self.edge_occupancy[edge].sort(key=lambda t: t.position)
    
    def can_train_move(self, train_id: str, edge: Tuple[str, str], new_position: float, train_length: float) -> Tuple[bool, str, Optional[float]]:
        """
        Check if train can move to new position
        Returns: (can_move, reason, halt_position)
        """
        edge_data = self.network.get_edge_data(edge[0], edge[1])
        edge_length = edge_data.get('length', 0)
        
        # Check node safety distance
        if new_position + self.NODE_SAFETY_DISTANCE > edge_length:
            halt_pos = edge_length - self.NODE_SAFETY_DISTANCE
            return False, f"Node safety distance violation at {edge[1]}", max(0, halt_pos)
        
        # Check collision with other trains
        trains_on_edge = self.edge_occupancy[edge]
        current_train_idx = None
        
        # Find current train index
        for i, train in enumerate(trains_on_edge):
            if train.train_id == train_id:
                current_train_idx = i
                break
        
        # Check collision with train ahead
        for i, other_train in enumerate(trains_on_edge):
            if other_train.train_id == train_id:
                continue
            
            # Train ahead check
            if other_train.rear_position > new_position:
                required_distance = other_train.rear_position - (new_position + train_length)
                if required_distance < self.MIN_TRAIN_DISTANCE:
                    halt_pos = max(0, other_train.rear_position - self.MIN_TRAIN_DISTANCE - train_length)
                    return False, f"Too close to train {other_train.train_id} ahead", halt_pos
            
            # Train behind check  
            if other_train.front_position < new_position:
                required_distance = (new_position - train_length) - other_train.front_position
                if required_distance < self.MIN_TRAIN_DISTANCE:
                    halt_pos = max(0, other_train.front_position + self.MIN_TRAIN_DISTANCE + train_length)
                    return False, f"Too close to train {other_train.train_id} behind", halt_pos
        
        return True, "", None
    
    def halt_train(self, train_id: str, edge: Tuple[str, str], halt_position: float, reason: str):
        """Halt a train at specific position"""
        for train in self.edge_occupancy[edge]:
            if train.train_id == train_id:
                train.position = halt_position
                train.is_halted = True
                train.halt_reason = reason
                self.halted_trains[train_id] = {
                    'edge': edge,
                    'position': halt_position,
                    'reason': reason,
                    'halt_time': 0
                }
                break

    def tick_halted_trains(self):
        """Increment halt timers for all halted trains by one simulation step"""
        for info in self.halted_trains.values():
            info['halt_time'] = info.get('halt_time', 0) + 1
    
    def get_edge_status(self, edge: Tuple[str, str]) -> Dict:
        """Get detailed status of an edge"""
        trains = self.edge_occupancy[edge]
        edge_data = self.network.get_edge_data(edge[0], edge[1])
        edge_length = edge_data.get('length', 0)
        
        return {
            'edge': f"{edge[0]} → {edge[1]}",
            'length': edge_length,
            'train_count': len(trains),
            'trains': [
                {
                    'id': train.train_id,
                    'position': train.position,
                    'front': train.front_position,
                    'rear': train.rear_position,
                    'halted': train.is_halted,
                    'halt_reason': train.halt_reason
                }
                for train in trains
            ],
            'is_congested': len(trains) > 1
        }
    
    def get_system_overview(self) -> Dict:
        """Get overview of entire railway system"""
        total_edges = len(self.edge_occupancy)
        occupied_edges = sum(1 for trains in self.edge_occupancy.values() if trains)
        total_trains = sum(len(trains) for trains in self.edge_occupancy.values())
        halted_count = len(self.halted_trains)
        
        return {
            'total_edges': total_edges,
            'occupied_edges': occupied_edges,
            'total_trains_on_tracks': total_trains,
            'halted_trains': halted_count,
            'system_utilization': f"{(occupied_edges/total_edges)*100:.1f}%" if total_edges > 0 else "0%"
        }
    
    def print_system_status(self):
        """Print detailed system status"""
        overview = self.get_system_overview()
        print(f"\n📊 RAILWAY SYSTEM STATUS:")
        print(f"   🛤️  Total edges: {overview['total_edges']}")
        print(f"   🚂 Trains on tracks: {overview['total_trains_on_tracks']}")
        print(f"   ⛔ Halted trains: {overview['halted_trains']}")
        print(f"   📈 System utilization: {overview['system_utilization']}")
        
        # Show congested edges
        congested_edges = [
            edge for edge, trains in self.edge_occupancy.items() 
            if len(trains) > 1
        ]
        
        if congested_edges:
            print(f"\n🚧 CONGESTED TRACKS:")
            for edge in congested_edges:
                status = self.get_edge_status(edge)
                print(f"   {status['edge']}: {status['train_count']} trains")
                for train in status['trains']:
                    halt_info = f" [HALTED: {train['halt_reason']}]" if train['halted'] else ""
                    print(f"     - {train['id']}: {train['position']:.1f}m{halt_info}")
        
        # Show halted trains
        if self.halted_trains:
            print(f"\n⛔ HALTED TRAINS:")
            for train_id, info in self.halted_trains.items():
                edge_str = f"{info['edge'][0]} → {info['edge'][1]}"
                print(f"   {train_id} on {edge_str} at {info['position']:.1f}m: {info['reason']}")

# Deep Q-Learning Implementation
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class DQN(nn.Module):
    """Enhanced Deep Q-Network for railway traffic control"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 1024):  # Increased from 256
        super(DQN, self).__init__()
        # Much larger network
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size * 2)
        self.fc3 = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.fc4 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc6 = nn.Linear(hidden_size // 2, action_size)
        
        # Add batch normalization and dropout for larger network
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size * 2)
        self.bn3 = nn.BatchNorm1d(hidden_size * 2)
        self.bn4 = nn.BatchNorm1d(hidden_size)
        
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        # Handle batch size for BatchNorm compatibility
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension
            single_sample = True
        else:
            single_sample = False
            
        # Deeper network with more computations
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        
        # Remove batch dimension if single sample
        if single_sample:
            x = x.squeeze(0)
            
        return x

class ReplayBuffer:
    """Experience replay buffer for DQN"""
    
    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size: int):
        """Sample a batch of transitions"""
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class RailwayDQNAgent:
    """Enhanced DQN Agent with higher GPU utilization"""
    
    def __init__(self, state_size: int, action_size: int, lr: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        
        # Prefer GPU when available, otherwise fallback to CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"🎯 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            self.device = torch.device("cpu")
            print("⚙️  Using CPU (CUDA not available)")
        
        # Enable all GPU optimizations
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.enabled = True
        
        # Much larger networks (1024 hidden units instead of 256)
        self.q_network = DQN(state_size, action_size, hidden_size=1024).to(self.device)
        self.target_network = DQN(state_size, action_size, hidden_size=1024).to(self.device)
        
        # Use mixed precision for faster training (CUDA only)
        if self.device.type == 'cuda':
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            # No-op scaler for CPU
            class _NoOpScaler:
                def scale(self, loss):
                    return loss
                def step(self, optimizer):
                    optimizer.step()
                def update(self):
                    pass
            self.scaler = _NoOpScaler()
        
        # More aggressive optimizer
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=lr, weight_decay=1e-4)
        
        # Much larger replay buffer and batch size
        self.memory = ReplayBuffer(50000)  # Increased from 10000
        self.batch_size = 512  # Increased from 64
        
        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update = 50  # More frequent updates
        self.steps = 0
        
        # Warmup parameters for higher GPU usage
        self.warmup_steps = 1000
        self.parallel_envs = 4  # Simulate multiple environments
    
    def get_state(self, global_state: GlobalStateManager, train_id: str) -> np.ndarray:
        """Convert railway state to DQN input"""
        state_vector = []
        
        # Train-specific features
        train_info = None
        for edge, trains in global_state.edge_occupancy.items():
            for train in trains:
                if train.train_id == train_id:
                    train_info = train
                    edge_data = global_state.network.get_edge_data(edge[0], edge[1])
                    edge_length = edge_data.get('length', 1000)
                    
                    # Normalized position on edge (0-1)
                    state_vector.append(train.position / edge_length)
                    # Is halted
                    state_vector.append(1.0 if train.is_halted else 0.0)
                    # Distance to end of edge
                    state_vector.append((edge_length - train.position) / edge_length)
                    break
        
        if train_info is None:
            # Train is at node - add default values
            state_vector.extend([0.0, 0.0, 1.0])
        
        # System-wide features
        overview = global_state.get_system_overview()
        state_vector.append(overview['total_trains_on_tracks'] / 10.0)  # Normalized
        state_vector.append(overview['halted_trains'] / 5.0)  # Normalized
        
        # Local traffic density (trains on same edge)
        local_density = 0
        if train_info:
            for edge, trains in global_state.edge_occupancy.items():
                if any(t.train_id == train_id for t in trains):
                    local_density = len(trains)
                    break
        state_vector.append(local_density / 3.0)  # Normalized
        
        # Safety distances to other trains
        min_distance = 1000.0  # Default large distance
        if train_info:
            for edge, trains in global_state.edge_occupancy.items():
                if any(t.train_id == train_id for t in trains):
                    current_train = next(t for t in trains if t.train_id == train_id)
                    for other_train in trains:
                        if other_train.train_id != train_id:
                            distance = abs(current_train.position - other_train.position)
                            min_distance = min(min_distance, distance)
                    break
        
        state_vector.append(min(min_distance / 500.0, 1.0))  # Normalized
        
        # Pad or truncate to fixed size
        while len(state_vector) < self.state_size:
            state_vector.append(0.0)
        
        return np.array(state_vector[:self.state_size], dtype=np.float32)
    
    def act(self, state: np.ndarray) -> int:
        """Choose action using epsilon-greedy policy"""
        if random.random() > self.epsilon:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            self.q_network.eval()  # Set to evaluation mode for BatchNorm
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            self.q_network.train()  # Set back to training mode
            return q_values.cpu().data.numpy().argmax()
        else:
            return random.choice(range(self.action_size))
    
    def remember(self, state, action, reward, next_state):
        """Store experience in replay buffer"""
        self.memory.push(state, action, next_state, reward)
    
    def replay(self):
        """Enhanced training with mixed precision and larger batches"""
        if len(self.memory) < self.batch_size * 2:  # Wait for more samples
            return
        
        # Process multiple batches per replay for higher GPU usage
        num_batches = 4  # Process 4 batches at once
        total_loss = 0
        
        for _ in range(num_batches):
            transitions = self.memory.sample(self.batch_size)
            batch = Transition(*zip(*transitions))
            
            # Convert to tensors efficiently
            state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32, device=self.device)
            action_batch = torch.tensor(batch.action, dtype=torch.long, device=self.device)
            reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=self.device)
            next_state_batch = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=self.device)
            
            # Use mixed precision on CUDA, otherwise regular precision
            from contextlib import nullcontext
            amp_ctx = torch.amp.autocast('cuda') if self.device.type == 'cuda' else nullcontext()
            with amp_ctx:
                # Compute Q(s_t, a)
                current_q_values = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1))
                
                # Compute V(s_{t+1}) for all next states
                with torch.no_grad():
                    next_q_values = self.target_network(next_state_batch).max(1)[0]
                target_q_values = (next_q_values * self.gamma) + reward_batch
                
                # Compute loss
                loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
            
            # Backward pass with gradient scaling
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target network more frequently
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return total_loss / num_batches
    
    def warmup_gpu(self):
        """Warmup GPU with dummy computations to maximize usage"""
        if self.device.type != 'cuda':
            # Skip heavy warmup on CPU
            print("🔥 Skipping GPU warmup (CUDA not available)")
            return
        print("🔥 Warming up GPU...")
        
        # Create dummy data to saturate GPU
        dummy_states = torch.randn(self.batch_size * 4, self.state_size).to(self.device)
        
        for i in range(100):  # 100 warmup iterations
            with torch.amp.autocast('cuda'):
                # Forward pass on main network
                q1 = self.q_network(dummy_states)
                # Forward pass on target network  
                q2 = self.target_network(dummy_states)
                # Dummy loss computation
                loss = F.mse_loss(q1, q2.detach())
            
            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if i % 20 == 0:
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                print(f"   Warmup {i}/100 - GPU Memory: {allocated:.2f}GB")
        
        print("🔥 GPU warmup complete!")
    
    def get_gpu_memory_usage(self) -> str:
        """Get current GPU memory usage"""
        if torch.cuda.is_available() and self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            cached = torch.cuda.memory_reserved(0) / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return f"GPU Memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached, {total:.1f}GB total"
        return "GPU not available"
    
    def calculate_reward(self, global_state: GlobalStateManager, train_id: str, 
                        action: int, prev_state: Dict) -> float:
        """Calculate reward for the action taken"""
        reward = 0.0
        
        # Base reward for movement (encourage progress)
        reward += 1.0
        
        # Penalty for being halted
        if train_id in global_state.halted_trains:
            reward -= 10.0
        
        # Penalty for causing congestion
        congested_edges = sum(1 for trains in global_state.edge_occupancy.values() if len(trains) > 1)
        reward -= congested_edges * 2.0
        
        # Bonus for maintaining safe distances
        min_distance = 1000.0
        for edge, trains in global_state.edge_occupancy.items():
            if any(t.train_id == train_id for t in trains):
                current_train = next(t for t in trains if t.train_id == train_id)
                for other_train in trains:
                    if other_train.train_id != train_id:
                        distance = abs(current_train.position - other_train.position)
                        min_distance = min(min_distance, distance)
        
        if min_distance > 150.0:  # Safe distance maintained
            reward += 2.0
        elif min_distance < 100.0:  # Too close
            reward -= 5.0
        
        return reward
    
    def get_enhanced_state(self, global_state: GlobalStateManager, train_id: str) -> np.ndarray:
        """Enhanced state representation with more features"""
        # Expanded state vector for larger network
        state_vector = []
        
        # Train-specific features (expanded)
        train_info = None
        current_edge_info = None
        
        for edge, trains in global_state.edge_occupancy.items():
            for train in trains:
                if train.train_id == train_id:
                    train_info = train
                    current_edge_info = edge
                    edge_data = global_state.network.get_edge_data(edge[0], edge[1])
                    edge_length = edge_data.get('length', 1000)
                    
                    # More detailed position features
                    state_vector.extend([
                        train.position / edge_length,  # Normalized position
                        1.0 if train.is_halted else 0.0,  # Halt status
                        (edge_length - train.position) / edge_length,  # Distance to end
                        train.position / 1000.0,  # Absolute position (normalized)
                        edge_length / 10000.0,  # Edge length (normalized)
                        train.front_position / edge_length,  # Front position
                        train.rear_position / edge_length,  # Rear position
                    ])
                    break
        
        if train_info is None:
            # Train at node - expanded default values
            state_vector.extend([0.0, 0.0, 1.0, 0.0, 0.5, 0.0, 0.0])
        
        # Enhanced system features
        overview = global_state.get_system_overview()
        state_vector.extend([
            overview['total_trains_on_tracks'] / 10.0,
            overview['halted_trains'] / 5.0,
            overview['occupied_edges'] / 25.0,
        ])
        
        # Local traffic analysis (more detailed)
        if current_edge_info:
            edge_trains = global_state.edge_occupancy[current_edge_info]
            
            # Traffic density features
            state_vector.extend([
                len(edge_trains) / 5.0,  # Train count on edge
                sum(1 for t in edge_trains if t.is_halted) / max(len(edge_trains), 1),  # Halt ratio
            ])
            
            # Distance analysis to other trains
            distances = []
            if train_info:
                for other_train in edge_trains:
                    if other_train.train_id != train_id:
                        dist = abs(train_info.position - other_train.position)
                        distances.append(dist)
            
            if distances:
                state_vector.extend([
                    min(distances) / 1000.0,  # Closest train
                    sum(distances) / (len(distances) * 1000.0),  # Average distance
                ])
            else:
                state_vector.extend([1.0, 1.0])  # No other trains
        else:
            # Default traffic values
            state_vector.extend([0.0, 0.0, 1.0, 1.0])
        
        # Network congestion features
        congested_edges = sum(1 for trains in global_state.edge_occupancy.values() if len(trains) > 1)
        total_edges = len(global_state.edge_occupancy)
        
        state_vector.extend([
            congested_edges / max(total_edges, 1),  # Congestion ratio
            len(global_state.halted_trains) / 10.0,  # System halt ratio
        ])
        
        # Pad to fixed size (expanded from 10 to 20 features)
        target_size = 20
        while len(state_vector) < target_size:
            state_vector.append(0.0)
        
        return np.array(state_vector[:target_size], dtype=np.float32)

class RouteManager:
    """Manages predefined routes and dynamic path calculation"""
    
    def __init__(self, network: nx.DiGraph):
        self.network = network
        self.depot = "carshed"
        
        # Predefined route segments (list of lists)
        self.predefined_routes = [
            ['A1', 'B1', 'C1'],
            ['A2', 'B2', 'C2', 'D1'],
            ['A1', 'C1', 'B2'],
            ['B1', 'A2', 'D1'],
            ['A1', 'B1', 'C1'],
            ['C2', 'B1', 'A1']
        ]
    
    def find_path(self, start: str, end: str) -> List[str]:
        """Find shortest path between two nodes using NetworkX"""
        try:
            path = nx.shortest_path(self.network, start, end)
            return path
        except nx.NetworkXNoPath:
            print(f"❌ No path found from {start} to {end}")
            return []
        except nx.NodeNotFound as e:
            print(f"❌ Node not found: {e}")
            return []
    
    def build_complete_route(self, predefined_route_index: int) -> Tuple[List[str], str]:
        """
        Build complete route: carshed → route_start → predefined_route → carshed
        Returns: (complete_route, route_description)
        """
        if predefined_route_index < 0 or predefined_route_index >= len(self.predefined_routes):
            print(f"❌ Invalid route index. Available routes: 0-{len(self.predefined_routes)-1}")
            return [], ""
        
        predefined_route = self.predefined_routes[predefined_route_index]
        route_start = predefined_route[0]
        route_end = predefined_route[-1]
        
        # Build complete route
        complete_route = []
        
        # 1. Path from carshed to route start
        if route_start != self.depot:
            path_to_start = self.find_path(self.depot, route_start)
            if not path_to_start:
                return [], ""
            complete_route.extend(path_to_start[:-1])  # Exclude last node to avoid duplication
        
        # 2. Build the predefined route with path finding between each stop
        expanded_route = []
        for i in range(len(predefined_route)):
            expanded_route.append(predefined_route[i])
            
            # Find path to next stop (if not the last stop)
            if i < len(predefined_route) - 1:
                current_stop = predefined_route[i]
                next_stop = predefined_route[i + 1]
                
                # Try direct connection first
                if self.network.has_edge(current_stop, next_stop):
                    continue  # Direct connection exists, no need for intermediate path
                else:
                    # Find path between stops
                    path_between = self.find_path(current_stop, next_stop)
                    if not path_between:
                        print(f"❌ No path found between {current_stop} and {next_stop}")
                        return [], ""
                    # Add intermediate nodes (excluding first and last to avoid duplication)
                    expanded_route.extend(path_between[1:-1])
        
        # Add the expanded predefined route
        complete_route.extend(expanded_route)
        
        # 3. Path from route end back to carshed
        if route_end != self.depot:
            path_to_depot = self.find_path(route_end, self.depot)
            if not path_to_depot:
                return [], ""
            complete_route.extend(path_to_depot[1:])  # Exclude first node to avoid duplication
        
        # Create description
        description = f"Route {predefined_route_index + 1}: {' → '.join(predefined_route)}"
        
        return complete_route, description
    
    def get_available_routes(self) -> Dict[int, str]:
        """Get all available predefined routes"""
        routes = {}
        for i, route in enumerate(self.predefined_routes):
            routes[i] = f"Route {i + 1}: {' → '.join(route)}"
        return routes
    
    def add_custom_route(self, route: List[str]) -> int:
        """Add a custom route to the predefined routes"""
        # Validate route
        for node in route:
            if node not in self.network.nodes:
                print(f"❌ Invalid node in route: {node}")
                return -1
        
        self.predefined_routes.append(route)
        new_index = len(self.predefined_routes) - 1
        print(f"✅ Added custom route {new_index + 1}: {' → '.join(route)}")
        return new_index

class RailwayDQNEnvironment:
    """Enhanced environment with higher GPU utilization"""
    
    def __init__(self, network: nx.DiGraph):
        self.network = network
        self.controller = SimpleTrainController(network)
        # Enhanced agent with larger state space
        self.agent = RailwayDQNAgent(state_size=20, action_size=5)  # Expanded state
        self.route_manager = RouteManager(network)
        
        # Enhanced action space with more nuanced control
        self.action_meanings = {
            0: "maintain_speed",
            1: "accelerate", 
            2: "decelerate",
            3: "emergency_stop",
            4: "priority_override"
        }
    
    def step(self, train_id: str, action: int) -> Tuple[np.ndarray, float, bool]:
        """Execute one step in the environment"""
        # Get current state
        prev_state = self.agent.get_enhanced_state(self.controller.global_state, train_id)

        # Apply action (modify train behavior based on action)
        self._apply_action(train_id, action)

        # Step the simulation
        self.controller.step_simulation()

        # Get new state
        new_state = self.agent.get_enhanced_state(self.controller.global_state, train_id)

        # Calculate reward
        reward = self.agent.calculate_reward(self.controller.global_state, train_id, action, {})

        # Check if episode is done (train reached destination or major collision)
        done = self._is_episode_done(train_id)

        return new_state, reward, done

    def step_multi(self, actions: Dict[str, int]) -> Dict[str, Tuple[np.ndarray, float, bool]]:
        """Execute one concurrent step for multiple trains.
        actions: mapping train_id -> action index
        Returns per-train (next_state, reward, done)
        """
        results: Dict[str, Tuple[np.ndarray, float, bool]] = {}
        # Pre-capture states for all trains
        prev_states: Dict[str, np.ndarray] = {}
        for train_id in list(self.controller.trains.keys()):
            prev_states[train_id] = self.agent.get_enhanced_state(self.controller.global_state, train_id)

        # Apply all actions before stepping
        for train_id, action in actions.items():
            self._apply_action(train_id, action)

        # Single global simulation step
        self.controller.step_simulation()

        # Collect per-train outcomes
        for train_id in prev_states.keys():
            next_state = self.agent.get_enhanced_state(self.controller.global_state, train_id)
            # Use last applied action for reward; default to maintain_speed if none
            action = actions.get(train_id, 0)
            reward = self.agent.calculate_reward(self.controller.global_state, train_id, action, {})
            done = self._is_episode_done(train_id)
            results[train_id] = (next_state, reward, done)

        return results
    
    def _apply_action(self, train_id: str, action: int):
        """Apply DQN action to train"""
        if train_id not in self.controller.trains:
            return
        
        train = self.controller.trains[train_id]
        
        if action == 0:  # maintain_speed
            pass  # No change
        elif action == 1:  # accelerate
            train.speed_kmh = min(train.speed_kmh * 1.1, 80.0)
        elif action == 2:  # decelerate
            train.speed_kmh = max(train.speed_kmh * 0.9, 20.0)
        elif action == 3:  # emergency_stop
            train.speed_kmh = 0.1
        elif action == 4:  # priority_override
            # Could implement priority signaling here
            pass
    
    def _is_episode_done(self, train_id: str) -> bool:
        """Check if episode should end"""
        if train_id not in self.controller.trains:
            return True
        
        train = self.controller.trains[train_id]
        
        # Episode ends if train reaches destination
        if train.current_node == train.destination:
            return True
        
        # Episode ends if train is stuck for too long
        if train_id in self.controller.global_state.halted_trains:
            halt_info = self.controller.global_state.halted_trains[train_id]
            if halt_info.get('halt_time', 0) > 50:  # Stuck for 50 steps
                return True
        
        return False
    
    def train_agent_enhanced(self, episodes: int = 2000, route_index: int = None, num_trains: int = 3, steps_per_episode: int = 1000):
        """Enhanced training with higher GPU utilization and concurrent multi-train operation"""
        # GPU warmup to maximize usage
        self.agent.warmup_gpu()
        
        scores = deque(maxlen=100)
        losses = deque(maxlen=100)
        
        # Pre-fill memory buffer for consistent GPU usage
        print("🎯 Pre-filling memory buffer...")
        for _ in range(1000):
            # Generate random experience
            state = np.random.rand(20).astype(np.float32)
            action = random.randint(0, 4)
            reward = random.uniform(-10, 10)
            next_state = np.random.rand(20).astype(np.float32)
            self.agent.remember(state, action, reward, next_state)
        
        print(f"🏋️ Starting enhanced training for {episodes} episodes...")
        print(f"🚀 Batch size: {self.agent.batch_size}, Network size: 1024+ hidden units")
        
        for episode in range(episodes):
            # Reset environment
            self.controller = SimpleTrainController(self.network)
            train_ids: List[str] = []
            # Add multiple trains
            for i in range(num_trains):
                tid = f"TRAIN_{i+1:03d}"
                if self.controller.add_train(tid, "carshed"):
                    train_ids.append(tid)
            
            # Assign routes to all trains
            for tid in train_ids:
                if route_index is None:
                    sel_idx = random.randint(0, len(self.route_manager.predefined_routes) - 1)
                else:
                    sel_idx = route_index
                complete_route, _ = self.route_manager.build_complete_route(sel_idx)
                if complete_route:
                    self.controller.set_custom_route(tid, complete_route)
            
            total_reward = 0.0
            episode_loss = 0.0
            
            # Concurrent steps for fixed horizon
            for step in range(steps_per_episode):
                # Build actions for all active trains
                actions: Dict[str, int] = {}
                states: Dict[str, np.ndarray] = {}
                for tid in list(self.controller.trains.keys()):
                    states[tid] = self.agent.get_enhanced_state(self.controller.global_state, tid)
                    actions[tid] = self.agent.act(states[tid])
                
                # Step once with all actions
                results = self.step_multi(actions)
                
                # Store transitions and handle per-train completion
                for tid, (next_state, reward, done) in results.items():
                    self.agent.remember(states[tid], actions[tid], reward, next_state)
                    total_reward += reward
                    
                    if tid in self.controller.trains:
                        train = self.controller.trains[tid]
                        # If train reached destination, immediately assign a new route
                        if train.current_node == train.destination and train.destination is not None:
                            # Small completion bonus, more if returned to carshed
                            if train.current_node == "carshed":
                                total_reward += 100.0
                            # Assign a new random route
                            sel_idx = random.randint(0, len(self.route_manager.predefined_routes) - 1) if route_index is None else route_index
                            complete_route, _ = self.route_manager.build_complete_route(sel_idx)
                            if complete_route:
                                self.controller.set_custom_route(tid, complete_route)
                
                # Train more frequently for higher GPU usage
                if step % 4 == 0:
                    loss = self.agent.replay()
                    if loss is not None:
                        episode_loss += loss
            
            scores.append(total_reward)
            if episode_loss > 0:
                losses.append(episode_loss)
            
            # Enhanced progress reporting
            if episode % 50 == 0:  # More frequent reporting
                avg_score = np.mean(scores)
                avg_loss = np.mean(losses) if losses else 0
                gpu_info = self.agent.get_gpu_memory_usage()
                
                print(f"Episode {episode:4d} | Score: {avg_score:7.2f} | Loss: {avg_loss:6.3f} | ε: {self.agent.epsilon:.3f}")
                print(f"🚀 {gpu_info}")
                
                # GPU utilization check
                allocated_gb = torch.cuda.memory_allocated(0) / 1024**3
                max_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                utilization = (allocated_gb / max_memory_gb) * 100
                print(f"🎯 GPU Utilization: {utilization:.1f}% ({allocated_gb:.2f}GB / {max_memory_gb:.1f}GB)")
                
        print("🎉 Enhanced training completed!")
        final_gpu_info = self.agent.get_gpu_memory_usage()
        print(f"🚀 Final {final_gpu_info}")

class SimpleTrainController:
    def __init__(self, network: nx.DiGraph):
        self.network = network
        self.trains: Dict[str, TrainState] = {}
        self.time_step = 0
        self.global_state = GlobalStateManager(network)
    
    def add_train(self, train_id: str, initial_node: str = "carshed") -> bool:
        """Add a train at a specific node"""
        if initial_node not in self.network.nodes:
            print(f"Error: Node {initial_node} not found")
            return False
        
        if train_id in self.trains:
            print(f"Error: Train {train_id} already exists")
            return False
        
        train_state = TrainState(
            train_id=train_id,
            position_type=TrainPosition.AT_NODE,
            current_node=initial_node
        )
        
        self.trains[train_id] = train_state
        self.global_state.add_train_to_node(train_id, initial_node)
        print(f"🚂 Train {train_id} added at {initial_node}")
        return True
    
    def set_custom_route(self, train_id: str, route: List[str]) -> bool:
        """Set a custom route for the train"""
        if train_id not in self.trains:
            print(f"Error: Train {train_id} not found")
            return False
        
        train = self.trains[train_id]
        
        # Validate all nodes exist and edges are connected
        for i in range(len(route) - 1):
            if route[i] not in self.network.nodes:
                print(f"Error: Node {route[i]} not found")
                return False
            if not self.network.has_edge(route[i], route[i+1]):
                print(f"Error: No edge from {route[i]} to {route[i+1]}")
                return False
        
        train.destination = route[-1]
        train.planned_route = route
        train.route_index = 0
        print(f"📍 Route set for {train_id}: {' → '.join(route)}")
        return True
    
    def move_train(self, train_id: str) -> bool:
        """Move train one step forward"""
        if train_id not in self.trains:
            return False
        
        train = self.trains[train_id]
        
        if train.position_type == TrainPosition.AT_NODE:
            return self._start_edge_movement(train)
        else:
            return self._continue_edge_movement(train)
    
    def _start_edge_movement(self, train: TrainState) -> bool:
        """Start moving train from node onto edge"""
        if not train.planned_route or train.route_index >= len(train.planned_route) - 1:
            return False
        
        current_node = train.planned_route[train.route_index]
        next_node = train.planned_route[train.route_index + 1]
        
        # Check if edge exists
        if not self.network.has_edge(current_node, next_node):
            print(f"Error: No edge from {current_node} to {next_node}")
            return False
        
        # Check if train can move onto edge
        edge_key = (current_node, next_node)
        can_move, reason, halt_pos = self.global_state.can_train_move(
            train.train_id, edge_key, train.distance_per_step, train.length
        )
        
        # Remove train from current node
        self.global_state.remove_train_from_node(train.train_id, current_node)
        
        # Move train onto edge
        train.position_type = TrainPosition.ON_EDGE
        train.current_edge = edge_key
        train.current_node = None
        
        if can_move:
            train.position_on_edge = train.distance_per_step
            self.global_state.add_train_to_edge(train.train_id, edge_key, train.position_on_edge, train.length)
        else:
            train.position_on_edge = halt_pos if halt_pos is not None else 0
            self.global_state.add_train_to_edge(train.train_id, edge_key, train.position_on_edge, train.length)
            self.global_state.halt_train(train.train_id, edge_key, train.position_on_edge, reason)
            print(f"⛔ Train {train.train_id} HALTED: {reason} at {train.position_on_edge:.1f}m")
        
        edge_data = self.network.get_edge_data(current_node, next_node)
        edge_length = edge_data.get('length', 0)
        
        print(f"🚄 Train {train.train_id} started moving on edge {current_node} → {next_node} (length: {edge_length}m)")
        print(f"   🏃 Speed: {train.speed_kmh} km/h ({train.speed_ms:.1f} m/s) - Distance per step: {train.distance_per_step:.1f}m")
        return True
    
    def _continue_edge_movement(self, train: TrainState) -> bool:
        """Continue moving train along current edge"""
        if not train.current_edge:
            return False
        
        from_node, to_node = train.current_edge
        edge_data = self.network.get_edge_data(from_node, to_node)
        edge_length = edge_data.get('length', 0)
        
        # Calculate new position
        new_position = train.position_on_edge + train.distance_per_step
        
        # Check if train can move to new position
        can_move, reason, halt_pos = self.global_state.can_train_move(
            train.train_id, train.current_edge, new_position, train.length
        )
        
        if new_position >= edge_length:
            # Train will reach the end of this edge
            print(f"🎯 Train {train.train_id} completing edge {from_node} → {to_node}")
            return self._complete_edge_movement(train, to_node)
        elif can_move:
            # Train can continue moving
            train.position_on_edge = new_position
            self.global_state.update_train_position_on_edge(train.train_id, train.current_edge, new_position)
            remaining = edge_length - new_position
            print(f"🚄 Train {train.train_id}: {new_position:.0f}m from {from_node}, {remaining:.0f}m to {to_node}")
            print(f"   📏 Moved {train.distance_per_step:.1f}m in {train.time_step_seconds}s at {train.speed_kmh} km/h")
            return True
        else:
            # Train must halt
            halt_position = halt_pos if halt_pos is not None else train.position_on_edge
            train.position_on_edge = halt_position
            self.global_state.halt_train(train.train_id, train.current_edge, halt_position, reason)
            remaining = edge_length - halt_position
            print(f"⛔ Train {train.train_id} HALTED at {halt_position:.0f}m from {from_node}, {remaining:.0f}m to {to_node}")
            print(f"   🚫 Reason: {reason}")
            return True
    
    def _complete_edge_movement(self, train: TrainState, arrival_node: str) -> bool:
        """Complete movement to arrival node"""
        # Remove train from edge
        if train.current_edge:
            self.global_state.remove_train_from_edge(train.train_id, train.current_edge)
        
        # Update train position
        train.position_type = TrainPosition.AT_NODE
        train.current_node = arrival_node
        train.current_edge = None
        train.position_on_edge = 0.0
        train.route_index += 1
        
        # Add train to node
        self.global_state.add_train_to_node(train.train_id, arrival_node)
        
        print(f"🏁 Train {train.train_id} arrived at {arrival_node}")
        
        # Check if reached destination
        if arrival_node == train.destination:
            print(f"🎉 Train {train.train_id} reached FINAL DESTINATION: {arrival_node}")
            return False  # Stop movement
        
        return True
    
    def get_train_status(self, train_id: str) -> Dict:
        """Get detailed train status"""
        if train_id not in self.trains:
            return {}
        
        train = self.trains[train_id]
        
        status = {
            'train_id': train.train_id,
            'position': train.get_position_info(),
            'destination': train.destination,
            'route_progress': f"{train.route_index}/{len(train.planned_route) - 1}" if train.planned_route else "0/0"
        }
        
        return status
    
    def print_train_status(self, train_id: str):
        """Print formatted train status"""
        if train_id not in self.trains:
            print(f"Train {train_id} not found")
            return
        
        train = self.trains[train_id]
        print(f"🚂 {train.train_id}: {train.get_position_info()}")
        if train.destination:
            progress = f"{train.route_index}/{len(train.planned_route) - 1}"
            remaining_route = train.planned_route[train.route_index:]
            print(f"   🎯 Destination: {train.destination} (Progress: {progress})")
            if len(remaining_route) > 1:
                print(f"   📍 Next stops: {' → '.join(remaining_route[1:])}")
    
    def step_simulation(self):
        """Advance simulation by one time step (5 seconds of simulated time)"""
        self.time_step += 1
        simulated_time = self.time_step * 5  # Each step = 5 seconds
        print(f"\n{'='*60}")
        print(f"⏰ TIME STEP {self.time_step} (Simulated time: {simulated_time}s)")
        print(f"{'='*60}")
        
        for train_id in list(self.trains.keys()):
            self.move_train(train_id)
            self.print_train_status(train_id)
        
        # Increment halt timers each simulation tick
        self.global_state.tick_halted_trains()
        
        # Print global system status every 10 steps or when there are halted trains
        if self.time_step % 10 == 0 or self.global_state.halted_trains:
            self.global_state.print_system_status()
    
# Example usage with DQN integration
if __name__ == "__main__":
    from env import railway
    
    print("🚂 Railway Deep Q-Learning System")
    print("=" * 50)
    print("Choose simulation mode:")
    print("1. Classic Simulation (Original)")
    print("2. DQN Training Mode")
    print("3. Test Trained Agent")
    print("4. Route Demonstration (Show Dynamic Routing)")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        # Original simulation
        print("\n🚂 Starting classic railway simulation...")
        controller = SimpleTrainController(railway)
        try:
            num_trains = int(input("Number of trains (default 1): ") or "1")
        except Exception:
            num_trains = 1
        
        # Add trains and assign different routes to each using RouteManager
        route_manager = RouteManager(railway)
        available = route_manager.get_available_routes()
        route_indices = list(available.keys())
        if not route_indices:
            print("❌ No predefined routes available.")
        for i in range(num_trains):
            tid = f"TRAIN_{i+1:03d}"
            if controller.add_train(tid, "carshed"):
                # Pick a route index in round-robin fashion
                sel_idx = route_indices[i % len(route_indices)]
                complete_route, desc = route_manager.build_complete_route(sel_idx)
                if complete_route:
                    print(f"Assigning {tid} -> {desc}")
                    controller.set_custom_route(tid, complete_route)
        
        print("Starting railway simulation with global state management...")
        print("=" * 80)
        
        for step in range(50):
            print(f"\n⏰ STEP {step + 1} (Time: {(step + 1) * 5} seconds)")
            print("-" * 60)
            
            controller.step_simulation()
            controller.global_state.print_system_status()
            
            # Stop early if all trains reached destination
            all_done = True
            for tid in list(controller.trains.keys()):
                train = controller.trains[tid]
                if train.current_node != "C1":
                    all_done = False
            if all_done:
                print(f"\n🎉 All trains have reached destination C1!")
                break
            
            time.sleep(0.5)
        
        print("\n" + "=" * 80)
        print("Simulation completed!")
    
    elif choice == "2":
        # DQN Training Mode
        print("\n🤖 Starting DQN Training Mode...")
        print("This will train a Deep Q-Learning agent to control trains intelligently")
        dqn_env = RailwayDQNEnvironment(railway)
        
        # Show available routes
        print("\n🛤️  Available Predefined Routes:")
        available_routes = dqn_env.route_manager.get_available_routes()
        for index, description in available_routes.items():
            print(f"   {index + 1}. {description}")
        
        print(f"   {len(available_routes) + 1}. Random routes (AI will train on all routes)")
        
        # Route selection
        route_choice = input(f"\nSelect training route (1-{len(available_routes) + 1}): ").strip()
        
        try:
            route_num = int(route_choice)
            if route_num == len(available_routes) + 1:
                selected_route_index = None  # Random routes
                route_description = "All routes (random selection)"
            elif 1 <= route_num <= len(available_routes):
                selected_route_index = route_num - 1
                route_description = available_routes[selected_route_index]
            else:
                print("❌ Invalid route selection. Using random routes.")
                selected_route_index = None
                route_description = "All routes (random selection)"
        except ValueError:
            print("❌ Invalid input. Using random routes.")
            selected_route_index = None
            route_description = "All routes (random selection)"
        
        # Set up training parameters
        episodes = int(input("Enter number of training episodes (default 500): ") or "500")
        try:
            num_trains = int(input("Concurrent trains during training (default 3): ") or "3")
        except Exception:
            num_trains = 3
        try:
            steps_per_episode = int(input("Steps per episode (default 1000): ") or "1000")
        except Exception:
            steps_per_episode = 1000
        
        print(f"\n🏋️ Training DQN agent for {episodes} episodes...")
        print(f"🛤️  Training route: {route_description}")
        print("State space: Train position, speed, local traffic, safety distances")
        print("Action space: maintain_speed, accelerate, decelerate, emergency_stop, priority_override")
        print("-" * 60)
        
        # Start enhanced training
        dqn_env.train_agent_enhanced(episodes, selected_route_index, num_trains=num_trains, steps_per_episode=steps_per_episode)
        
        # Save the trained model
        torch.save(dqn_env.agent.q_network.state_dict(), 'railway_dqn_model.pth')
        print("\n💾 Model saved as 'railway_dqn_model.pth'")
    
    elif choice == "3":
        # Test trained agent
        print("\n🧪 Testing Trained DQN Agent...")
        
        try:
            dqn_env = RailwayDQNEnvironment(railway)
            dqn_env.agent.q_network.load_state_dict(torch.load('railway_dqn_model.pth'))
            dqn_env.agent.epsilon = 0.0  # No exploration, pure exploitation
            
            print("Loaded trained model. Running intelligent train simulation...")
            
            # Show available routes for testing
            print("\n🛤️  Available Test Routes:")
            available_routes = dqn_env.route_manager.get_available_routes()
            for index, description in available_routes.items():
                print(f"   {index + 1}. {description}")
            
            # Route selection for testing
            route_choice = input(f"\nSelect test route (1-{len(available_routes)}): ").strip()
            
            try:
                route_num = int(route_choice)
                if 1 <= route_num <= len(available_routes):
                    selected_route_index = route_num - 1
                else:
                    print("❌ Invalid selection. Using route 1.")
                    selected_route_index = 0
            except ValueError:
                print("❌ Invalid input. Using route 1.")
                selected_route_index = 0
            
            # Ask for number of concurrent test trains
            try:
                num_trains = int(input("Number of concurrent test trains (default 2): ") or "2")
            except Exception:
                num_trains = 2
            
            # Build routes and assign different ones to trains
            train_ids: List[str] = []
            available = dqn_env.route_manager.get_available_routes()
            route_indices = list(available.keys())
            if selected_route_index is not None:
                # Use chosen route as base but still vary by picking neighbors
                base_idx = selected_route_index
            for i in range(num_trains):
                tid = f"AI_TRAIN_{i+1:03d}"
                if dqn_env.controller.add_train(tid, "carshed"):
                    train_ids.append(tid)
                    # Choose route index round-robin; if a specific route chosen, offset around list
                    if selected_route_index is None:
                        sel_idx = route_indices[i % len(route_indices)]
                    else:
                        sel_idx = route_indices[(route_indices.index(base_idx) + i) % len(route_indices)] if base_idx in route_indices else base_idx
                    cr, _ = dqn_env.route_manager.build_complete_route(sel_idx)
                    if cr:
                        dqn_env.controller.set_custom_route(tid, cr)
            
            # Run concurrent simulation with AI control
            for step in range(300):
                print(f"\n🤖 AI STEP {step + 1}")
                print("-" * 40)
                
                actions: Dict[str, int] = {}
                for tid in list(dqn_env.controller.trains.keys()):
                    state = dqn_env.agent.get_state(dqn_env.controller.global_state, tid)
                    actions[tid] = dqn_env.agent.act(state)
                
                results = dqn_env.step_multi(actions)
                
                # Show a brief summary
                for tid, (_, reward, done) in results.items():
                    print(f"{tid}: reward={reward:.2f}, done={done}")
                    if tid in dqn_env.controller.trains:
                        tr = dqn_env.controller.trains[tid]
                        if tr.current_node == tr.destination and tr.destination is not None:
                            # Immediately assign a new route to keep concurrency
                            cr, _ = dqn_env.route_manager.build_complete_route(selected_route_index)
                            if cr:
                                dqn_env.controller.set_custom_route(tid, cr)
                
                dqn_env.controller.global_state.print_system_status()
                time.sleep(0.2)
        
        except FileNotFoundError:
            print("❌ No trained model found. Please run training mode first (option 2)")
        except Exception as e:
            print(f"❌ Error testing model: {e}")
    
    elif choice == "4":
        # Route Demonstration Mode
        print("\n🛤️  Route Demonstration Mode")
        print("This shows how dynamic routing works: carshed → route → carshed")
        
        route_manager = RouteManager(railway)
        
        # Show all available routes
        print("\n📋 Available Predefined Routes:")
        available_routes = route_manager.get_available_routes()
        for index, description in available_routes.items():
            print(f"   {index + 1}. {description}")
        
        # Let user select or add custom route
        print(f"\n   {len(available_routes) + 1}. Add Custom Route")
        
        route_choice = input(f"\nSelect route to demonstrate (1-{len(available_routes) + 1}): ").strip()
        
        try:
            route_num = int(route_choice)
            if route_num == len(available_routes) + 1:
                # Add custom route
                print("\nEnter custom route (space-separated station names):")
                print("Available stations: A1, A2, B1, B2, C1, C2, D1")
                custom_input = input("Custom route: ").strip()
                custom_route = custom_input.split()
                
                if custom_route:
                    route_index = route_manager.add_custom_route(custom_route)
                    if route_index >= 0:
                        selected_route_index = route_index
                    else:
                        print("❌ Using default route")
                        selected_route_index = 0
                else:
                    print("❌ Empty route. Using default route")
                    selected_route_index = 0
            elif 1 <= route_num <= len(available_routes):
                selected_route_index = route_num - 1
            else:
                print("❌ Invalid selection. Using route 1.")
                selected_route_index = 0
        except ValueError:
            print("❌ Invalid input. Using route 1.")
            selected_route_index = 0
        
        # Build and demonstrate the complete route
        complete_route, route_description = route_manager.build_complete_route(selected_route_index)
        
        if complete_route:
            print(f"\n🎯 Demonstrating: {route_description}")
            print(f"🔍 Complete calculated path:")
            print(f"   📍 Total stops: {len(complete_route)}")
            print(f"   🛤️  Full route: {' → '.join(complete_route)}")
            
            # Break down the route segments
            predefined = route_manager.predefined_routes[selected_route_index]
            print(f"\n📋 Route Breakdown:")
            
            # Path to route start
            if predefined[0] != "carshed":
                to_start = route_manager.find_path("carshed", predefined[0])
                if len(to_start) > 1:
                    print(f"   🚀 To route start: {' → '.join(to_start)}")
            
            # Predefined route
            print(f"   🎯 Main route: {' → '.join(predefined)}")
            
            # Path back to depot
            if predefined[-1] != "carshed":
                to_depot = route_manager.find_path(predefined[-1], "carshed")
                if len(to_depot) > 1:
                    print(f"   🏠 Return to depot: {' → '.join(to_depot)}")
            
            # Run actual simulation
            run_demo = input("\nRun live simulation? (y/n): ").strip().lower()
            
            if run_demo == 'y':
                controller = SimpleTrainController(railway)
                controller.add_train("DEMO_TRAIN", "carshed")
                controller.set_custom_route("DEMO_TRAIN", complete_route)
                
                print(f"\n🚂 Starting live demonstration...")
                print("=" * 60)
                
                for step in range(100):
                    if "DEMO_TRAIN" not in controller.trains:
                        break
                        
                    print(f"\n⏰ DEMO STEP {step + 1}")
                    print("-" * 40)
                    
                    controller.step_simulation()
                    
                    # Check if train returned to depot
                    train = controller.trains.get("DEMO_TRAIN")
                    if train and train.current_node == "carshed" and step > 5:
                        print("🎉 DEMO COMPLETE: Train returned to depot!")
                        break
                    
                    if step % 5 == 0:  # Print status every 5 steps
                        controller.global_state.print_system_status()
                    
                    time.sleep(0.5)
        else:
            print("❌ Failed to build route")