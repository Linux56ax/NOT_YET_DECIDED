# Railway Deep Q-Learning System

## 🚂 Overview
This project integrates **Deep Q-Learning (DQN)** with a comprehensive railway simulation system, enabling AI-powered train control with collision detection and safety constraints.

## 🎯 Features

### Core Railway Simulation
- **NetworkX-based railway network** with stations (A1, A2, B1, B2, C1, C2, D1) and intersections
- **Realistic train physics**: 60-80 km/h speeds with 5-second simulation steps
- **Global State Manager** with comprehensive collision detection
- **Safety constraints**: 100m minimum train separation, 25m node safety distance
- **Real-time monitoring** of system utilization and train positions

### Deep Q-Learning Integration
- **Neural Network Architecture**: 4-layer DNN (256 hidden units each, dropout 0.2)
- **State Space**: 10 features including:
  - Train position on edge (normalized 0-1)
  - Halt status and distance to edge end
  - System-wide traffic density
  - Local traffic density
  - Safety distances to other trains
- **Action Space**: 5 actions
  - `maintain_speed`: Keep current speed
  - `accelerate`: Increase speed by 10%
  - `decelerate`: Decrease speed by 10%
  - `emergency_stop`: Reduce speed to 0.1 km/h
  - `priority_override`: Override safety protocols (future feature)
- **Reward Function**: Balances efficiency, safety, and traffic management
  - +1.0 for movement progress
  - -10.0 for being halted
  - -2.0 per congested edge
  - +2.0 for maintaining safe distances (>150m)
  - -5.0 for dangerous proximity (<100m)

### Technical Components
- **Experience Replay Buffer**: 10,000 transitions
- **Target Network**: Updated every 100 steps
- **Epsilon-Greedy Exploration**: 1.0 → 0.01 decay
- **Adam Optimizer**: Learning rate 0.001
- **MSE Loss Function** for Q-value updates

## 🚀 Usage

### 1. Classic Simulation Mode
```bash
python ai.py
# Choose option 1
```
- Traditional rule-based train control
- Collision detection and safety systems
- Real-time visualization of train movements

### 2. DQN Training Mode
```bash
python ai.py
# Choose option 2
# Enter number of episodes (e.g., 500)
```
- Trains AI agent to control trains intelligently
- Learns to balance speed, safety, and efficiency
- Saves trained model as `railway_dqn_model.pth`

### 3. Test Trained Agent
```bash
python ai.py
# Choose option 3
```
- Loads trained DQN model
- Demonstrates AI-controlled train operation
- Pure exploitation (no exploration)

## 📊 Training Results
The DQN agent learns to:
- **Dynamically adjust speed** based on traffic conditions
- **Avoid collisions** through predictive braking
- **Optimize route efficiency** while maintaining safety
- **Handle complex railway scenarios** with multiple trains

## 🔧 Architecture

### Classes
- **`TrainState`**: Manages individual train properties and physics
- **`GlobalStateManager`**: Centralized collision detection and system monitoring
- **`DQN`**: Neural network for Q-value estimation
- **`RailwayDQNAgent`**: Complete RL agent with experience replay
- **`RailwayDQNEnvironment`**: Training environment wrapper
- **`SimpleTrainController`**: Traditional rule-based control (comparison baseline)

### Key Files
- `ai.py`: Main simulation and DQN implementation
- `env.py`: Railway network definition (NetworkX graph)
- `railway_dqn_model.pth`: Saved trained model weights

## 🎯 Example Training Output
```
Episode 0, Average Score: -127.45, Epsilon: 0.995
Episode 100, Average Score: -45.23, Epsilon: 0.605
Episode 200, Average Score: -12.78, Epsilon: 0.366
Episode 300, Average Score: 8.45, Epsilon: 0.222
Episode 400, Average Score: 23.67, Epsilon: 0.134
Training completed!
💾 Model saved as 'railway_dqn_model.pth'
```

## 🚦 Safety Features
- **100m minimum separation** between trains on same edge
- **25m node safety distance** when approaching stations/junctions
- **Emergency halt system** when safety constraints violated
- **Real-time collision detection** with reason tracking
- **System congestion monitoring** and alerts

## 🔬 Research Applications
This system enables research in:
- **Multi-agent reinforcement learning** for railway traffic optimization
- **Safety-critical AI systems** with hard constraints
- **Real-time decision making** under uncertainty
- **Traffic flow optimization** in complex networks
- **Human-AI collaboration** in transportation systems

## 🛠️ Dependencies
- Python 3.11+
- PyTorch 2.8+ (CUDA support available)
- NetworkX 3.5+
- NumPy 2.3+

## 📈 Future Enhancements
- Multi-train DQN coordination
- Passenger demand modeling
- Energy optimization objectives
- Integration with real railway timetables
- Distributed training for large networks
