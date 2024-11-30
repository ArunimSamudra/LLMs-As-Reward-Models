import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import numpy as np

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Define the Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)


# class DQNAgent:
#     def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.epsilon_decay = epsilon_decay
#         self.min_epsilon = min_epsilon
#         self.q_network = QNetwork(state_dim, action_dim)
#         self.target_network = QNetwork(state_dim, action_dim)
#         self.target_network.load_state_dict(self.q_network.state_dict())
#         self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
#         self.replay_buffer = ReplayBuffer(capacity=10000)

#     def select_action(self, state):
#         if np.random.rand() < self.epsilon:
#             return np.random.choice(self.action_dim)
#         else:
#             state = torch.FloatTensor(state).unsqueeze(0)
#             with torch.no_grad():
#                 q_values = self.q_network(state)
#             return torch.argmax(q_values).item()

#     def train(self, batch_size):
#         if len(self.replay_buffer) < batch_size:
#             return

#         states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
#         states = torch.FloatTensor(states)
#         actions = torch.LongTensor(actions)
#         rewards = torch.FloatTensor(rewards)
#         next_states = torch.FloatTensor(next_states)
#         dones = torch.FloatTensor(dones)

#         # Compute target Q-values
#         with torch.no_grad():
#             next_q_values = self.target_network(next_states)
#             max_next_q_values = next_q_values.max(dim=1)[0]
#             target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

#         # Compute current Q-values
#         q_values = self.q_network(states)
#         q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

#         # Compute loss and update network
#         loss = nn.MSELoss()(q_values, target_q_values)
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()

#     def update_target_network(self):
#         self.target_network.load_state_dict(self.q_network.state_dict())


class DQNAgent:
    def __init__(self, env_name, state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.state_dim = state_dim  # Total number of states in FrozenLake
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.env_name = env_name

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_dim)
        else:
            # Convert state to one-hot encoding before passing to network
            state_one_hot = np.zeros(self.state_dim)
            state_one_hot[int(state)] = 1.0
            state_tensor = torch.FloatTensor(state_one_hot).unsqueeze(0)  # Convert to tensor
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()

    def train(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # One-hot encode the states and next states
        states_one_hot = np.zeros((batch_size, self.state_dim))
        next_states_one_hot = np.zeros((batch_size, self.state_dim))
        
        for i in range(batch_size):
            states_one_hot[i][int(states[i])] = 1.0
            next_states_one_hot[i][int(next_states[i])] = 1.0

        states_tensor = torch.FloatTensor(states_one_hot)
        actions_tensor = torch.LongTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards)
        next_states_tensor = torch.FloatTensor(next_states_one_hot)
        dones_tensor = torch.FloatTensor(dones)

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states_tensor)
            max_next_q_values = next_q_values.max(dim=1)[0]
            target_q_values = rewards_tensor + (1 - dones_tensor) * self.gamma * max_next_q_values

        # Compute current Q-values
        q_values = self.q_network(states_tensor)
        q_values = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        # Compute loss and update network
        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
