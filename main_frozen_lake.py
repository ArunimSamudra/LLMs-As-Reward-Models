import numpy as np
import gymnasium as gym

from algos.DQN_frozen_lake import DQNAgent
from trainer import Trainer

def evaluate_agent(env, agent, episodes=100):
    total_rewards = 0
    win_count = 0
    loss_count = 0

    for episode in range(episodes):
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32)
        done = False
        episode_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            state = np.array(next_state, dtype=np.float32)
            episode_reward += reward

        total_rewards += episode_reward

        # In FrozenLake, the agent wins if it reaches the goal (reward = 1), and loses if it falls in a hole (reward = 0).
        if episode_reward == 1:  # Win
            win_count += 1
        else:  # Loss
            loss_count += 1

    avg_reward = total_rewards / episodes
    win_ratio = win_count / episodes
    loss_ratio = loss_count / episodes

    print(f"Evaluation Results over {episodes} episodes:")
    print(f"  Average Reward: {avg_reward:.2f}")
    print(f"  Win Ratio: {win_ratio * 100:.2f}%")
    print(f"  Loss Ratio: {loss_ratio * 100:.2f}%")

    return avg_reward, win_ratio, loss_ratio


if __name__ == "__main__":
    env_name = "FrozenLake-v1"
    env = gym.make(env_name, render_mode=None, is_slippery=True)  # Add 'is_slippery' to simulate the stochastic environment

    state_dim = env.observation_space.n  # FrozenLake state space is discrete (size = number of grid positions)
    action_dim = env.action_space.n  # FrozenLake action space (4: up, down, left, right)

    agent = DQNAgent(env_name, state_dim, action_dim)
    trainer = Trainer()

    rewards = trainer.train_agent(env, agent, 1500)  # Train the agent

    print("Training complete!")

    print("Evaluating the trained agent...")
    evaluate_agent(env, agent)
