import numpy as np
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

from algos.DQN_frozen_lake import DQNAgent
from trainer import Trainer

import matplotlib.pyplot as plt

def plot_losses(losses, llm_losses, title="Loss Over Time", xlabel="Episodes", ylabel="Loss"):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, marker='o', linestyle='-', label="Loss")
    plt.plot(range(1, len(losses) + 1), llm_losses, marker='o', linestyle='-', label="Loss")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

def evaluate_agent(agent, env, episodes=100, randomize_env=False):
    rewards = []
    total_rewards = 0
    win_count = 0
    loss_count = 0

    for episode in range(episodes):
        if(randomize_env):
            env = gym.make(env_name, render_mode='human', is_slippery=False, desc=generate_random_map(size=4))  # Add 'is_slippery' to simulate the stochastic environment

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
        rewards.append(episode_reward)

        # In FrozenLake, the agent wins if it reaches the goal (reward = 1), and loses if it falls in a hole (reward = 0).
        if episode_reward == 1:  # Win
            win_count += 1
        else:  # Loss
            loss_count += 1

    print("Average reward: " + str(np.mean(np.array(rewards))))
    agent.plot(rewards, final=True)
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
    map = generate_random_map(size=4)
    print(map)
    env = gym.make(env_name, render_mode=None, is_slippery=False, desc=generate_random_map(size=4))  # Add 'is_slippery' to simulate the stochastic environment

    state_dim = env.observation_space.n  # FrozenLake state space is discrete (size = number of grid positions)
    action_dim = env.action_space.n  # FrozenLake action space (4: up, down, left, right)
    trainer = Trainer()
    episodes = 500

    llm_agent = DQNAgent(env_name, state_dim, action_dim)
    llm_rewards, llm_losses = trainer.train_agent_with_llm(env, llm_agent, episodes, map=map)
    llm_agent.plot(llm_rewards, final=True)

    agent = DQNAgent(env_name, state_dim, action_dim)
    rewards, losses = trainer.train_agent(env, agent, episodes)  # Train the agent
    agent.plot(rewards, final=True)

    # Randomize Map
    # for episode in range(episodes):
    #     env = gym.make(env_name, render_mode=None, is_slippery=False, desc=generate_random_map(size=4))  # Add 'is_slippery' to simulate the stochastic environment
    #     rewards = trainer.train_agent(env, agent, episodes=1, verbose=False)  # Train the agent
    #     print(f"Episode {episode + 1}, Reward: {rewards[-1]}, Epsilon: {agent.epsilon:.3f}")

    print("Training complete!")

    print("Evaluating the trained agent...")

    #env = gym.make(env_name, render_mode='human', is_slippery=False) #, desc=generate_random_map(size=4))  # Add 'is_slippery' to simulate the stochastic environment, render_mode='human' for visualization, desc=generate_random_map(size=4) for random map
    evaluate_agent(agent, env, randomize_env=False)
    evaluate_agent(llm_agent, env, randomize_env=False)



    plot_losses(losses, llm_losses)
