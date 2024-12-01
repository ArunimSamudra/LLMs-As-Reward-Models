import gym
import numpy as np
from algos.DDPG import DDPG  # Import the DDPG implementation

# Define the options for DDPG
class Options:
    def __init__(self):
        self.gamma = 0.99  # Discount factor
        self.alpha = 1e-3  # Learning rate
        self.layers = [256, 256]  # Hidden layers for actor-critic networks
        self.replay_memory_size = 100000  # Replay buffer size
        self.batch_size = 64  # Batch size for training
        self.steps = 200  # Max steps per episode
        self.train_episodes = 200  # Number of training episodes
        self.test_episodes = 20

def main():
    # Create the Pendulum environment
    env = gym.make("Pendulum-v1")
    eval_env = gym.make("Pendulum-v1", render_mode='human')  # Separate evaluation environment
    
    # Define options
    options = Options()
    
    # Instantiate the DDPG solver
    agent = DDPG(env, eval_env, options)

    # Training loop
    episode_rewards = []
    for episode in range(options.train_episodes):
        # Run one episode
        agent.train_episode()
        
        # Evaluate the policy
        total_reward = 0
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            state = next_state
        
        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1}/{options.train_episodes}, Reward: {total_reward:.2f}")

    # Plot results
    agent.plot(episode_rewards, final=True)

    # Evaluating
    test_episode_rewards = []

    for episode in range(options.test_episodes):
        
        # Evaluate the policy
        total_reward_test = 0
        state, _ = eval_env.reset()
        done = False
        while not done:
            action = agent.policy(state)
            next_state, reward, terminated, truncated, _ = eval_env.step(action)
            total_reward_test += reward
            done = terminated or truncated
            state = next_state
        
        test_episode_rewards.append(total_reward_test)
        print(f"Episode {episode + 1}/{options.test_episodes}, Reward: {total_reward_test:.2f}")

if __name__ == "__main__":
    main()
