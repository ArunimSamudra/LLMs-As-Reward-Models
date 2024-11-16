import numpy as np
import gymnasium as gym

from algos.DQN import DQNAgent
from trainer import Trainer


def evaluate_agent(env, agent, episodes=100):
    total_rewards = 0
    win_count = 0
    loss_count = 0
    draw_count = 0

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

        # Update total rewards
        total_rewards += episode_reward

        # Determine outcome
        if episode_reward > 0:  # Win
            win_count += 1
        elif episode_reward < 0:  # Loss
            loss_count += 1
        else:  # Draw
            draw_count += 1

    avg_reward = total_rewards / episodes
    win_ratio = win_count / episodes
    loss_ratio = loss_count / episodes
    draw_ratio = draw_count / episodes

    print(f"Evaluation Results over {episodes} episodes:")
    print(f"  Average Reward: {avg_reward:.2f}")
    print(f"  Win Ratio: {win_ratio * 100:.2f}%")
    print(f"  Loss Ratio: {loss_ratio * 100:.2f}%")
    print(f"  Draw Ratio: {draw_ratio * 100:.2f}%")

    return avg_reward, win_ratio, loss_ratio, draw_ratio


if __name__ == "__main__":
    env = gym.make("Blackjack-v1", render_mode=None)
    state_dim = 3  # The state space includes the player's hand, dealer's card, and usable ace indicator
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)
    trainer = Trainer()
    rewards = trainer.train_agent(env, agent)

    print("Training complete!")

    print("Evaluating the trained agent...")
    evaluate_agent(env, agent)
