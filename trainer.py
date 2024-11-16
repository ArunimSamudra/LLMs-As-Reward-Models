import numpy as np
import openai


class Trainer:

    def get_llm_reward(self, description, objective_prompt="Maximize winnings while minimizing risk"):
        prompt = f"""
    Objective:
    {objective_prompt}

    Description:
    {description}

    Question:
    Based on the objective, should the agent's action be rewarded? Respond with '1' for yes and '0' for no.
    """
        response = openai.Completion.create(
            model="gpt-4o-mini",
            prompt=prompt,
            max_tokens=1,
            temperature=0.0
        )
        reward = int(response.choices[0].text.strip())
        return reward

    def train_agent_with_llm(self, env, agent, episodes=500, batch_size=64, target_update_freq=10):
        for episode in range(episodes):
            state, _ = env.reset()
            state = np.array(state, dtype=np.float32)
            episode_reward = 0
            done = False

            while not done:
                action = agent.select_action(state)
                next_state, _, done, _, _ = env.step(action)

                # Describe the state transition in natural language
                description = f"The agent chose action {action} and transitioned to state {next_state}."

                # Get reward from LLM
                reward = self.get_llm_reward(description)

                # Add experience to replay buffer
                next_state = np.array(next_state, dtype=np.float32)
                agent.replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward

                # Train the agent
                agent.train(batch_size)

            # Update target network periodically
            if episode % target_update_freq == 0:
                agent.update_target_network()

            print(f"Episode {episode + 1}, Reward: {episode_reward}, Epsilon: {agent.epsilon:.3f}")

    def train_agent(self, env, agent, episodes=3000, batch_size=64, target_update_freq=10):
        rewards = []
        for episode in range(episodes):
            state, _ = env.reset()
            state = np.array(state, dtype=np.float32)
            episode_reward = 0
            done = False

            while not done:
                action = agent.select_action(state)
                next_state, reward, done, _, _ = env.step(action)
                next_state = np.array(next_state, dtype=np.float32)
                agent.replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward

                # Train the agent
                agent.train(batch_size)

            rewards.append(episode_reward)
            agent.epsilon = max(agent.min_epsilon, agent.epsilon * agent.epsilon_decay)

            # Update the target network
            if episode % target_update_freq == 0:
                agent.update_target_network()

            print(f"Episode {episode + 1}, Reward: {episode_reward}, Epsilon: {agent.epsilon:.3f}")

        return rewards
