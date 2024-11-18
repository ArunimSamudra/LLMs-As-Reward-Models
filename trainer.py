import os
import re
import time

import numpy as np
from openai import OpenAI

from prompts import PromptGenerator


class Trainer:

    def __init__(self):
        self.prompt_generator = PromptGenerator()
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    def get_llm_reward(self, description):
        prompt = self.prompt_generator.get_prompt(description)
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": description
                }
            ]
        )
        response = response.choices[0].message.content

        # Extract the reason using regex
        # reason_match = re.search(r"Reason:\s*(.+?)(?:\n|$)", response, re.DOTALL)
        # reason = reason_match.group(1).strip() if reason_match else None
        #
        # # Extract the output using regex
        # output_match = re.search(r"Output:\s*(-?\d+(\.\d+)?)", response)
        # output = float(output_match.group(1)) if output_match else None
        #
        # print(description)
        # print(reason)
        # print(output)

        return int(response)

    def train_agent_with_llm(self, env, agent, episodes=500, batch_size=64, target_update_freq=10):
        rewards = []
        start_time = time.time()
        for episode in range(episodes):
            state, _ = env.reset()
            state = np.array(state, dtype=np.float32)
            episode_reward = 0
            done = False

            while not done:
                action = agent.select_action(state)
                next_state, algo_reward, done, _, _ = env.step(action)

                # Describe the state transition in natural language
                description = f"The agent chose action {action} and transitioned to state {next_state}. The dealer's cards are {env.env.env.dealer}"

                # Get reward from LLM
                reward = self.get_llm_reward(description)

                # Add experience to replay buffer
                next_state = np.array(next_state, dtype=np.float32)
                agent.replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward

                # Train the agent
                agent.train(batch_size)

            rewards.append(episode_reward)
            agent.epsilon = max(agent.min_epsilon, agent.epsilon * agent.epsilon_decay)

            # Update target network periodically
            if episode % target_update_freq == 0:
                agent.update_target_network()

            print(f"Episode {episode + 1}, Reward: {episode_reward}, Epsilon: {agent.epsilon:.3f}")
            print(f"----------------------------------------------------------------------------")
        end_time = time.time()

        # Calculate the time taken
        execution_time = end_time - start_time
        print("Time taken: {:.4f} seconds".format(execution_time))
        return rewards

    def train_agent(self, env, agent, episodes=3000, batch_size=64, target_update_freq=10):
        rewards = []
        start_time = time.time()
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
        end_time = time.time()

        # Calculate the time taken
        execution_time = end_time - start_time
        print("Time taken: {:.4f} seconds".format(execution_time))
        return rewards
