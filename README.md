# LLMs-As-Reward-Models
## Introduction
We explore the use of Large Language Models (LLMs) for designing reward functions in reinforcement learning (RL), aiming to provide a more intuitive and efficient alternative to traditional methods.
We test our method over three different environments and compare the results to the baseline rewards - 

- Blackjack: [main.py](main.py)
- FrozenLake: [main_frozen_lake.py](main_frozen_lake.py)
- Pendulum: [main_pendulum.py](main_pendulum)

## Dependancies
We use the following libraries for our project, make sure you have them installed before you run the any of the files
- numpy
- torch
- gymnasium
- openai
- pygame

Note that you will also require an OpenAI API Key in order to run these files. Follow these steps if you don't have a key setup already - 
1. Go to the following [link](https://platform.openai.com/settings/organization/api-keys) and create a key after logging in
2. Copy the generated key and create a new system variable with the name $OPENAI_API_KEY
