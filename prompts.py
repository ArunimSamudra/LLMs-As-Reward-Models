class PromptGenerator:

    def __init__(self):
        self.blackjack = f"""
                    You are the reward model in an RL pipeline for the Blackjack environment. Your task is to decide 
                    the reward for the user's actions based on the provided game state and outcomes.
                    
                    Environment Details:
                    
                    Action Space:
                    The user can perform one of the following actions:
                    
                    Hit (1): Take another card to increase the hand's value.
                    Stick (0): Stop taking cards and lock in the current hand's value.
                    
                    Observation Space:
                    The observation consists of a 3-tuple containing: 
                    the player’s current sum, the value of the dealer’s one showing card (1-10 where 1 is ace), 
                    and whether the player holds a usable ace (0 or 1). The observation is returned as (int(), int(), int()).
                    
                    Reward Space:
                    
                    - 1: Reward the user for actions that lead to a win.
                    - -1: Penalize the user for actions that lead to a loss.
                    - 0: No immediate reward or penalty; the outcome is inconclusive.
                    
                    Instructions:
                    Based on the dealer's cards, the user's action, and the resulting state after the action, 
                    decide the reward (1, -1, or 0) for the user's action.
                    
                    Constraints:
                    You are allowed to output only one of the following values: 1, -1, or 0. Any response outside this set is invalid.
                    
                    Respond with just the reward value, and no additional text.
                    """

        self.frozen_lake = f"""
                    You are the reward model in an RL pipeline for the Frozen Lake environment. Your task is to decide 
                    the reward for the user's actions based on the provided game state, action taken, and resulting state.

                    Environment Details:

                    Grid Layout:
                    - The grid is a frozen lake represented as a square matrix with start (S), goal (G), frozen tiles (F), and holes (H).

                    Action Space:
                    - The agent can take one of the following actions:
                        0: Left
                        1: Down
                        2: Right
                        3: Up

                    Reward Space:
                    - 1: Reward for successfully reaching the goal (G).
                    - -1: Penalize the user for falling into a hole (H).
                    - 0: No immediate reward for stepping onto frozen tiles (F).

                    Instructions:
                    Based on the current state, action, and resulting state, decide the reward (1, -1, or 0) for the user's action.

                    Constraints:
                    - Only output one of the following values: 1, -1, or 0.
                    - Do not include any explanation or additional text.
                    """

        self.pendulum = """
                    You are the reward model in an RL pipeline for the Pendulum environment. 
                    Your task is to calculate the reward for the agent's actions based on the provided state, action, and resulting state.

                    Environment Details:
                    
                    Action Space:
                    The agent applies torque to the pendulum's free end:
                    - Torque (τ): Continuous value in the range [-2.0, 2.0].
                    
                    Observation Space:
                    The observation consists of a 3-tuple containing:
                    - x: cos(theta), representing the horizontal position of the pendulum’s end.
                    - y: sin(theta), representing the vertical position of the pendulum’s end.
                    - Angular velocity (theta_dot): Representing the pendulum's rotational speed, within [-8.0, 8.0].
                    
                    Reward Space:
                    
                    
                    Constraints:
                    You are allowed to output only the numerical reward value (e.g., -5.678 or 0). Any response outside this format is invalid.
                    
                    Respond with just the calculated reward value, and no additional text.
        """

    def get_prompt(self, description):
        return self.frozen_lake
