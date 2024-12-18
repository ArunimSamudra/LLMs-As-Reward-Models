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
                    - There is only one Goal and start, however there can be multiple frozen tiles and holes.
                    - Format of the grid:
                    - The grid is represented as a list of strings, where each string corresponds to a row of the grid.
                        - For example, a 4x4 grid might look like this:
                            ['SFHF', 
                             'FFFH', 
                             'FFHF', 
                             'FFFG']

                    Action Space:
                    - The agent can take one of the following actions:
                        0: Left
                        1: Down
                        2: Right
                        3: Up
                        
                    Observation Space:
                    - The observation is a value representing the player’s current position as current_row * ncols + current_col 
                    (where both the row and col start at 0).

                    - For example, the goal position in the 4x4 map can be calculated as follows: 3 * 4 + 3 = 15. 
                    The number of possible observations is dependent on the size of the map.
                    
                    State Mapping:
                    - Each state is represented as an integer corresponding to its position in the grid, with positions counted row-wise from 0.
                        For example, in a 4x4 grid:
                        - The top-left corner (row 0, column 0) is state 0.
                        - The top-right corner (row 0, column 3) is state 3.
                        - The bottom-left corner (row 3, column 0) is state 12.
                        - The bottom-right corner (row 3, column 3) is state 15.
                    - The grid values at these positions are S (start), G (goal), F (frozen tile), or H (hole).
    

                    Reward Space:
                    - **1**: Reward for successfully reaching the goal (G).  
                        - This reward of **+1 MUST ONLY be given if the `next_state` corresponds to the position of 'G' in the grid**.
                    - **0**: Reward for all other cases:
                        - Falling into a hole (H).
                        - Moving to a frozen tile (F).
                        - Staying at or returning to the start (S).
                    
                    Reward of +1 should ONLY be given if the state reaches the Goal (G).
                    
                    IMPORTANT CONSTRAINT:
                    - Reward of **1** must ONLY be given if the `next_state` corresponds to 'G'.
                    - For all other cases, including transitions to S, F, or H, the reward MUST be 0.
                    - If the `next_state` is NOT the goal (G), output **only 0**. No exceptions.

                    Instructions:
                    Based on the action which is either 0,1,2 or 3 and the resulting state which represents the grid value, 
                    decide the reward (1 or 0) for the user's action.
                    Ensure that the output adheres to the rules stated above.

                    Constraints:
                    - Only output one of the following values: 1 or 0.
                    - Do not include any explanation or additional text.
                    
                    Example of generating a reward:
                        - action is 1, next state is 7, and the grid is ['SFHF', 
                             'FFFH', 
                             'FFHF', 
                             'FFFG']
                        - Use the grid size (number of rows and columns) to compute the row and column.
                        - Row index = `next_state // number_of_columns`.
                        - Column index = `next_state % number_of_columns`.
                        - For example, in a 4x4 grid:
                            - Row index = 7 // 4 = 1
                            - Column index = 7 % 4 = 3
                            - So, `next_state` 7 corresponds to the grid position (1, 3).
                        - (1, 3) corresponds to 'H' which means reward should be 0.
                        - If `next_state` had corresponded to the goal (G), the reward would be **1**
                    """

        self.pendulum = f"""
                   
                    You are the reward model in an RL pipeline for the Pendulum environment. 
                    The system consists of a pendulum attached at one end to a fixed point, and the other end being free. The pendulum starts in a random position and the goal is to apply torque on the free end to swing it into an upright position, with its center of gravity right above the fixed point.
                    The Pendulum coordinate system is as follows - 
                        x-y: cartesian coordinates of the pendulum’s end in meters.
                        theta : angle in radians.
                        tau: torque in N m. Defined as positive counter-clockwise.

                    Each input prompt consists of the torque [-2, 2] and a 3 element tensor - x[-1, 1], y[-1, 1], angular velocity[-8, 8]
                    x = cos(0)
                    y = sin(0)

                    You need to obtain 0 [-pi, pi] from x and y
                    Then you need to use 0, torque and the angular velocity to define the reward

                    The reward varies between 0 and -10.
                    It is maximum when 0, angular velocity and torque are all 0.
                    It is minimum when 0, angular velocity and torque are at the ends of their limits.

                    Design a reward function that best models this behaviour.
                    Your response should consist **only of a floating-point number** (no additional text, explanation, or formatting) which is the output of the reward function based on the input that you'll get in the next prompt. 
                      
                    
        """

    def get_prompt(self, description):
        return self.frozen_lake
