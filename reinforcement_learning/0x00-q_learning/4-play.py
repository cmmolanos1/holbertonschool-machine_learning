#!/usr/bin/env python3

import numpy as np


def play(env, Q, max_steps=100):
    """has the trained agent play an episode.

    Args:
        env: the FrozenLakeEnv instance
        Q (np.ndarray): Q-table
        max_steps (int): the maximum number of steps in the episode

    Returns:
        the total rewards for the episode.
    """
    # reset and print the state
    state = env.reset()
    env.render()

    for step in range(max_steps):

        # take the action with maximum expected future reward form the q-table
        action = np.argmax(Q[state, :])

        new_state, reward, done, info = env.step(action)
        env.render()

        if done:
            # check if the agent reached the goal(G) or fell into a hole(H)
            break

        state = new_state

    # close the connection to the environment
    env.close()

    return reward
