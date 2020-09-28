#!/usr/bin/env python3

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Uses epsilon-greedy to determine the next action.

    Args:
        Q (np.ndarray): The Q table.
        state: the current state.
        epsilon (float): the epsilon to use for the calculation.

    Returns:
        the next action index.
    """
    if np.random.uniform(0, 1) < epsilon:
        action = np.random.randint(Q.shape[1])
    else:
        action = np.argmax(Q[state, :])
    return action
