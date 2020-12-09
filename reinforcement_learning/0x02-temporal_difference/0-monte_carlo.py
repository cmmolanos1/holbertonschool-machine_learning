#!/usr/bin/env python3

def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """performs the Monte Carlo algorithm.

    Args:
        env: the openAI environment instance.
        V (np.ndarray): shape (s,) containing the value estimate.
        policy: is a function that takes in a state and returns the next
                action to take.
        episodes (int): is the total number of episodes to train over.
        max_steps (int): the maximum number of steps per episode.
        alpha (float): the learning rate.
        gamma (float): the discount rate.

    Returns:
        V, the updated value estimate.
    """
    return V