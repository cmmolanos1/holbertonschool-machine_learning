#!/usr/bin/env python3

def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """performs SARSA(λ).

    Args:
        env: the openAI environment instance.
        Q (np.ndarray): of shape (s,a) containing the Q table.
        lambtha: the eligibility trace factor.
        episodes (int): is the total number of episodes to train over.
        max_steps (int): the maximum number of steps per episode.
        alpha (float): the learning rate.
        gamma (float): the discount rate.
        epsilon (int): the initial threshold for epsilon greedy.
        min_epsilon (float): the minimum value that epsilon should decay to.
        epsilon_decay (float): the decay rate for updating epsilon between
                               episodes.

    Returns:
        Q: the updated Q table
    """

    return Q
