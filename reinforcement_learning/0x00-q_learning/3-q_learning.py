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


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """Performs Q-learning.

    Args:
        env: the FrozenLakeEnv instance.
        Q (np.ndarray): Q table.
        episodes (int): the total number of episodes to train over
        max_steps (int): the maximum number of steps per episode.
        alpha (float): the learning rate.
        gamma (float): the discount rate.
        epsilon (float): the initial threshold for epsilon greedy.
        min_epsilon (float): the minimum value that epsilon should decay to.
        epsilon_decay (float): the decay rate for updating epsilon between
                               episodes.
    Returns:
        Q, total_rewards
        - Q is the updated Q-table.
        - total_rewards is a list containing the rewards per episode.
    """

    rewards = []
    initial_epsilon = epsilon

    # let the agent play for defined number of episodes
    for episode in range(episodes):
        # reset the environment for each episode
        state = env.reset()
        # define initial parameters
        step = 0
        # to keep track whether the agent dies
        done = False
        # keep track of rewards at each episode
        total_rewards = 0

        # run for each episode
        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)

            new_state, reward, done, info = env.step(action)

            if done and reward == 0:
                reward = -1

            # update the state-action reward value in the q-table using the
            # Bellman equation
            # Q(s,a) = Q(s,a) + learning_rate*[Reward(s,a) +
            # gamma*max Q(snew,anew) - Q(s,a)]
            Q[state, action] = Q[state, action] + alpha * \
                (reward + gamma * np.max(Q[new_state, :]) - Q[state, action])

            # add to total rewards for this episode
            total_rewards += reward

            # define new state
            state = new_state

            # end the episode if agent dies
            if done is True:
                break

        # reduce the epsilon after each episode
        epsilon = min_epsilon + (initial_epsilon - min_epsilon) * \
            np.exp(-epsilon_decay * episode)

        # keep track of total rewards for each episode
        rewards.append(total_rewards)

    return Q, rewards
