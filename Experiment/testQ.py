from __future__ import print_function, division

import pickle
import matplotlib.pyplot as plt
from utils import max_dict, print_values, print_policy
import numpy as np
import gymnasium as gym
from collections import defaultdict

GAMMA = 0.9 # discounting
ALPHA = 1/5000 # step size
EPSILON = 0.1 # sprinkle in some randomness for exploration sake
N_EPISODES = 1

env = gym.make('Taxi-v3', render_mode="ansi")

# epsilon greedy action selection
def epsilon_action(a, random, eps=0.1):
  p = np.random.random()
  if p < (1 - eps):
    return a
  else:
    return random

def discountedReturn(states_actions_rewards, GAMMA):
    r = 0
    for index, val in enumerate(states_actions_rewards):
        r += ((GAMMA ** index) * val[2])
    return r

def play_game(Q, epsilon):
    states_actions_rewards = []
    episode_over = False
    observation, info = env.reset(seed=42)

    while not episode_over:
        if observation not in Q:
            Q[observation] = {}

        best_action, _ = max_dict(Q[observation])
        if best_action is None:
            best_action = env.action_space.sample()

        a = epsilon_action(best_action, env.action_space.sample(), epsilon)
        print('observation', observation, best_action, epsilon, a)

        observation, reward, terminated, truncated, info = env.step(a)
        print(env.render())

        print("1best_action: "+ str(best_action), "action: "+ str(a))
        print(
        '[\n',
            'observation: ' + str(observation), '\n',
            'info: ' + str(info), '\n',
            'action: ' + str(a), '\n',
            'reward: ' + str(reward), '\n',
            ' ', '\n',
            'terminated: ' + str(terminated), '\n',
            'truncated: ' + str(truncated), '\n',
            ' '
        '\n]')
        states_actions_rewards.append((observation, a, reward))
        episode_over = terminated or truncated
    env.close()

    states_actions_returns = []
    seen_state_action_pairs = set()

    for index, val in enumerate(states_actions_rewards):
        # first-visit Monte Carlo optimization
        s = val[0]
        a = val[1]
        sa = (s, a)

        try:
            if s not in Q:
                Q[s] = {}
            if a not in Q[s]:
                Q[s][a] = 0
        except KeyError:
            print('KeyError', Q, s, a)

        if sa not in seen_state_action_pairs:
            G = discountedReturn(states_actions_rewards[index:], GAMMA)
            states_actions_returns.append((s, a, G))
    return states_actions_returns


def monte_carlo(eps_start = 1.0, eps_decay = 0.99999, eps_min = 0.05):
    nA = env.action_space.n

    with open('Q.dat', 'rb') as loaded_file:
        Q = pickle.load(loaded_file)

    # keep track of how much our Q values change each episode so we can know when it converges
    deltas = []
    epsilon = EPSILON

    # repeat for the number of episodes specified (enough that it converges)
    for t in range(N_EPISODES):

        if t % 1000 == 0:
            print(t, epsilon)

        # generate an episode using the current policy
        biggest_change = 0
        states_actions_returns = play_game(Q, EPSILON)

        # calculate Q(s,a)
        for s, a, G in states_actions_returns:
            # check if we have already seen s
            # first-visit Monte Carlo optimization
            old_q = Q[s][a]
            Q[s][a] = Q[s][a] + (ALPHA * (G-Q[s][a]))
            biggest_change = max(biggest_change, np.abs(old_q - Q[s][a]))
        deltas.append(biggest_change)

    # calculate values for each state (just to print and compare)
    # V(s) = max[a]{ Q(s,a) }
    # V = {}
    # for s in policy.keys():
    #     V[s] = max_dict(Q[s])[1]

    return deltas

if __name__ == '__main__':


  deltas = monte_carlo()

  plt.plot(deltas)
  plt.show()
