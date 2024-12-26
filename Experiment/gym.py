from __future__ import print_function, division

import pickle
import matplotlib.pyplot as plt
from utils import max_dict, print_values, print_policy
import numpy as np
import gymnasium as gym
from collections import defaultdict

GAMMA = 1 # discounting
ALPHA = 1/5000 # step size
EPSILON = 0.5 # sprinkle in some randomness for exploration sake
N_EPISODES = 50000

env = gym.make('Taxi-v3')

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


def get_probs(Q_state, epsilon, nA):
    '''
    Obtains action probabilities corresponding to epsilon-greedy policy
    ___Argumnets___
        Q_state : (subset of Q) array containing action-value functions for a single state
        epsilon : p(equiprobable random policy)  &&  1 - epsilon : p(greedy policy)
        nA      : # of actions
    '''
    # 1. Initialize w/ probabilities for non-greedy actions:
    probs = np.ones(nA) * epsilon / nA
    # 2. Set the probability for greedy action:
    greedy_action = np.argmax(Q_state)
    probs[greedy_action] = 1 - epsilon + epsilon / nA

    return probs

def play_game(Q, epsilon):
    states_actions_rewards = []
    episode_over = False
    observation, info = env.reset(seed=42)

    while not episode_over:
        best_action = env.action_space.sample()
        if observation in Q:
            best_action, _ = max_dict(Q[observation])
        a = epsilon_action(best_action, env.action_space.sample(), epsilon)

        observation, reward, terminated, truncated, info = env.step(a)

        # print("action: "+ str(a))
        # print(
        # '[\n',
        #     'observation: ' + str(observation), '\n',
        #     'action: ' + str(a), '\n',
        #     'reward: ' + str(reward), '\n',
        #     ' ', '\n',
        #     'terminated: ' + str(terminated), '\n',
        #     'truncated: ' + str(truncated), '\n',
        #     ' '
        # '\n]')
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


def monte_carlo(eps_start = 1.0, eps_decay = 0.97, eps_min = 0.05):
    nA = env.action_space.n

    try:
        with open('Q.dat', 'rb') as loaded_file:
            Q = pickle.load(loaded_file)
    except FileNotFoundError:
        Q = {}

    N = {}

    # keep track of how much our Q values change each episode so we can know when it converges
    deltas = []
    epsilon = eps_start

    # repeat for the number of episodes specified (enough that it converges)
    for t in range(N_EPISODES):

        if t % 1000 == 0:
            if (len(Q) != 0):
                with open('Q.dat', 'wb') as file:
                    pickle.dump(Q, file)
            print(t, epsilon)
            # 1. Update epsilon:
            epsilon = max(epsilon * eps_decay, eps_min)

        biggest_change = 0
        # generate an episode using the current policy
        states_actions_returns = play_game(Q, epsilon)

        # calculate Q(s,a)
        for s, a, G in states_actions_returns:
            # first-visit Monte Carlo optimization

            try:
                if s not in N:
                    N[s] = {}
                if a not in N[s]:
                    N[s][a] = 1
                else:
                    N[s][a] += 1
            except KeyError:
                print('KeyError N', N, s, a)
            old_q = Q[s][a]
            Q[s][a] = Q[s][a] + (1/N[s][a] * (G-Q[s][a]))
            biggest_change = max(biggest_change, np.abs(old_q - Q[s][a]))
        deltas.append(biggest_change)

    # calculate values for each state (just to print and compare)
    # V(s) = max[a]{ Q(s,a) }
    # V = {}
    # for s in policy.keys():
    #     V[s] = max_dict(Q[s])[1]

    return deltas

if __name__ == '__main__':

  # print rewards
  # print("rewards:")
  # print_values(grid.rewards, grid)

  deltas = monte_carlo()

  # print("final values:")
  # print_values(V, grid)
  # print("final policy:")
  # print_policy(policy, grid)

  plt.plot(deltas)
  plt.show()
