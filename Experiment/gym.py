from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import matplotlib.pyplot as plt
from utils import max_dict, print_values, print_policy

GAMMA = 0.9 # discounting
ALPHA = 0.01 # step size
EPSILON = 0.2 # sprinkle in some randomness for exploration sake
N_EPISODES = 10000
import numpy as np
import gymnasium as gym
env = gym.make('Taxi-v3')
observation, info = env.reset()

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

def play_game(Q):
    # print('play the game')
    states_actions_rewards = []
    episode_over = False
    observation, info = env.reset(seed=42)

    while not episode_over:
        if observation not in Q:
            Q[observation] = {}

        best_action, _ = max_dict(Q[observation])
        if best_action is None:
            best_action = env.action_space.sample()
        # print("best_action: "+ str(best_action))
        a = epsilon_action(best_action, env.action_space.sample(), EPSILON)
        # print("action: "+ str(a))

        observation, reward, terminated, truncated, info = env.step(a)

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
        # print(observation, reward, terminated, truncated, info)
        episode_over = terminated or truncated
    env.close()
    # print(states_actions_rewards)

    states_actions_returns = []
    seen_state_action_pairs = set()

    for index, val in enumerate(states_actions_rewards):
        # check if we have already seen s
        # first-visit Monte Carlo optimization
        s = val[0]
        a = val[1]
        G = 0
        sa = (s, a)
        if a not in Q[s]:
            Q[s][a] = 0

        if sa not in seen_state_action_pairs:
            G = discountedReturn(states_actions_rewards[index:], GAMMA)
            states_actions_returns.append((s, a, G))
    # print(states_actions_returns)
    return states_actions_returns



def running_mean(x):
    mu = 0
    mean_values = []
    for k in np.arange(0, len(x)):
        mu = mu + (1.0/(k+1))*(x[k] - mu)
        mean_values.append(mu)
    return mean_values

# print(running_mean([1,2,3]))


def monte_carlo():
    # initialize a random policy
    policy = {}
    # for s in grid.actions.keys():
    #     policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)

    # initialize Q(s,a) and returns
    Q = {}
    # states = grid.non_terminal_states()
    # for s in states:
    #     Q[s] = {}
    #     for a in ALL_POSSIBLE_ACTIONS:
    #         Q[s][a] = 0

    # keep track of how much our Q values change each episode so we can know when it converges
    deltas = []
    # repeat for the number of episodes specified (enough that it converges)
    for t in range(N_EPISODES):
        if t % 1000 == 0:
            print(t)

        # generate an episode using the current policy
        biggest_change = 0
        states_actions_returns = play_game(Q)

        # calculate Q(s,a)
        for s, a, G in states_actions_returns:
            # check if we have already seen s
            # first-visit Monte Carlo optimization
            # print(Q[s])
            old_q = Q[s][a]
            # the new Q[s][a] is the sample mean of all our returns for that (state, action)
            Q[s][a] = Q[s][a] + (ALPHA * (G-Q[s][a]))
            biggest_change = max(biggest_change, np.abs(old_q - Q[s][a]))
        deltas.append(biggest_change)

        # calculate new policy pi(s) = argmax[a]{ Q(s,a) }
        # for s in policy.keys():
        #     a, _ = max_dict(Q[s])
        #     policy[s] = a

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
