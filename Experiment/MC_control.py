import numpy as np
import gymnasium as gym
import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import logging


'''
Create an instance of openAI gym's Black Jack environment, which has:

        state = tuple(current sum of player's cards, 
                      dealer's face-up card, 
                      whether or not the player has a usable Ace)   # means Ace & 10 card 
'''
env = gym.make('Taxi-v3')
logger = logging.getLogger(__name__)
logging.basicConfig(filename='taxi.log', encoding='utf-8', level=logging.DEBUG)

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


def generate_episode(env, Q, epsilon, nA):
    '''
    Generates an episode using policy = epsilon-greedy(Q)
    ___Arguments___
        env     : openAI gym environment
        Q       : action value function table
        epsilon : p(equiprobable random policy)  &&  1 - epsilon : p(greedy policy)
        nA      : # of actions
    '''
    episode = []
    state, info = env.reset()  # Get Initial State (S0)

    while True:
        # Generate action with epsilon-greedy policy
        action = np.random.choice(np.arange(nA), p=get_probs(Q[state], epsilon, nA)) \
            if state in Q else env.action_space.sample()
        next_state, reward, done, truncated, info = env.step(action)
        episode.append((state, action, reward))
        state = next_state

        if done:
            break

    return episode

def discountedReturn(states_actions_rewards, GAMMA):
    r = 0
    for index, (s, a, r) in enumerate(states_actions_rewards):
        r += ((GAMMA ** index) * r)
    return r


def update_Q(env, episode, Q, deltas, alpha, gamma):
    '''
    Updates Q using the most recently generated episode
    ___Arguments___
        env     : openAI gym environment
        episode : most recently generated episode
        Q       : action value function table
        alpha   : constant rate for updating action value functions in Q
        gamma   : reward discount rate
    '''

    states_actions_returns = []
    seen_state_action_pairs = set()

    for index, (s, a, R) in enumerate(episode):
        # first-visit Monte Carlo optimization
        sa = (s, a)

        if sa not in seen_state_action_pairs:
            G = discountedReturn(episode[index:], gamma)
            states_actions_returns.append((s, a, G))

    biggest_change = 0

    for s, a, G in states_actions_returns:
        # first-visit Monte Carlo optimization
        old_q = Q[s][a]
        # calculate Q(s,a)
        Q[s][a] = Q[s][a] + (alpha * (G - Q[s][a]))
        biggest_change = max(biggest_change, np.abs(old_q - Q[s][a]))
        deltas.append(biggest_change)

    return deltas, Q


def mc_control(env, num_episodes, alpha, gamma=1.0,
               eps_start=1.0, eps_decay=0.9999, eps_min=0.05):
    '''
    Executes Constant-Alpha Monte Carlo Control, using epsilon-greedy policy for each episode
    ___Arguments___
        env         : openAI gym environment
        num_episode : total # of episodes to iterate over
        alpha       : constant rate for updating action value functions in Q
        gamma       : reward discount rate
        eps_start   : initial epsilon value
        eps_decay   : rate at which epsilon will decay after each episode
        eps_min     : smallest allowable value of epsilon
                      (epsilon will stop decaying at this value and stay constant afterwards)
    '''
    nA = env.action_space.n
    Q = defaultdict(lambda: np.zeros(nA))
    deltas = []
    epsilon = eps_start

    for i_episode in range(num_episodes):
        # 1. Update epsilon:
        epsilon = max(epsilon * eps_decay, eps_min)

        # 2. Generate episode:
        episode = generate_episode(env, Q, epsilon, nA)

        print(i_episode, epsilon, len(episode))
        logger.debug('%s %s %s', str(i_episode), str(epsilon), str(len(episode)))

        # 3. Update Q
        deltas, Q = update_Q(env, episode, Q, deltas, alpha, gamma)

        # 4. Monitor progress:
        if i_episode % 1000 == 0:
            print('\rEpisode {}/{}.'.format(i_episode, num_episodes), end="")
            logger.info(Q)
            sys.stdout.flush()

    # 5. Construct a greedy Policy using the optimized Q over all iterations
    policy = dict((state, np.argmax(action)) for state, action in Q.items())

    return policy, Q, deltas


'''Execute Constant-Alpha Monte Carlo Control'''
num_episodes = 500000
alpha = 0.02
policy, Q, deltas = mc_control(env, num_episodes, alpha)

'''Plot results using helper file provided by Udacity Deep Reinforcement Learning Nanodegree'''
from plot_utils import plot_blackjack_values, plot_policy

# obtain the corresponding state-value function
V = dict((k, np.max(v)) for k, v in Q.items())

# plot the state-value function
plot_blackjack_values(V)

# plot the policy
plot_policy(policy)

plt.plot(deltas)
plt.show()