#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# TEAM MEMBERS

# Alejandro Jarabo Peñas
# 19980430-T472
# aljp@kth.se

# Xavi ...
# XXXXXXXX-ZXXX
# xavi@kth.se

# Load packages
import eligibility_sarsa as es
import numpy as np
import gym
import matplotlib.pyplot as plt

# HELPER FUNCTIONS
def running_average(x, N):
    ''' Function used to compute the running mean
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y

# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()
nA = env.action_space.n      # tells you the number of actions

# Initialize Fourier Linear Approx Class
etas = [[0,1],[1,0],[1,1]]
fla = es.FourierLinearApprox(etas,nA)
fla.show()

# Training hyperparameters
elig_lambda = 1
gamma = 1
alpha = 0.001
epsilon = 0
n_episodes = 100
max_iters = 200



# Train with Eligibility SARSA
print('Training Eligibility SARSA')
print('Hyperparameters: ')
print('elig_lambda = ',elig_lambda)
print('gamma = ',gamma)
print('alpha = ',alpha)
print('epsilon = ',epsilon)
print('n_episodes = ',n_episodes)
print('max_iters = ',max_iters)
episodes_reward = es.eligibility_sarsa(env,fla,elig_lambda,gamma,alpha,epsilon,n_episodes,max_iters)

# Final parameters matrix w
print('----------------------------------------')
print('After training over episodes')
print('Params. matrix w :\n',fla.w)
print('----------------------------------------')

# Plot statistics
plt.plot([i for i in range(1, n_episodes+1)], episodes_reward, label='Episode reward')
plt.plot([i for i in range(1, n_episodes+1)], running_average(episodes_reward, 10), label='Average episode reward')
plt.xlabel('Episodes')
plt.ylabel('Total reward')
plt.title('Total Reward vs Episodes')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
