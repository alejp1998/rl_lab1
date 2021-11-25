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
import plotly.graph_objects as go

# HELPER FUNCTIONS
def running_average(x, N):
    ''' Running mean of the last N elements of a vector x'''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y

def plot_opt_policy_or_value_func(fla,type) :
    # Sample Qw(s,a) values
    x_low, x_high = -1.2, 0.6
    y_low, y_high = -0.07, 0.07
    x = np.linspace(x_low, x_high, 100)
    y = np.linspace(y_low, y_high, 100)

    # Sample values from continuous function
    V = np.zeros((100,100))
    for k1 in range(len(x)) :
        x_i = x[k1]
        for k2 in range(len(y)):
            y_i = y[k2]
            scaled_x_i = (x_i - x_low) / (x_high - x_low)
            scaled_y_i = (y_i - y_low) / (y_high - y_low)
            if type == 'opt_policy' :
                V[(k1,k2)] = np.argmax([fla.Qw((scaled_x_i,scaled_y_i),a) for a in range(fla.nA)])
            elif type == 'value_func' : 
                V[(k1,k2)] = np.max([fla.Qw((scaled_x_i,scaled_y_i),a) for a in range(fla.nA)])

    fig = go.Figure(data =
        go.Contour(
            z=V,
            x=x, # horizontal axis
            y=y, # vertical axis
            colorbar=dict(
                title='Optimal action' if type == 'opt_policy' else 'State Value V(s)',
            )
        ))
    fig.update_layout(
        title='Optimal Policy' if type == 'opt_policy' else 'Value Function',
        xaxis_title='Position',
        yaxis_title='Speed',
    )
    fig.show()

# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()
nA = env.action_space.n      # tells you the number of actions

# Initialize Fourier Linear Approx Class
etas_2_uncoupled = [[0,1],[1,0]]
etas_3_coupled = [[0,1],[1,0],[1,1]]
etas_3_coupled_2 = [[0,2],[2,0],[1,1]]
etas_5_coupled = [[0,1],[1,0],[0,2],[2,0],[1,1]]
etas = etas_3_coupled
fla = es.FourierLinearApprox(etas,nA)
fla.show()

# Training hyperparameters
elig_lambda = 0.9
gamma = 1
alpha = 0.005
epsilon = 0
n_episodes = 100
max_iters = 200
decrease_alpha = True
debug = False

# Train with Eligibility SARSA
print('\nTraining Eligibility SARSA')
print('Hyperparameters: ')
print('elig_lambda = ',elig_lambda)
print('gamma = ',gamma)
print('alpha = ',alpha)
print('epsilon = ',epsilon)
print('n_episodes = ',n_episodes)
print('max_iters = ',max_iters)
print('decrease_alpha = ',decrease_alpha)
print('debug = ',debug,'\n')
episodes_reward, episodes_alpha = es.eligibility_sarsa(env,fla,elig_lambda,gamma,alpha,epsilon,n_episodes,max_iters,decrease_alpha,debug)

# Final parameters matrix w
#plot_value_function(fla)
plot_opt_policy_or_value_func(fla,'opt_policy')
plot_opt_policy_or_value_func(fla,'value_func')
print('--------------------------------------------------\n')
print('After training over episodes')
print('Params. matrix w :\n',fla.w)
print('\n--------------------------------------------------')

# Plot statistics
plt.plot([i for i in range(1, n_episodes+1)], episodes_reward, label='Episode reward')
plt.plot([i for i in range(1, n_episodes+1)], running_average(episodes_reward, 10), label='Average episode reward')
plt.plot([i for i in range(1, n_episodes+1)], episodes_alpha, label='Episode learning rate')
plt.xlabel('Episodes')
plt.ylabel('Total reward')
plt.title('Total Reward vs Episodes')
plt.legend()
plt.grid(alpha=0.3)
plt.show()


