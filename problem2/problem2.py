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
import pickle
import numpy as np
import gym
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# HELPER FUNCTIONS
def running_average (x, N):
    ''' Running mean of the last N elements of a vector x'''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y

def plot_opt_policy_or_value_func (fla,type) :
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

def test_lambdas_alphas (fla,N) :
    alphas = [i*(0.05-0.001)/N for i in range(1,N+1)]
    elig_lambdas = [i/N for i in range(N+1)]

    decrease_alpha = False
    # Test several alphas
    elig_lambda = 0.95
    alphas_episodes_reward = []
    for alpha in alphas :
        fla = es.FourierLinearApprox(etas,nA,null_base)
        episodes_reward, episodes_alpha = es.eligibility_sarsa(env,fla,elig_lambda,gamma,alpha,epsilon,n_episodes,max_iters,decrease_alpha,decrease_epsilon,debug)
        alphas_episodes_reward.append(np.mean(episodes_reward))
    # Test several lambdas
    alpha = 0.0075
    elig_lambdas_episodes_reward = []
    for elig_lambda in elig_lambdas :
        fla = es.FourierLinearApprox(etas,nA,null_base)
        episodes_reward, episodes_alpha = es.eligibility_sarsa(env,fla,elig_lambda,gamma,alpha,epsilon,n_episodes,max_iters,decrease_alpha,decrease_epsilon,debug)
        elig_lambdas_episodes_reward.append(np.mean(episodes_reward))

    # Plot comparison of results
    fig = go.Figure()
    # Add traces
    fig.add_trace(
        go.Scatter(x=alphas, y=alphas_episodes_reward, name="Dep. on Learning Rates")
    )
    # Add figure title
    fig.update_layout(
        title_text="Average Reward depending on Alphas"
    )
    
    # Set x-axis title
    fig.update_xaxes(title_text="Average Reward")
    # Set y-axes title
    fig.update_yaxes(title_text="Learning Rate")
    fig.show()

    # Plot comparison of results
    fig = go.Figure()
    # Add traces
    fig.add_trace(
        go.Scatter(x=elig_lambdas, y=elig_lambdas_episodes_reward, name="Dep. on Elig. Lambda")
    )
    # Add figure title
    fig.update_layout(
        title_text="Average Reward depending on Elig. Lambdas"
    )
    
    # Set x-axis title
    fig.update_xaxes(title_text="Average Reward")
    # Set y-axes title
    fig.update_yaxes(title_text="Elig. Lambda")
    fig.show()

# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()
nA = env.action_space.n      # tells you the number of actions

# Initialize Fourier Linear Approx Class
null_base = True
etas_2_uncoupled = [[0,1],[1,0]]
etas_3_coupled = [[0,1],[1,0],[1,1]]
etas_3_coupled_2 = [[0,2],[2,0],[1,1]]
etas_5_coupled = [[0,1],[1,0],[0,2],[2,0],[1,1]]
etas = etas_3_coupled
fla = es.FourierLinearApprox(etas,nA,null_base)
fla.show()

# Training hyperparameters
elig_lambda = 0.95
gamma = 1
alpha = 0.01
epsilon = 0
n_episodes = 50
max_iters = 200
decrease_alpha = True
decrease_epsilon = False
debug = False

# Analyze average total reward of policy as function of alpha and lambda
print('\nAnalyzing average total reward of policy as function of alpha and lambda')
N = 20
test_lambdas_alphas(fla,N)
print('\n--------------------------------------------------')



# Train with Eligibility SARSA
average_episodes_reward = -200
reward_th = -130 
n = 0
while average_episodes_reward < reward_th :
    print('\n',n,'. Average Reward = ',average_episodes_reward,' < ', reward_th, ', training again...')
    fla = es.FourierLinearApprox(etas,nA,null_base)
    episodes_reward, episodes_alpha = es.eligibility_sarsa(env,fla,elig_lambda,gamma,alpha,epsilon,n_episodes,max_iters,decrease_alpha,decrease_epsilon,debug)
    average_episodes_reward = np.mean(episodes_reward[-25:])
    n += 1
print('\nAverage Reward = ',average_episodes_reward,' > -110, training finished!')

# Final parameters matrix w
print('--------------------------------------------------\n')
print('After training over episodes')
W = fla.w.transpose()
N = fla.etas
print('W:\n',W)
print('N:\n',N)
print('\n--------------------------------------------------')

# Plot color optimal policy and value function
plot_opt_policy_or_value_func(fla,'opt_policy')
plot_opt_policy_or_value_func(fla,'value_func')

# Plot statistics
fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('Episodes')
ax1.set_ylabel('Total Reward')
ax1.plot([i for i in range(1, n_episodes+1)], episodes_reward, label='Episode reward',color='blue')
ax1.plot([i for i in range(1, n_episodes+1)], running_average(episodes_reward, 10), label='Average episode reward',color='green')
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('Learning Rate (alpha)')
ax2.plot([i for i in range(1, n_episodes+1)], episodes_alpha, label='Episode learning rate',color='red')
plt.title('Total Reward and Learning Rate vs Episodes')
ax1.legend()
ax2.legend()
plt.grid(alpha=0.3)
plt.show()

# Create pickle file
data = {'W':W, 'N': N}
filename = 'weights.pkl'
print('\nSaving weights and etas to ',filename)
pickle.dump(data, open(filename, "wb" ))


