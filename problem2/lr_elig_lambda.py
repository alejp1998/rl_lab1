#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# TEAM MEMBERS

# Alejandro Jarabo Peñas
# 19980430-T472
# aljp@kth.se

# Xavi de Gibert Duart
# 19970105-T477
# xdgd@kth.se

# Load packages
import eligibility_sarsa as es
import numpy as np
import gym
import matplotlib.pyplot as plt

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
elig_lambda = 0.9
gamma = 1
alpha = 0.0075
epsilon = 0
momentum = 0.5
n_episodes = 50
max_iters = 200
decrease_alpha = False
decrease_epsilon = False
debug = False

# Params. to try
N = 20
K = 10
alpha_low, alpha_high = 0.0005, 0.01
elig_lambda_low, elig_lambda_high = 0.5,1
alphas = np.linspace(alpha_low, alpha_high, N)
elig_lambdas = np.linspace(elig_lambda_low, elig_lambda_high, N)

# Test several alphas
elig_lambda = 0.9
alphas_episodes_samples_reward = []
alphas_episodes_avg_reward = []
print('\nComputing rewards for varying Learning Rate and fixed Elig. Lambda = ',elig_lambda)
for alpha in alphas :
    alpha_rewards = []
    for i in range(K) :
        fla = es.FourierLinearApprox(etas,nA,null_base)
        episodes_reward, episodes_alpha = es.eligibility_sarsa(env,fla,elig_lambda,gamma,alpha,epsilon,momentum,n_episodes,max_iters,decrease_alpha,decrease_epsilon,debug)
        alpha_rewards.append(np.mean(episodes_reward))

    alphas_episodes_samples_reward.append(alpha_rewards)
    alphas_episodes_avg_reward.append(np.mean(alpha_rewards))

# Test several lambdas
alpha = 0.0075
elig_lambdas_episodes_samples_reward = []
elig_lambdas_episodes_avg_reward = []
print('\nComputing rewards for varying Elig. Lambda and fixed Learning Rate = ',alpha)
for elig_lambda in elig_lambdas :
    elig_lambda_rewards = []
    for i in range(K) :
        fla = es.FourierLinearApprox(etas,nA,null_base)
        episodes_reward, episodes_alpha = es.eligibility_sarsa(env,fla,elig_lambda,gamma,alpha,epsilon,momentum,n_episodes,max_iters,decrease_alpha,decrease_epsilon,debug)
        elig_lambda_rewards.append(np.mean(episodes_reward))

    elig_lambdas_episodes_samples_reward.append(elig_lambda_rewards)
    elig_lambdas_episodes_avg_reward.append(np.mean(elig_lambda_rewards))

# Compute confidence intervals
ci_alphas, ci_elig_lambdas = [],[]
for i in range(N) :
    # 95% confidence intervals
    ci_alphas.append(1.96 * np.std(alphas_episodes_samples_reward[i]) / alphas_episodes_avg_reward[i])
    ci_elig_lambdas.append(1.96 * np.std(elig_lambdas_episodes_samples_reward[i]) / elig_lambdas_episodes_avg_reward[i])

# Convert to numpy arrays
alphas_episodes_avg_reward = np.array(alphas_episodes_avg_reward)
alphas_episodes_samples_reward = np.array(alphas_episodes_samples_reward)
elig_lambdas_episodes_avg_reward = np.array(elig_lambdas_episodes_avg_reward)
elig_lambdas_episodes_samples_reward = np.array(elig_lambdas_episodes_samples_reward)
ci_alphas = np.array(ci_alphas)
ci_elig_lambdas = np.array(ci_elig_lambdas)

# Reassign fixed values
alpha = 0.0075
elig_lambda = 0.9

# Build plots
fig, axs = plt.subplots(2)
fig.suptitle('Rewards depending on Learning Rate and Elig. Lambda')
axs[0].plot(alphas, alphas_episodes_avg_reward,color='yellow')
axs[0].fill_between(alphas, (alphas_episodes_avg_reward - ci_alphas), (alphas_episodes_avg_reward + ci_alphas))
axs[0].set_title('Varying Learning Rate and fixed Elig. Lambda = '+str(elig_lambda))
axs[1].plot(elig_lambdas, elig_lambdas_episodes_avg_reward,color='magenta')
axs[1].fill_between(elig_lambdas, (elig_lambdas_episodes_avg_reward - ci_elig_lambdas), (elig_lambdas_episodes_avg_reward + ci_elig_lambdas))
axs[1].set_title('Varying Elig. Lambda and fixed Learning Rate = '+str(alpha))
x_axis_titles = ['Alphas','Elig. Lambdas']
for i in range(2):
    axs[i].set(xlabel=x_axis_titles[i],ylabel='Averaged Reward over {} trainings of {} episodes'.format(K,n_episodes))
plt.show()