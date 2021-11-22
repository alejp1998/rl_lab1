# -*- coding: utf-8 -*-

# TEAM MEMBERS

# Alejandro Jarabo Peñas
# 19980430-T472
# aljp@kth.se

# Xavi ...
# XXXXXXXX-ZXXX
# xavi@kth.se

# Load packages
import numpy as np
from numpy import cos, pi

class FourierLinearApprox :
    def __init__(self, etas, nA):
        """ Constructor of eligibility SARSA """
        # Order of the Fourier basis functions
        self.order = len(etas)
        # Basis function parameters
        self.etas = [[0,0]] + etas
        # Number of available actions
        self.nA = nA
        # Parameters matrix
        self.w = np.zeros((nA,len(etas) + 1))

    def basis_functions (self, s) :
        ''' Compute basis functions vector for a given state '''
        return [cos(pi*np.transpose(self.etas[i]).dot(s)) for i in range(self.order)]
    
    def Qw (self, s, a) :
        ''' Compute Q for a given state and action '''
        return s.dot(self.w[a])
    
    def grad_Qwa (self, s, a) :
        ''' Compute Q for a given state and action '''
        return s.dot(self.w[a])
    
    def update_weights (self,alpha,delta_t,z) :
        ''' Update approx. parameters'''
        # Learning rate values
        alphas = [alpha] + [alpha/np.linalg.norm(self.etas[i]) for i in range(self.order)]

        # Multiply each column in z by respective learning rate
        for col in range(self.order+1) :
            z[:,col] = z[:,col].dot(alphas[col])
        
        # Update weights
        self.w = self.w + z.dot(delta_t)

def scale_state_variables(self, s, low=env.observation_space.low, high=env.observation_space.high):
    ''' Rescaling of s to the box [0,1]^2 '''
    x = (s - low) / (high - low)
    return x

def choose_action(fla, s, epsilon) :
    ''' Choose next action based on actual state '''
    # We sample a float from uniform distribution
    explore = np.random.uniform(0,1) < epsilon
    if explore:
        # If it's lower than epsilon, we select random action (EXPLORE)
        a = np.random.randint(0, fla.nA)
    else:
        # If it's higher we take best action we have learned so far (EXPLOIT)
        a = np.argmax([fla.Qw(s,a) for a in range(fla.nA)])
    return a

def eligibility_sarsa(env, fla, elig_lambda=1, gamma=1, alpha=0.001, epsilon=0, n_episodes=100, max_iters=200) :
    """ Finds solution using eligibility SARSA
        :input Gym env            : Environment for which we want to find the best policy
        :input float elig_lambda  : The eligibility factor.
        :input float gamma        : The discount factor.
        :input float alpha        : The initial learning rate.
        :input float epsilon      : The nitial exploring probability.
        :input int n_episodes     : # of episodes to simulate.
        :input int max_iters      : max. # of steps of each episode.
    """

    # Minimum epsilon (min exploration prob)
    epsilon_min = 0.01
    init_epsilon = epsilon

    # Reward collected in each episode
    episodes_reward = []

    # Iteration over episodes
    for e in range(n_episodes):
        # Reset enviroment data
        done = False
        s = scale_state_variables(env.reset())
        episodes_reward[e] = 0

        # Initialize starting action
        a = choose_action(fla,s,epsilon)

        # Initialize eligibility traces
        z = np.zeros((fla.nA,fla.order+1))

        # For each step in the episode
        for i in range(max_iters):
            # We take one step in the environment
            # Step environment and get next state and make it a feature
            next_s, reward, done, _ = env.step(a)
            next_s = scale_state_variables(next_s)

            # Choose next action
            next_a = choose_action(fla,s,epsilon)
            
            # Update the eligibility traces
            for action in range(fla.nA) :
                if action == a :
                    z[action] = z[action].dot(gamma*elig_lambda) + fla.grad_Qwa(s,a)
                else : 
                    z[action] = z[action].dot(gamma*elig_lambda)
            
            # Compute temporal difference error
            delta_t = reward + gamma*fla.Qw(next_s,next_a) - fla.Qw(s,a)

            # Update the parameters vector
            fla.update_weights(alpha,delta_t,z)

            # Update episode rewards 
            episodes_reward[e] += reward

            # If the episode is finished, we leave the for loop
            if done :
                break

            # Update next state and action
            s = next_s
            a = next_a
    
    return episodes_reward
