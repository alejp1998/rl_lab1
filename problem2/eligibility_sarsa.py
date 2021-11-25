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
import time

class FourierLinearApprox :
    def __init__(self, etas, nA, null_base = True):
        ''' Constructor of eligibility SARSA '''
        # Order of the Fourier basis functions
        self.order = len(etas)
        if null_base : 
            # Basis function parameters
            self.etas = np.array([[0,0]] + etas)
            # Parameters matrix
            self.w = np.zeros((len(etas) + 1,nA))
        else : 
            # Basis function parameters
            self.etas = np.array(etas)
            # Parameters matrix
            self.w = np.zeros((len(etas),nA))
        # Number of available actions
        self.nA = nA
        

    def basis_functions (self, s) :
        ''' Compute basis functions vector for a given state '''
        return np.array([cos(pi*np.dot(np.array(self.etas[i]),s)) for i in range(len(self.etas))])
    
    def Qw (self, s, a) :
        ''' Compute Q for a given state and action '''
        return np.dot(self.w[:,a],self.basis_functions(s))
    
    def grad_Qwa (self, s) :
        ''' Compute grad w.r.t. w_a of Q(s,a) for a given state and action '''
        return self.basis_functions(s)
    
    def update_weights (self,m,v,delta_t,z_alpha) :
        ''' Update approx. parameters'''
        # Update weights
        self.w = self.w + m*v + delta_t*z_alpha
    
    def show(self) :
        print('--------------------------------------------------\n')
        print('Fourier Linear Approx. Description')
        print('Order = ',self.order)
        print('Basis vectors : \n',self.etas)
        print('# Actions = ',self.nA)
        print('Params. matrix w :\n',self.w)
        print('\n--------------------------------------------------')

def scale_state_variables(s, low, high) :
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

def eligibility_sarsa(env, fla, elig_lambda=1, gamma=1, alpha=0.001, epsilon=0, n_episodes=100, max_iters=200, decrease_alpha = False, decrease_epsilon = False,  debug = False) :
    ''' Finds solution using eligibility SARSA
        :input Gym env            : Environment for which we want to find the best policy
        :input float elig_lambda  : The eligibility factor.
        :input float gamma        : The discount factor.
        :input float alpha        : The initial learning rate.
        :input float epsilon      : The nitial exploring probability.
        :input int n_episodes     : # of episodes to simulate.
        :input int max_iters      : max. # of steps of each episode.
    '''

    if debug :
        print('\nTraining Eligibility SARSA')
        print('Hyperparameters: ')
        print('elig_lambda = ',elig_lambda)
        print('gamma = ',gamma)
        print('alpha = ',alpha)
        print('epsilon = ',epsilon)
        print('n_episodes = ',n_episodes)
        print('max_iters = ',max_iters)
        print('decrease_alpha = ',decrease_alpha)
        print('decrease_epsilon = ',decrease_epsilon)
        print('debug = ',debug,'\n')

    # Environment lower and upper state space bounds
    low, high = env.observation_space.low, env.observation_space.high

    # Minimum epsilon (min exploration prob)
    epsilon_min = 0.0001
    init_epsilon = epsilon
    decay_delta = 2/3

    # Initial learning rate
    init_alpha = alpha
    alpha_min = 0.0001

    # Reward collected in each episode
    max_episode_reward = -200 #initial max episode reward
    episodes_reward = []
    episodes_alpha = []

    # Iteration over episodes
    for e in range(n_episodes):
        # Reset enviroment data
        done = False
        s = scale_state_variables(env.reset(),low,high)
        episode_reward = 0
        
        # Initialize starting action
        a = choose_action(fla,s,epsilon)
        
        # Initialize eligibility traces
        z = np.zeros((len(fla.etas),fla.nA))

        # Initialize the velocity term
        v = np.zeros((len(fla.etas),fla.nA))

        # Clip the eligibility traces values to avoid exploding gradient
        np.clip(z, -5, 5)

        if debug and (e%10 == 0 or e==n_episodes-1) : 
            print('\n\nEPISODE ',e)
            print('Initial state = ',s)
            print('Initial action = ',a)
            print('Initial z = \n',z)
            print('Initial v = \n',v)
            print('alpha = ',alpha)
            time.sleep(5)

        # For each step in the episode
        for i in range(max_iters):
            # We take one step in the environment
            # Step environment and get next state and make it a feature
            next_s, reward, done, _ = env.step(a)
            next_s = scale_state_variables(next_s,low,high)
            
            # Choose next action
            next_a = choose_action(fla,s,epsilon)

            # Update the eligibility traces
            for action in range(fla.nA) :
                if action == a :
                    z[:,action] = gamma*elig_lambda*z[:,action] + fla.grad_Qwa(s)
                else : 
                    z[:,action] = gamma*elig_lambda*z[:,action]
            
            # Compute temporal difference error
            delta_t = reward + gamma*fla.Qw(next_s,next_a) - fla.Qw(s,a)

            # Learning rate values
            if len(fla.etas) > fla.order :
                alphas = [alpha] + [alpha/np.linalg.norm(fla.etas[i]) for i in range(1,fla.order+1)]
            else :
                alphas = [alpha/np.linalg.norm(fla.etas[i]) for i in range(0,len(fla.etas))]

            # Multiply each column in z by respective learning rate
            z_alpha = z.copy()
            for row in range(len(fla.etas)) :
                z_alpha[row,:] = alphas[row]*z_alpha[row,:]
            
            # Update velocity term with Nesterov Acceleration
            m = 0.5
            v = m*v + delta_t*z_alpha

            # Update weights with Nesterov Acceleration
            fla.update_weights(m,v,delta_t,z_alpha)

            # Update episode rewards 
            episode_reward += reward

            # If the episode is finished, we leave the for loop
            if done :
                break
            
            if debug and (e%10 == 0 or e==n_episodes-1) : 
                print('\nIteration ',i)
                print('Current state = ',s)
                print('State basis functions = ',fla.basis_functions(s))
                print('Current action = ',a)
                print('Next state = ',next_s)
                print('Next action = ',next_a)
                print('Updated z = \n',z)
                print('Updated v = \n',v)
                print('Temporal diff. error = ',delta_t)
                print('Updated weights = \n',fla.w)
                env.render()
                time.sleep(0.05)

            # Update next state and action
            s = next_s
            a = next_a

        # Append total episode collected reward and learning rate
        episodes_reward.append(episode_reward)
        episodes_alpha.append(alpha)
        
        # Decrease learning rate if we are getting close to solution
        if episode_reward > max_episode_reward and decrease_alpha :
            max_episode_reward = episode_reward
            alpha = max(init_alpha*(-max_episode_reward/200),alpha_min)

        # Update epsilon according to exponential decay formula
        if decrease_epsilon :
            epsilon = max(epsilon_min, init_epsilon/((e+1)**(decay_delta)))
            
    
    return episodes_reward, episodes_alpha
