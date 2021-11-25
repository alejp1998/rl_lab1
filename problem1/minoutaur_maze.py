# -*- coding: utf-8 -*-

# TEAM MEMBERS

# Alejandro Jarabo Peñas
# 19980430-T472
# aljp@kth.se

# Xavi de Gibert Duart
# 19970105-T477
# xdgd@kth.se

import numpy as np
import random
import matplotlib.pyplot as plt
import time
from IPython import display

# Implemented methods
methods = ['DynProg', 'ValIter', 'Q-Learning', 'SARSA']

# Some colours
LIGHT_RED    = '#FFC4CC'
LIGHT_GREEN  = '#95FD99'
BLACK        = '#000000'
WHITE        = '#FFFFFF'
LIGHT_PURPLE = '#E8D0FF'
LIGHT_ORANGE = '#FAE0C3'

# Map a color to each cell in the maze
col_map = {
    0: WHITE, 
    1: BLACK, 
    2: LIGHT_GREEN, 
    3: LIGHT_PURPLE
}
# Map a name to each cell
text_map = {
    0: '', 
    1: '', 
    2: 'EXIT', 
    3: 'KEY'
}

class MinotaurMaze:
    # Actions
    WAIT       = 0
    MOVE_LEFT  = 1
    MOVE_RIGHT = 2
    MOVE_UP    = 3
    MOVE_DOWN  = 4

    # Give names to actions
    actions_names = {
        WAIT: "wait",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Prob of approaching/moving away
    P_MOVE_TOWARDS = 0.35
    P_MOVE_AWAY = 0.65

    # Reward values
    STEP_REWARD = -1
    KEY_REWARD = 100
    VICTORY_REWARD = 500
    LOSS_REWARD = -1000
    IMPOSSIBLE_REWARD = -np.Inf


    def __init__(self, maze, minotaur_can_wait = False, key_needed = False):
        """ Constructor of the environment Maze."""
        self.maze                                          = maze
        self.key_needed                                    = key_needed
        self.minotaur_can_wait                             = minotaur_can_wait
        self.actions, self.acts_thomas, self.acts_minotaur = self.__actions()
        self.states, self.subset, self.mapping, self.acts_minotaur_approach, self.acts_minotaur_dist = self.__states()
        self.n_actions                                     = len(self.actions)
        self.n_states                                      = len(self.states)
        self.transition_probabilities                      = self.__transitions()
        self.rewards                                       = self.__rewards()

    def __actions(self):
        actions = {}
        acts_thomas = {} # possible set of actions for each cell
        acts_minotaur = {} # possible set of actions for each cell

        actions[self.WAIT]       = ( 0, 0)
        actions[self.MOVE_LEFT]  = ( 0,-1)
        actions[self.MOVE_RIGHT] = ( 0, 1)
        actions[self.MOVE_UP]    = (-1, 0)
        actions[self.MOVE_DOWN]  = ( 1, 0)
        
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                acts_thomas[i,j] = []
                acts_minotaur[i,j] = []
                for a in actions :
                    # Compute the future position given current (state, action)
                    next_i = i + actions[a][0]
                    next_j = j + actions[a][1]
                    # Is the future position an impossible one ?
                    outside_limits =  (next_i == -1) or (next_i == self.maze.shape[0]) or \
                                        (next_j == -1) or (next_j == self.maze.shape[1])
                    hitting_wall = (self.maze[next_i,next_j] == 1) if not outside_limits else False
                    
                    if (self.maze[i,j] != 1) and (not outside_limits) and (not hitting_wall) :
                        acts_thomas[i,j].append(a)

                    if not outside_limits :
                        if self.minotaur_can_wait :
                            acts_minotaur[i,j].append(a)
                        elif a != self.WAIT :
                            acts_minotaur[i,j].append(a)
                    
        return actions, acts_thomas, acts_minotaur

    def __states(self):
        states = {}
        mapping = {} # mapping from coords to state identifier
        subset = {} # -1 if loss, 0 if safe, 1 if victorius
        acts_minotaur_approach = {}
        acts_minotaur_dist = {}
        
        s = 0
        for key in [1,0] :
            # Assume we already have the key when its not needed
            if not self.key_needed and key == 0 :
                break 

            for i_m in range(self.maze.shape[0]):
                for j_m in range(self.maze.shape[1]):
                    for i_t in range(self.maze.shape[0]):
                        for j_t in range(self.maze.shape[1]):
                            if self.maze[i_t,j_t] != 1:
                                # Possible states
                                states[s] = (i_t,j_t,i_m,j_m,key)
                                mapping[(i_t,j_t,i_m,j_m,key)] = s
                                
                                # Victory states
                                if (self.maze[i_t,j_t] == 2) and ((i_t,j_t) != (i_m,j_m)) and key == 1:
                                    subset[s] = 1
                                # Loss states
                                elif (i_t,j_t) == (i_m,j_m) :
                                    subset[s] = -1
                                # Safe (non-victorius) states
                                else :
                                    subset[s] = 0    

                                # Approaching and distancing minotaur actions
                                diff = (i_t-i_m, j_t-j_m)
                                # Actions to approach Thomas
                                acts_minotaur_approach[s] = []
                                if diff[0] > 0 :
                                    acts_minotaur_approach[s].append(self.MOVE_DOWN)
                                elif diff[0] < 0 :
                                    acts_minotaur_approach[s].append(self.MOVE_UP)
                                if diff[1] > 0 :
                                    acts_minotaur_approach[s].append(self.MOVE_RIGHT)
                                elif diff[1] < 0 :
                                    acts_minotaur_approach[s].append(self.MOVE_LEFT)
                                # Actions to move away from Thomas
                                acts_minotaur_dist[s] = []
                                for act in self.acts_minotaur[(i_m,j_m)] :
                                    if act not in acts_minotaur_approach[s] :
                                        acts_minotaur_dist[s].append(act)

                                # Increase by one state identifier
                                s += 1
                                
        return states, subset, mapping, acts_minotaur_approach, acts_minotaur_dist

    def __transitions(self):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probabilities tensor (S,S,A)
        dimensions = (self.n_states,self.n_states,self.n_actions)
        transition_probabilities = np.zeros(dimensions)

        # Compute the transition probabilities.
        for s in range(self.n_states):
            # Is the state terminal
            is_terminal = self.subset[s] == -1 or self.subset[s] == 1
            # Current coords
            (i_t,j_t,i_m,j_m,key) = self.states[s]
            # Iterate through possible actions for Thomas
            for a in self.acts_thomas[(i_t,j_t)] :
                if is_terminal :
                    transition_probabilities[s, s, a] = 1
                    continue 

                # Compute the future position given current (state, action)
                next_i_t = i_t + self.actions[a][0]
                next_j_t = j_t + self.actions[a][1]
                
                # If we are in the key position we pick it
                if self.maze[next_i_t,next_j_t] == 3 :
                    key = 1

                # Iterate through possible minotaur actions
                minotaur_actions = self.acts_minotaur[(i_m,j_m)]
                for minotaur_action in minotaur_actions :
                    next_i_m = i_m + self.actions[minotaur_action][0]
                    next_j_m = j_m + self.actions[minotaur_action][1]
                    next_s = self.mapping[(next_i_t,next_j_t,next_i_m,next_j_m,key)]
                    # Number of possible approaching/distancing actions
                    Na = len(self.acts_minotaur_approach[s])
                    Nd = len(self.acts_minotaur_dist[s])
                    # Probability of each approaching/distancing action
                    pa = self.P_MOVE_TOWARDS/Na if Nd > 0 else 1/Na
                    pd = self.P_MOVE_AWAY/Nd if Nd > 0 else 0
                    if not self.key_needed :
                        transition_probabilities[next_s, s, a] = 1/len(minotaur_actions)
                    else : 
                        if minotaur_action in self.acts_minotaur_approach[s] :
                            transition_probabilities[next_s, s, a] = pa
                        else :
                            transition_probabilities[next_s, s, a] = pd

        return transition_probabilities

    def __rewards(self):
        """ Computes the reward for every state action pair.
            :return numpy.tensor rewards: tensor of rewards of dimension S*S*A
        """
        # Initialize the rewards tensor (S,A)
        rewards = np.zeros((self.n_states, self.n_actions))

        for s in range(self.n_states):
            # Is the state terminal
            is_terminal = self.subset[s] == -1 or self.subset[s] == 1
            # Current coords
            (i_t,j_t,i_m,j_m,key) = self.states[s]
            # Iterate through posible actions for Thomas
            for a in self.actions : 
                # If the action is not available
                if a not in self.acts_thomas[(i_t,j_t)] :
                    rewards[s,a] = self.IMPOSSIBLE_REWARD
                    continue
                # If we are already in a terminal state
                elif is_terminal :
                    rewards[s,a] = 0
                    continue

                # Compute the future position given current (state, action)
                next_i_t = i_t + self.actions[a][0]
                next_j_t = j_t + self.actions[a][1]

                # Check all possible next states
                minotaur_actions = self.acts_minotaur[(i_m,j_m)]
                for minotaur_action in minotaur_actions :
                    next_i_m = i_m + self.actions[minotaur_action][0]
                    next_j_m = j_m + self.actions[minotaur_action][1]

                    next_s = self.mapping[(next_i_t,next_j_t,next_i_m,next_j_m,key)]

                    # If one of the states is inside LOSS subset
                    if self.subset[next_s] == -1 :
                        rewards[s,a] = self.LOSS_REWARD
                        break
                    # If one of the states is inside VICTORY subset
                    elif self.subset[next_s] == 1 :
                        rewards[s,a] = self.VICTORY_REWARD
                    # If we'll end in the key position and we didn't have it before
                    elif (self.maze[next_i_t,next_j_t] == 3) and key == 0 :
                        rewards[s,a] = self.KEY_REWARD
                    # If one of the states is inside SAFE subset and none of the previous is inside VICTORY
                    elif (self.subset[next_s] == 0) and (rewards[s,a] != self.VICTORY_REWARD) :
                        rewards[s,a] = self.STEP_REWARD

        return rewards

    def move(self, state, action):
        """ Makes a step in the maze, given a current position and an action.
            Thomas: moves to position (j_t,i_t) determined by action
            Minotaur: moves to one of the adjacent cells (j_m+-1,i_m+-1) randomly
            :return int next_state: state (j_t,i_t,j_m,i_m) on the maze that agent transitions to.
        """

        # If the current state is in VICTORY or LOSS subset
        if self.subset[state] == -1 or self.subset[state] == 1 :
            return state
        
        # Current coords
        (i_t,j_t,i_m,j_m,key) = self.states[state] 
        
        # Compute the future position of Thomas given current (state, action)
        next_i_t = i_t + self.actions[action][0]
        next_j_t = j_t + self.actions[action][1]

        # If we are in the key position we pick it
        if self.maze[next_i_t,next_j_t] == 3 :
            key = 1

        # Compute next random minotaur action
        if not self.key_needed :
            minotaur_action = random.choice(self.acts_minotaur[(i_m,j_m)])
        else :
            # We sample a float from uniform distribution
            if np.random.uniform(0,1) < self.P_MOVE_TOWARDS or len(self.acts_minotaur_dist[state]) == 0:
                # If it's lower than P_MOVE_TOWARDS we approach
                minotaur_action = random.choice(self.acts_minotaur_approach[state])
            else:
                # If it's higher we move away
                minotaur_action = random.choice(self.acts_minotaur_dist[state])

        # Compute next minotaur position
        next_i_m = i_m + self.actions[minotaur_action][0]
        next_j_m = j_m + self.actions[minotaur_action][1]    


        next_state = self.mapping[(next_i_t,next_j_t,next_i_m,next_j_m,key)]
        
        return next_state
        
    def simulate(self, start, policy, method):
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods)
            raise NameError(error)

        path = list()
        victory_prob = 1
        if method == methods[0]:
            # Deduce the horizon from the policy shape
            horizon = policy.shape[1]
            # Initialize current state and time
            t = 0
            s = self.mapping[start]
            # Add the starting position in the maze to the path
            path.append(start)
            # Probabilities of being caught for each policy action
            caught_probs = []
            while t < horizon :
                # If we reached a terminal state
                if self.subset[s] == -1 or self.subset[s] == 1 :
                    break

                # Action to take according to policy
                a = int(policy[s,t])
                
                # Prob. of being caught after taking action a in state s
                (i_t,j_t,i_m,j_m,key) = self.states[s]
                minotaur_actions = self.acts_minotaur[(i_m,j_m)]
                if self.rewards[s,a] == self.LOSS_REWARD :
                    caught_prob = 1/len(minotaur_actions)
                else : 
                    caught_prob = 0
                caught_probs.append(caught_prob)

                # Compute next state
                next_s = self.move(s,a)
                
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t += 1
                s = next_s
            
            # Calculate victory prob
            if self.subset[s] != 1 :
                victory_prob = 0
            else :
                victory_prob = 1
                for caught_prob in caught_probs :
                    victory_prob = victory_prob*(1-caught_prob)

        if method in methods[1:] :
            # Initialize current state, next state and time
            t = 1
            s = self.mapping[start]
            # Add the starting position in the maze to the path
            path.append(start)
            # Move to next state given the policy and the current state
            next_s = self.move(s,policy[s])
            # Add the position in the maze corresponding to the next state
            # to the path
            path.append(self.states[next_s])
            # Probabilities of being caught for each policy action
            caught_probs = []
            # Loop while state is not terminal (loss or victory)
            while s != next_s:
                # Update state
                s = next_s

                # Action to take according to policy
                a = int(policy[s])
                
                # Prob. of being caught after taking action a in state s
                (i_t,j_t,i_m,j_m,key) = self.states[s]
                minotaur_actions = self.acts_minotaur[(i_m,j_m)]
                if self.rewards[s,a] == self.LOSS_REWARD :
                    caught_prob = 1/len(minotaur_actions)
                else : 
                    caught_prob = 0
                caught_probs.append(caught_prob)

                # Move to next state given the policy and the current state
                next_s = self.move(s,a)
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t +=1
            # Calculate victory prob
            if self.subset[s] != 1 :
                victory_prob = 0
            else :
                victory_prob = 1
                for caught_prob in caught_probs :
                    victory_prob = victory_prob*(1-caught_prob)

        return path, victory_prob

    def minotaur_cell_probs (self, maze, origin, horizon):
        '''
        Computes the probability of the minotaur to be in a given cell
        after t steps up to t=T=horizon
        '''
        # Maze shape
        rows, cols = maze.shape

        # Initialize the position probabilities tensor (S,S,A)
        dimensions = (horizon+1,rows,cols)
        # cell_probs[t,i,j] = prob of being in cell (i,j) after t steps
        cell_probs = np.zeros(dimensions)

        # Initialize depending on starting position
        cell_probs[0,origin[0],origin[1]] = 1

        # Iterate through time steps
        for t in range(1,horizon+1):
            cond_probs = {}
            k = 0
            # Calculate conditional probs
            for i in range(rows) :
                for j in range(cols) :
                    if cell_probs[t-1,i,j] != 0 :
                        cond_probs[k] = np.zeros(dimensions[1:])
                        for a in self.acts_minotaur[(i,j)] :
                            next_i = i + self.actions[a][0]
                            next_j = j + self.actions[a][1]
                            cond_probs[k][(next_i,next_j)] = 1/len(self.acts_minotaur[(i,j)])
                    k += 1

            # Calculate total prob of each cell as weigthed sum of cond probs
            k = 0
            for i in range(rows) :
                for j in range(cols) :
                    if cell_probs[t-1,i,j] != 0 :
                       cell_probs[t,:,:] += cell_probs[t-1,i,j]*cond_probs[k]
                    k += 1
            
        return cell_probs

    def fixed_path_vict_prob (self,maze,path,cell_probs) :
        '''
        Computes the probability of being victorius for a given fixed
        path inside the maze
        '''
        caught_probs = []
        for t in range(len(path)-1) :
            (i_t,j_t,i_m,j_m,key) = path[t]
            # If the state already victorius
            if path[t][:2] == path[t-1][:2] and maze[(i_t,j_t)] == 2 :
                caught_probs.append(0)
            # If there is still risk of being caught
            else : 
                caught_probs.append(cell_probs[t,i_t,j_t])

        # If the path doesnt end in target cell
        if maze[path[-1][:2]] != 2 :
            victory_prob = 0
        # If the path is potentially victorius
        else: 
            victory_prob = 1
            for caught_prob in caught_probs :
                victory_prob = victory_prob*(1-caught_prob)
                
        return victory_prob


    def show(self):
        print('The states are :')
        print(self.states)
        print('The actions are:')
        print(self.actions)
        print('The mapping of the states:')
        print(self.mapping)
        print('The rewards:')
        print(self.rewards)

def dynamic_programming(env, horizon):
    """ Finds solution (optimal policy) using dynamic programming
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input int horizon        : The time T up to which we solve the problem.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """

    # The dynamic prgramming requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p         = env.transition_probabilities
    r         = env.rewards
    n_states  = env.n_states
    n_actions = env.n_actions
    T         = horizon

    # The variables involved in the dynamic programming backwards recursions
    V      = np.zeros((n_states, T+1))
    policy = np.zeros((n_states, T+1))
    Q      = np.zeros((n_states, n_actions))


    # Initialization
    Q            = np.copy(r)
    V[:, T]      = np.max(Q,1)
    policy[:, T] = np.argmax(Q,1)

    # The dynamic programming bakwards recursion
    for t in range(T-1,-1,-1):
        # Update the value function acccording to the bellman equation
        for s in range(n_states):
            for a in range(n_actions):
                # Update of the temporary Q values
                Q[s,a] = r[s,a] + np.dot(p[:,s,a],V[:,t+1])
        # Update by taking the maximum Q value w.r.t the action a
        V[:,t] = np.max(Q,1)
        # The optimal action is the one that maximizes the Q function
        policy[:,t] = np.argmax(Q,1)
    return V, policy

def value_iteration(env, gamma, epsilon):
    """ Finds solution (optimal policy) using value iteration
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input float gamma        : The discount factor.
        :input float epsilon      : accuracy of the value iteration procedure.
        :return numpy.array V     : Optimal values for every state, dimension S
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S
    """
    # The value itearation algorithm requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p         = env.transition_probabilities
    r         = env.rewards
    n_states  = env.n_states
    n_actions = env.n_actions

    # Required variables and temporary ones for the VI to run
    V   = np.zeros(n_states)
    Q   = np.zeros((n_states, n_actions))
    BV  = np.zeros(n_states)
    # Iteration counter
    n = 0
    # Tolerance error
    tol = (1 - gamma)* epsilon/gamma

    # Initialization of the VI
    for s in range(n_states):
        for a in range(n_actions):
            Q[s, a] = r[s, a] + gamma*np.dot(p[:,s,a],V)
    BV = np.max(Q, 1)

    # Iterate until convergence
    while np.linalg.norm(V - BV) >= tol and n < 200:
        # Increment by one the numbers of iteration
        n += 1
        # Update the value function
        V = np.copy(BV)
        # Compute the new BV
        for s in range(n_states):
            for a in range(n_actions):
                Q[s, a] = r[s, a] + gamma*np.dot(p[:,s,a],V)
        BV = np.max(Q, 1)
        # Show error
        #print(np.linalg.norm(V - BV))

    # Compute policy
    policy = np.argmax(Q,1)
    # Return the obtained policy
    return V, policy

def choose_action(env,Q,s,epsilon) :
    ''' Choose next action based on actual state '''
    (i_t,j_t,i_m,j_m,key) = env.states[s]
    # We sample a float from uniform distribution
    explore = np.random.uniform(0,1) < epsilon
    if explore:
        # If it's lower than epsilon, we select random action (EXPLORE)
        a = random.choice(env.acts_thomas[(i_t,j_t)])
    else:
        # If it's higher we take best action we have learned so far (EXPLOIT)
        a_ind = np.argmax(Q[s,env.acts_thomas[(i_t,j_t)]])
        a = env.acts_thomas[(i_t,j_t)][a_ind]
    return a

def qLearning(env, start, gamma, epsilon, n_episodes=10000, max_iters=100, decay_delta=0, lr_alpha = 2/3) :
    """ Finds solution using Q-learning
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input float gamma        : The discount factor.
        :input float epsilon      : Prob. of selection an action uniformly at random at each step.
        :input int n_episodes     : # of episodes to simulate.
        :input int max_iters      : max. # of steps of each episode.
        :input float exp_decreasing_decay  : rate at which exploration prob. will decay.
        
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S
    """
    # Minimum epsilon (min exploration prob)
    epsilon_min = 0.01
    init_epsilon = epsilon

    # Reward collected in each episode
    init_Vs = []

    # The Q-learning algorithm requires the knowledge of :
    # - State space
    # - Action space
    # - Rewards after taking an action in a state
    r         = env.rewards
    n_states  = env.n_states
    n_actions = env.n_actions

    # Initialize the Q-table to 0
    Q = np.zeros((n_states,n_actions))

    # Initialize table of # visits
    n = np.zeros((n_states,n_actions))

    # Iteration over episodes
    for e in range(n_episodes):
        # Initialize to starting state
        s = env.mapping[start]
        is_terminal = False
        
        # For each step in the episode
        for i in range(max_iters):
            # We choose the next action
            a = choose_action(env,Q,s,epsilon)

            # Increment # of visits of pair [s,a]
            n[s,a] += 1

            # We take the action in the environment 
            next_s = env.move(s,a)

            # Are we already in a terminal state?
            is_terminal = env.subset[next_s] == -1 or env.subset[next_s] == 1
            
            # We update our Q-table using the Q-learning iteration
            lr = 1/(n[s,a]**lr_alpha)
            predict = Q[s,a]
            target = r[s,a] + gamma*max(Q[next_s,:])
            Q[s,a] = Q[s,a] + lr*(target - predict)

            # If the episode is finished, we leave the for loop
            if is_terminal :
                break

            # Update next state
            s = next_s
        
        # Value function of the initial state
        init_V = max(Q[env.mapping[start],env.acts_thomas[(start[0],start[1])]])

        # Update epsilon according to exponential decay formula
        epsilon = max(epsilon_min, init_epsilon/((e+1)**(decay_delta)))
        init_Vs.append(init_V)

    # Compute optimal policy with what we have learned
    policy = [choose_action(env,Q,s,0) for s in range(n_states)]
    
    return Q, policy, init_Vs

def sarsa(env, start, gamma, epsilon, n_episodes=10000, max_iters=100, decay_delta=0, lr_alpha = 2/3) :
    """ Finds solution using SARSA
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input float gamma        : The discount factor.
        :input float epsilon      : Prob. of selection an action uniformly at random at each step.
        :input int n_episodes     : # of episodes to simulate.
        :input int max_iters      : max. # of steps of each episode.
        :input float exp_decreasing_decay  : rate at which exploration prob. will decay.
        
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S
    """

    # Minimum epsilon (min exploration prob)
    epsilon_min = 0.01
    init_epsilon = epsilon

    # Reward collected in each episode
    init_Vs = []

    # The SARSA algorithm requires the knowledge of :
    # - State space
    # - Action space
    # - Rewards after taking an action in a state
    r         = env.rewards
    n_states  = env.n_states
    n_actions = env.n_actions

    # Initialize the Q-table to 0
    Q = np.zeros((n_states,n_actions))

    # Initialize table of # visits
    n = np.zeros((n_states,n_actions))

    # Iteration over episodes
    for e in range(n_episodes):
        # Initialize to starting state
        s = env.mapping[start]
        is_terminal = False

        # Initialize starting action
        a = choose_action(env,Q,s,epsilon)

        # For each step in the episode
        for i in range(max_iters):
            # Increment # of visits of pair [s,a]
            n[s,a] += 1

            # We take the action in the environment 
            next_s = env.move(s,a)
            next_a = choose_action(env,Q,next_s,epsilon)

            # Are we already in a terminal state?
            is_terminal = env.subset[next_s] == -1 or env.subset[next_s] == 1
            
            # We update our Q-table using the SARSA iteration
            lr = 1/(n[s,a]**lr_alpha)
            predict = Q[s,a]
            target = r[s,a] + gamma*Q[next_s,next_a]
            Q[s,a] = Q[s,a] + lr*(target - predict)

            # If the episode is finished, we leave the for loop
            if is_terminal :
                break

            # Update next state and action
            s = next_s
            a = next_a
            
        
        # Value function of the initial state
        init_V = max(Q[env.mapping[start],env.acts_thomas[(start[0],start[1])]])

        # Update epsilon according to exponential decay formula
        epsilon = max(epsilon_min, init_epsilon/((e+1)**(decay_delta)))
        init_Vs.append(init_V)

    # Compute optimal policy with what we have learned
    policy = [choose_action(env,Q,s,0) for s in range(n_states)]
    
    return Q, policy, init_Vs

def draw_maze(maze):
    # Size of the maze
    rows,cols = maze.shape

    # Give a color to each cell
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)]
    # Give a name to each cell
    texted_maze = [[text_map[maze[j,i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('The Maze')
    ax.set_xticks([])
    ax.set_yticks([])

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(
        cellText=texted_maze,
        cellColours=colored_maze,
        cellLoc='center',
        loc=(0,0),
        edges='closed'
    )

    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)

def compute_arrow (maze, current_cell, next_cell) :
    # Size of the maze
    rows,cols = maze.shape

    # Cells height and width
    h = 1.0/rows
    w = 1.0/cols

    # Compute cells difference
    diff = (next_cell[0]-current_cell[0],next_cell[1]-current_cell[1])

    # Arrow characteristics
    x_st = w*current_cell[1] + w/2
    y_st = abs(1 - h*current_cell[0] - h/2)
    x_size = 0.8*(diff[1]*w/2)
    y_size = 0.8*(-diff[0]*h/2)
    
    return x_st, y_st, x_size, y_size

def animate_solution(maze, path, show_arrows=False, fps=1):
    # Size of the maze
    rows,cols = maze.shape

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('Policy simulation')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)]
    # Give a name to each cell
    texted_maze = [[text_map[maze[j,i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(
        cellText=texted_maze,
        cellColours=colored_maze,
        cellLoc='center',
        loc=(0,0),
        edges='closed'
    )
    
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)


    # Update the color at each frame
    key_taken = False
    for i in range(len(path)):
        # NORMAL EVOLUTION
        if path[i][:2] != path[i][2:] :
            # REMOVE COLOR FROM KEY CELL IF WE TAKE IT
            if path[i-1][4] == 0 and path[i][4] == 1 :
                keys_cell = path[i][:2]
                key_taken = True
            if key_taken : 
                grid.get_celld()[(keys_cell)].set_facecolor(WHITE)
                grid.get_celld()[(keys_cell)].get_text().set_text('')
            

            # THOMAS CELLS
            if maze[path[i][:2]] == 2 :
                # VICTORY
                grid.get_celld()[(path[i][:2])].set_facecolor(LIGHT_GREEN)
                grid.get_celld()[(path[i][:2])].get_text().set_text('VICTORY')
            else :
                # IF SAFE
                grid.get_celld()[(path[i][:2])].set_facecolor(LIGHT_ORANGE)
                grid.get_celld()[(path[i][:2])].get_text().set_text('THOMAS')

            # THOMAS ARROWS 
            if show_arrows and i < len(path)-1:
                x_st, y_st, x_size, y_size = compute_arrow(maze,path[i][:2],path[i+1][:2])
                plt.arrow(x_st, y_st, x_size, y_size,width=0.002,color ='orange')
                plt.text(x_st+x_size/2,y_st+y_size/2,str(i),fontsize=14,color='orange')


            # IF HAVE MOVED WE CLEAR PREVIOUS CELL  
            if i > 0 and path[i][0:2] != path[i-1][:2]:
                grid.get_celld()[(path[i-1][:2])].set_facecolor(col_map[maze[path[i-1][:2]]])
                grid.get_celld()[(path[i-1][:2])].get_text().set_text('')
            
            # MINOTAUR CELLS
            grid.get_celld()[(path[i][2:4])].set_facecolor(LIGHT_RED)
            grid.get_celld()[(path[i][2:4])].get_text().set_text('MINOTAUR')
            # CLEAR PREV CELLS ONLY IF THOMAS HASNT MOVED THERE
            if path[i][:2] != path[i-1][2:4] and path[i][2:4] != path[i-1][2:4]:
                grid.get_celld()[(path[i-1][2:4])].set_facecolor(col_map[maze[path[i-1][2:4]]])
                grid.get_celld()[(path[i-1][2:4])].get_text().set_text('')

            # MINOTAUR ARROWS 
            if show_arrows and i < len(path)-1:
                x_st, y_st, x_size, y_size = compute_arrow(maze,path[i][2:4],path[i+1][2:4])
                plt.arrow(x_st, y_st, x_size, y_size,width=0.002,color ='red')
                plt.text(x_st+x_size/2,y_st+y_size/2,str(i),fontsize=14,color='red')
                
        # IF LOSS
        else :
            # GAME OVER
            grid.get_celld()[(path[i][:2])].set_facecolor(LIGHT_RED)
            grid.get_celld()[(path[i][:2])].get_text().set_text('GAME OVER')

        display.display(fig)
        display.clear_output(wait=True)
        time.sleep(1/fps)