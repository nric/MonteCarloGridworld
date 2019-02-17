#%%
import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid
from utils import max_dict, print_values, print_policy

CONST_GAMMA = 0.9
CONST_ACTION_LST = ('U','D','L','R')
CONST_N_EPISODES = 10000
CONST_EPSILON = 0.1

def epsilon_action(a,eps = 0.1):
    """ 
    epsilon greedy. Chooses either action a or an radom action with epsioln probablility
    Args: None
    Return: None
    """
    rand_number = np.random.random()
    if rand_number < (1-eps):
        return a
    else:
        return np.random.choice(CONST_ACTION_LST)

def play_a_game(grid,policy):
    """ 
    Function for playing a gridworld game
    Args: grid: the grid world object, policy: the current policy matrix 
    Return: statematrix, actions and gain (s,a,g)
    Examples: 
        states_actions_returns = play_game(grid, policy) 
        for s, a, G in states_actions_returns:
    """
    #set to the start state.
    s = (2,0)
    grid.set_state(s)
    a = epsilon_action(policy[s], CONST_EPSILON) #choose an action epsilon greedy

    #play till game over. g is always one step behind
    states_actions_rewards = [(s, a, 0)]
    while True:
        r = grid.move(a)
        s = grid.current_state()
        if grid.game_over():
            states_actions_rewards.append((s, None, r))
            break
        else:
            a = epsilon_action(policy[s], CONST_EPSILON)
            states_actions_rewards.append((s, a, r))    

    #caluculate the gains by working backwards form the terminal state
    G = 0
    state_action_gain = []
    first = True
    for s,a,r in reversed(states_actions_rewards):
        if first:
            first = False
        else:
            state_action_gain.append((s,a,G))
        G = r + CONST_GAMMA*G
    state_action_gain.reverse() #the loop above reverses the order
    return state_action_gain


def monte_carlo(grid):
    """ 
    Functions runs games and updates the value function monte carlo style
    Args: grid: The grid world
    Return: Value gunction, Policy fuction, Delta list
    """
    #Create a random policy
    policy = {}
    for s in grid.actions.keys():
        policy[s] = np.random.choice(CONST_ACTION_LST)
    
    #Init Q(s,a) function and return
    Q = {}
    returns =  {}
    for s in grid.non_terminal_states():
        Q[s] = {}
        for a in CONST_ACTION_LST:
            Q[s][a] = 0
            returns[(s,a)] = []
    
    # protocol changes of the Q values in each episode
    deltas = []
    # run the monte carlo approimation for a specified amount of times
    for t in range(CONST_N_EPISODES):
        if t % 1000 == 0 :
            print(t)

        biggest_change = 0
        states_actions_returns = play_a_game(grid, policy)

        # calculate Q(s,a)
        seen_state_action_pairs = set()
        for s, a, G in states_actions_returns:
        # check if we have already seen, otherwise skip (first-visit insead of every visit)
            sa = (s, a)
            if sa not in seen_state_action_pairs:
                returns[sa].append(G)
                old_q = Q[s][a]
                Q[s][a] = np.mean(returns[sa])
                biggest_change = max(biggest_change, np.abs(old_q - Q[s][a]))
                seen_state_action_pairs.add(sa)
            deltas.append(biggest_change)

        # calculate new policy p[s] = argmax[a]{Q(s,a)}
        for s in policy.keys():
            a, _ = max_dict(Q[s])
            policy[s] = a
    
    # calculate values for each state (just to print and compare)
    # V(s) = max[a]{ Q(s,a) }
    V = {}
    for s in policy.keys():
        V[s] = max_dict(Q[s])[1] #this function was givin by utils and is basically argmax for a python dict
    
    return V, policy, deltas

def main():
    grid = standard_grid(obey_prob=1.0,step_cost=None)
    print_values(grid.rewards, grid)
    V, Policy, Deltas = monte_carlo(grid)
    print_values(V,grid)
    print_policy(Policy,grid)
    plt.plot(Deltas)
    plt.show()

main()


#%%
