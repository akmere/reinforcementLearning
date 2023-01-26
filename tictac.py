import numpy as np
import matplotlib.pyplot as plt
import copy
import itertools
import keyboard
import datetime
import json
from numpyencoder import NumpyEncoder
from os.path import exists
from types import SimpleNamespace


def jsonKeys2int(x):
    if isinstance(x, dict):
            return {int(k):v for k,v in x.items()}
    return x

def convert_keys_to_int(d: dict):
    new_dict = {}
    for k, v in d.items():
        try:
            new_key = int(k)
        except ValueError:
            new_key = k
        if type(v) == dict:
            v = convert_keys_to_int(v)
        new_dict[new_key] = v
    return new_dict

class State:
    def __init__(self, grid, o_turn = True):
        self.grid = np.array(grid)
        self.o_turn = True if len([x for x in self.grid.flatten() if x == 1]) == len([x for x in self.grid.flatten() if x == 2]) else False
    def __eq__(self, __o: object):
        return np.array_equal(self.grid, __o.grid) and self.o_turn == __o.o_turn

# initial_state = State([[0,0,0],[0,0,0],[0,0,0]], True)
# states = []
actions = np.array(list(itertools.product((0,1,2),(0,1,2))))
# probs = {}
# q = {}
# gamma = 1
# a = 0.1
# e = 0.5

def load_data():
    q_return = {}
    probs_return = {}
    states_return = []
    if(exists('q.json')):
        f = open('q.json', "r")
        q_return = json.loads(f.read())
        q_return = convert_keys_to_int(q_return)
    if(exists('probs.json')):
        f = open('probs.json', "r")
        probs_return = json.loads(f.read())
        probs_return = convert_keys_to_int(probs_return)
    if(exists('states.json')):
        f = open('states.json', "r")
        states_return = json.loads(f.read())
        states_return = [State(**state) for state in states_return]
    return states_return, q_return, probs_return
    

# read_files = True
# if(read_files): states, q, probs = load_data()

def keywithmaxval(d):
     """ a) create a list of the dict's keys and values; 
         b) return the key with the max value"""  
     v = list(d.values())
     k = list(d.keys())
     maximum = max(v)
     
     return k[v.index(max(v))]

def get_state_index(state : State, states):
    index = None
    try:
        index = list(states).index(state)
    except:
        pass
    return index

def get_action_index(action, actions):
    return actions.tolist().index(action.tolist())

def draw_state(state : State):
    for i in range(len(state.grid)):
        for y in range(len(state.grid[0])):
            print(state.grid[y,i], end="")
        print("")
    print("")

def check_winner(state: State):
    options = [[[0,0],[0,1],[0,2]],[[1,0],[1,1],[1,2]],[[2,0],[2,1],[2,2]],[[0,0],[1,0],[2,0]],[[0,1],[1,1],[2,1]],[[0,2],[1,2],[2,2]],[[0,0],[1,1],[2,2]],[[0,2],[1,1],[2,0]]]
    blocked_options = 0
    for option in options:
        if(len(set([state.grid[a,b] for a,b in option])) == 1 and state.grid[tuple(option[0])] != 0): return state.grid[tuple(option[0])]
        if((len(set([state.grid[a,b] for a,b in option])) == 2 and not 0 in set([state.grid[a,b] for a,b in option])) or (len(set([state.grid[a,b] for a,b in option])) == 3)): blocked_options+=1
    if(blocked_options == len(options)): return 0 
    return -1

def get_available_actions(state: State):
    available_actions = []
    for y in range(len(state.grid)):
        for i in range(len(state.grid[y])):
            if(state.grid[i,y] == 0): available_actions.append([i,y])
    return np.array(available_actions)
            
def do_step(state: State, action):
    if(action not in get_available_actions(state)): return state
    state = copy.deepcopy(state)
    state.grid[action[0],action[1]] = 1 if state.o_turn else 2
    state.o_turn = not state.o_turn
    return state

def get_appropriate_action(state: State, policy, states):
    state_index = get_state_index(state, states)
    if(state_index == None):
        states.append(state)
        state_index = get_state_index(state, states)
    if (state_index not in policy): policy[state_index] = {}
    available_actions = get_available_actions(state)
    if(len(available_actions) == 0): return False
    if(len(policy[state_index]) == 0):
        for available_action in available_actions:
            policy[state_index][get_action_index(available_action, actions)] = 1/len(available_actions)
    chosen_action = actions[np.random.choice(list(policy[state_index].keys()), p= list(policy[state_index].values()))]
    return chosen_action

def do_episode(starting_state, policy, states, q, a, e, target_policy = False, improve_policy = True, starting_action = False, draw = False):
    current_state = copy.deepcopy(starting_state)
    current_action = starting_action
    if(current_action == False): current_action = copy.deepcopy(get_appropriate_action(current_state, policy, states))
    sa = []
    winner = check_winner(current_state)
    while(winner == -1):
        sa.append([copy.deepcopy(current_state), copy.deepcopy(current_action)])
        # if(draw): draw_state(current_state)
        current_state = do_step(current_state, current_action)
        current_action = get_appropriate_action(current_state, policy, states)
        winner = check_winner(current_state)
    final_reward = 100
    sa.reverse()
    if(draw): print("Last state: ")
    if(draw): draw_state(current_state)
    update_q_and_policy(sa, policy, states, q, a, e, target_policy=target_policy, improve_policy=improve_policy,draw=draw)
    
def update_q_and_policy(sa, policy, states, q, a, e, target_policy = False, improve_policy = True, draw = False):
    winner = check_winner(sa[0][0])
    for state, action in sa:
        final_reward = 100 if (winner == 1 and state.o_turn) or (winner == 2 and not state.o_turn) else 0 if (winner == 1 and not state.o_turn) or (winner == 2 and state.o_turn) else 50
        state_index = get_state_index(state, states)
        action_index = get_action_index(action, actions)
        if(draw): 
            print(f"Turn: {state.o_turn}")
            print(f"Winner: {winner}")
            print(f"Rewards: {final_reward}")
            print(f"Action: {actions[action_index]}")
            print(f"Action index: {action_index}")
        if state_index not in q: q[state_index] = {}
        if(len(q[state_index]) == 0):
            available_actions = get_available_actions(state)
            for action_index_helper in [get_action_index(available_action, actions) for available_action in available_actions]:
                q[state_index][action_index_helper] = 0
        if(draw): 
            print(f"old q[state_index]: {q[state_index]}")
            print(f"Action index: {action_index}")
        q[state_index][action_index] = q[state_index][action_index] + a * (final_reward - q[state_index][action_index])
        if(draw): print(f"new q[state_index]: {q[state_index]}")
        maximum = max(q[state_index].values())
        maximizing_indices = [key for key in q[state_index].keys() if q[state_index][key] == maximum]
        keys = list(q[state_index].keys())
        # print(f"keys: {keys}")
        # print(f"maximizing_indices: {maximizing_indices}")
        if(draw): print(f"old probs: {policy[state_index]}")
        values = [(1-e)/len(maximizing_indices) if x in maximizing_indices else 0 for x in keys]
        values = [value + e/len(values) for value in values]
        if(improve_policy): policy[state_index] = {k:v for k,v in zip(keys, values)}
        if(target_policy != False):
            values_for_target_policy = [(1)/len(maximizing_indices) if x in maximizing_indices else 0 for x in keys]
            target_policy[state_index] = {k:v for k,v in zip(keys, values_for_target_policy)}
        if(draw): 
            print(f"new probs: {policy[state_index]}")
            draw_state(state)
        # final_reward = gamma * final_reward
        # print(policy[state_index])
        
        
def get_deterministic_policy(q_array):    
    pi = {}
    for state_index in q_array:
        keys = list(q_array[state_index].keys())
        maximum = max(q_array[state_index].values())
        maximizing_indices = [key for key in q_array[state_index].keys() if q_array[state_index][key] == maximum]
        values_for_target_policy = [(1)/len(maximizing_indices) if x in maximizing_indices else 0 for x in keys]
        pi[state_index] = {k:v for k,v in zip(keys, values_for_target_policy)}
    return pi
    
# pi = get_deterministic_policy(q)
    
# while(i <= 1):
#     print(i)
#     do_episode(initial_state, probs, states, target_policy=pi)
#     i+=1
    
# f = open("states.json", "w")
# f.write(json.dumps([vars(state) for state in states], cls=NumpyEncoder))
# f = open("q.json", "w")
# f.write(json.dumps(q, cls=NumpyEncoder))
# f = open("policy.json", "w")
# f.write(json.dumps(pi, cls=NumpyEncoder))

# custom_state = State(grid=[[2,0,0],[0,0,0],[2,1,1]], o_turn=True)
# for i in range(1): do_episode(custom_state, pi, states, improve_policy=False,draw=True)
# print(len(states))

# # print(probs[0])
# print(actions)