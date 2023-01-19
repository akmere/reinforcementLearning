import numpy as np
import matplotlib.pyplot as plt
import copy
import itertools
import keyboard
import datetime

class State:
    def __init__(self, grid, o_turn):
        self.grid = np.array(grid)
        self.o_turn = o_turn
        self.winner = -1
    def __eq__(self, __o: object):
        return np.array_equal(self.grid, __o.grid) and self.o_turn == __o.o_turn

initial_state = State([[0,0,0],[0,0,0],[0,0,0]], True)
states = [initial_state]
actions = np.array(list(itertools.product((0,1,2),(0,1,2))))
probs = {}
q = {}
gamma = 0.5
a = 0.15

f = open("ehe.txt", "a")

def keywithmaxval(d):
     """ a) create a list of the dict's keys and values; 
         b) return the key with the max value"""  
     v = list(d.values())
     k = list(d.keys())
     maximum = max(v)
     
     return k[v.index(max(v))]

def get_state_index(state : State):
    index = False
    try:
        index = list(states).index(state)
    except:
        pass
    return index

def get_action_index(action):
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

def get_appropriate_action(state: State, policy):
    state_index = get_state_index(state)
    if(state_index == False):
        states.append(state)
        state_index = get_state_index(state)
    if (state_index not in policy): policy[state_index] = {}
    available_actions = get_available_actions(state)
    if(len(available_actions) == 0): return False
    if(len(policy[state_index]) == 0):
        for available_action in available_actions:
            policy[state_index][get_action_index(available_action)] = 1/len(available_actions)
    chosen_action = actions[np.random.choice(list(policy[state_index].keys()), p= list(policy[state_index].values()))]
    return chosen_action

def do_episode(starting_state, policy, starting_action = False, draw = False):
    current_state = copy.deepcopy(starting_state)
    current_action = starting_action
    if(current_action == False): current_action = copy.deepcopy(get_appropriate_action(current_state, policy))
    sa = []
    winner = check_winner(current_state)
    while(winner == -1):
        sa.append([copy.deepcopy(current_state), copy.deepcopy(current_action)])
        # if(draw): draw_state(current_state)
        current_state = do_step(current_state, current_action)
        current_action = get_appropriate_action(current_state, policy)
        winner = check_winner(current_state)
    final_reward = 1
    sa.reverse()
    if(draw): print("Last state: ")
    if(draw): draw_state(current_state)
    for state, action in sa:
        final_reward = abs(final_reward) if (winner == 1 and state.o_turn) or (winner == 2 and not state.o_turn) else -abs(final_reward) if (winner == 1 and not state.o_turn) or (winner == 2 and state.o_turn) else 0
        state_index = get_state_index(state)
        action_index = get_action_index(action)
        if(draw): 
            draw_state(state)
            print(f"Turn: {state.o_turn}")
            print(f"Winner: {winner}")
            print(f"Rewards: {final_reward}")
            print(f"Action: {actions[action_index]}")
            print(f"Action index: {action_index}")
        if state_index not in q: q[state_index] = {}
        if(len(q[state_index]) == 0):
            available_actions = get_available_actions(state)
            for action_index_helper in [get_action_index(available_action) for available_action in available_actions]:
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
        values = [(1-0.15)/len(maximizing_indices) if x in maximizing_indices else 0 for x in keys]
        values = [value + 0.15/len(values) for value in values]
        if(draw): print(f"old probs: {policy[state_index]}")
        policy[state_index] = {k:v for k,v in zip(keys, values)}
        if(draw): print(f"new probs: {policy[state_index]}")
        final_reward = gamma * final_reward
        # print(policy[state_index])
        pass

i=1
while(not keyboard.is_pressed('q')):
    print(i)
    do_episode(initial_state, probs)
    i+=1
for state in states:
    f.write(str(vars(state)))
f.write(f"\n{str(actions)}\n")
f.write(str(q))
f.write("\n")
f.write(str(probs))

pi = {}
for state_index in q:
    pi[state_index] = {}
    for action_index in q[state_index]:
        maximum = max(q[state_index].values())
        maximizing_indices = [key for key in q[state_index].keys() if q[state_index][key] == maximum]
        pi[state_index][action_index] = 1/len(maximizing_indices) if action_index in maximizing_indices else 0
do_episode(initial_state, pi,draw=True)
# print(probs[0])
print(actions)