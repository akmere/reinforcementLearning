import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
import time
import json
from numpyencoder import NumpyEncoder

class State:
    def __init__(self, x, y):
        self.x = x
        self.y = y

gamma = 0.9
states = []
for i in range(3):
    for y in range(4):
        newState = State(i, y)
        states.append(newState)
states = np.array(states)
rewards = np.zeros(states.shape)
terminal_states = []
for idx, state in enumerate(states):  
    if (state.x == 2 and state.y == 3):
        rewards[idx] = 1
        terminal_states.append(state)
    elif (state.x == 0 and state.y == 0):
        rewards[idx] = -1
        terminal_states.append(state)
actions = np.array(['up', 'right', 'down', 'left'])
probs = np.zeros((states.shape + actions.shape))
transitional_probs = np.zeros((states.shape + actions.shape + states.shape))

for idx, state in enumerate(states):
    for action_idx, action in enumerate(actions):
        next_state = copy.deepcopy(state)
        if (action == 'right' and next_state.x < 2):
            next_state.x += 1
        if (action == 'left' and next_state.x > 0):
            next_state.x += -1
        if (action == 'up' and next_state.y > 0):
            next_state.y += -1
        if (action == 'down' and next_state.y < 3):
            next_state.y += 1
        new_state_index = -1
        for idx_state, state_help in enumerate(states):
            if (state_help.x == next_state.x and state_help.y == next_state.y):
                new_state_index = idx_state
                next_state = states[new_state_index]
                break
        transitional_probs[idx, action_idx, new_state_index] = 1
probs.fill(0.25)
v = np.zeros(states.shape)
            
def evaluate_policy(states, terminal_states, rewards, actions, probs, transitional_probs, v, gamma = 0.9):
    previous_v = copy.deepcopy(v)
    for idx, state in enumerate(states):
        next_state_value = 0
        if (state not in terminal_states):
            next_state_value = sum(probs[idx].dot(transitional_probs[idx]) * previous_v)
        v[idx] = (rewards[idx] + gamma * next_state_value)
    return v

def evaluate_policy1(states, terminal_states, rewards, actions, probs, transitional_probs, v, theta = 0.1, gamma = 0.9):
    delta = theta + 1
    previous_v = copy.deepcopy(v)
    while(delta > theta):
        delta = 0
        v = evaluate_policy(states, terminal_states, rewards, actions, probs, transitional_probs, v, gamma)
        delta = max(delta, max(previous_v - v))
        previous_v = copy.deepcopy(v)
            

def improve_policy(states, terminal_states, rewards, actions, probs, transitional_probs, v, gamma = 0.9):
    new_probs = copy.deepcopy(probs)
    for idx, state in enumerate(states):
        if(state not in terminal_states):
            next_state_value = 0
            next_states = []
            additions = np.sum(gamma * transitional_probs[idx] * v, 1)
            new_probs[idx][new_probs[idx] != 0] = 0
            max_indices = additions == np.amax(additions)
            lll = len(max_indices[max_indices == True])
            new_probs[idx][additions == np.amax(additions)] = 1/lll
    if(np.any(probs != new_probs)): 
        probs[:] = new_probs
        return False
    probs[:] = new_probs
    return True

def iterate_policy(states, terminal_states, rewards, actions, probs, transitional_probs, v, theta = 0.1, gamma=0.9):
    evaluate_policy1(states, terminal_states, rewards, actions, probs, transitional_probs, v, theta, gamma)
    while(not improve_policy(states, terminal_states, rewards, actions, probs, transitional_probs, v)):
        evaluate_policy1(states, terminal_states, rewards, actions, probs, transitional_probs, v, theta, gamma)
    

def iterate_value(states, terminal_states, rewards, actions, probs, transitional_probs, v, theta=0.00001, gamma=0.9):
    delta = theta
    while (delta >= theta):
        previous_v = copy.deepcopy(v)
        next_probs = copy.deepcopy(probs)
        delta = 0
        for idx_state, state in enumerate(states):
            if (state not in terminal_states):  
                additions = np.sum(gamma * transitional_probs[idx_state] * v, 1)
                next_probs[idx_state][next_probs[idx_state] != 0] = 0
                max_indices = additions == np.amax(additions)
                # print('max_indices: ',max_indices)
                lll = len(max_indices[max_indices == True])
                next_probs[idx_state][additions == np.amax(additions)] = 1/lll
                next_probs[idx_state][probs[idx_state] != 0] = 0
                next_probs[idx_state][max_indices] = 1/len(max_indices[max_indices == True])
                v[idx_state] = np.amax(additions)
            else:
                v[idx_state] = rewards[idx_state]
            delta = max(delta, abs(v[idx_state] - previous_v[idx_state])) 
        probs[:] = next_probs

st=time.time()
# iterate_policy(states, terminal_states, rewards, actions, probs, v, theta=0.00001)
# print(v)
# print(probs)
# print(v)
# print(probs)
# iterate_policy(states, terminal_states, rewards, actions, probs, transitional_probs, v)
# iterate_value(states, terminal_states, rewards, actions, probs, transitional_probs, v)
# print(probs)
# print(v)

print(json.dumps(list(states), default = lambda x: x.__dict__))
print(json.dumps(terminal_states, default = lambda x: x.__dict__))
print(json.dumps(rewards, cls=NumpyEncoder))
print(json.dumps(actions, cls=NumpyEncoder))
print(json.dumps(probs, cls=NumpyEncoder))
print(json.dumps(transitional_probs, cls=NumpyEncoder))
print(json.dumps(v, cls=NumpyEncoder))

et=time.time()
print(et-st)