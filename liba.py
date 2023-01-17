import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy

# class State:
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y

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