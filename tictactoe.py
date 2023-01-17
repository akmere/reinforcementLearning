import numpy as np
import matplotlib.pyplot as plt
import copy
import itertools

class State:
    def __init__(self, result = -1):
        self.grid = np.zeros((9,))
        self.o_turn = True
        self.result = result
    
states = [State(0), State(1), State(2)]  
terminal_states = [states[0], states[1], states[2]]
probs = []
actions = np.array(range(9))
print(actions)

def get_state_index(state : State):
    try:
        return [vars(state1) for state1 in states].index(vars(state))
    except:
        return False
        

def do_step(state: State, action):
    current_state = copy.deepcopy(state)
    current_state.grid[action] = 1 if current_state.o_turn else 2
    if(get_state_index(current_state) == False): states.append(copy.deepcopy(current_state))
    



