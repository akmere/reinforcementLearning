import numpy as np
import copy
import matplotlib.pyplot as plt

#target policy and behavior policy
#importance sampling ratio: I[Pi(a|s)/b(a|s)]
# v_pi(s) = isr * G

class State:
    def __init__(self,is_terminal):
        self.is_terminal = is_terminal
        
gamma = 1
p = 0.9
states = [State(True), State(False)]
states = np.array(states)
rewards = np.ones((len(states),))
# rewards = np.array([1, 0])
actions = np.array(["action"])
tries = np.zeros((len(states),len(actions)))
transitional_probs = np.zeros((len(states), len(actions), len(states)))
for state_index, state in enumerate(states):
    for action_index, action in enumerate(actions):
        for end_state_index, end_state in enumerate(states):
            if(not state.is_terminal and not end_state.is_terminal and action == "action"): transitional_probs[state_index, action_index, end_state_index] = p
            elif(not state.is_terminal and end_state.is_terminal and action == "action"): transitional_probs[state_index, action_index, end_state_index] = 1 - p
probs = np.ones((len(states), len(actions)))
# print(transitional_probs)
# print(probs)

def get_state_index(state : State):
    return [vars(state1) for state1 in states].index(vars(state))

def do_step(initial_state : State, action):
    # print(probs[get_state_index(initial_state),:])
    # print((transitional_probs[get_state_index(initial_state),list(actions).index(action),:]))
    return np.random.choice(states, p=probs[get_state_index(initial_state),:] * (transitional_probs[get_state_index(initial_state),list(actions).index(action),:]))

q = np.zeros((len(states), len(actions)))
v = np.zeros((len(states),))

def do_episode():
    current_state = State(False)
    sa = [[current_state, actions[0]]]
    visited_states = [copy.deepcopy(current_state)]
    G = 0
    i = 0
    while(not current_state.is_terminal):
        # G = rewards[get_state_index(current_state)] + gamma * G
        # tries[get_state_index(current_state)] +=1 
        # v[get_state_index(current_state)] = (v[get_state_index(current_state)] * (tries[get_state_index(current_state)] -1) + G)/tries[get_state_index(current_state)]
        chosen_action = np.random.choice(actions, p = probs[get_state_index(current_state)])
        current_state = do_step(current_state, chosen_action)
        sa.append([copy.deepcopy(current_state),copy.deepcopy(chosen_action)])
    sa.reverse()
    return sa
        
every_state = True
for i in range(1):
    G = 0
    visited_sa = []
    sa = do_episode()
    for index, (state, action) in enumerate(sa):        
        G = rewards[get_state_index(state)] + gamma * G
        if(every_state or [get_state_index(state), action] not in list([get_state_index(ss[0]), ss[1]] for ss in sa[index+1:])):
            tries[get_state_index(state), list(actions).index(action)] += 1
            q[get_state_index(state), list(actions).index(action)] = ((q[get_state_index(state), list(actions).index(action)] * (tries[get_state_index(state), list(actions).index(action)] - 1)) + G)/tries[get_state_index(state), list(actions).index(action)]
            visited_sa.append([get_state_index(state), action])
            
print(q)