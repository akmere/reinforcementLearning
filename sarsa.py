import numpy as np
import matplotlib.pyplot as plt
import copy
import datetime

states = []

class State:
    def __init__(self, x, y, wind = [0,0]):
        self.x = x
        self.y = y
        self.wind = wind
    def __eq__(self, __o: object) -> bool:
        if self.x == __o.x and self.y == __o.y: return True
        else: return False

for y in range(6):
    for x in range(10):
        wind = [0,0]
        if x in [3,4,5,8]: wind = [0,-1]
        elif x in [6,7]: wind = [0,-2]
        state = State(x, y, wind)
        states.append(state)

max_x = 9
max_y = 5

def get_state_index(state : State, states):
    index = None
    try:
        index = list(states).index(state)
    except:
        pass
    return index

def get_action_index(action, actions):
    index = None
    try:
        index = list(actions).index(action)
    except:
        pass
    return index

rewards = np.array([0 if state.x == 7 and state.y == 3 else -1 for state in states])
terminal_states = [states[get_state_index(State(7,3), states)]]
starting_state = states[get_state_index(State(0,3), states)]
actions = [[-1,0], [1,0], [0,-1], [0,1]]
# actions = [[-1,0], [1,0], [0,-1], [0,1], [-1,1], [-1,-1], [1,1], [1,-1]]
# actions = [[-1,0], [1,0], [0,-1], [0,1], [-1,1], [-1,-1], [1,1], [1,-1], [0,0]]
q = np.zeros((len(states),len(actions)),dtype=float)
pi = np.full((len(states),len(actions)),1/len(actions), dtype=float)

def get_deterministic_policy(q: np.array):
    policy = np.array([[1/len([a for a in action if a == max(action)]) if action_value == max(action) else 0 for action_value in action] for action in q])
    return policy

def get_stochastic_policy(q: np.array, e):
    policy = np.array([[1/len([a for a in action if a == max(action)]) if action_value == max(action) else 0 for action_value in action] for action in q])
    return policy

def draw_map(state : State, states):
    for y in range(6):
        for x in range(10):
            if(state.x == x and state.y == y): print("x",end="")
            else: print("-",end="")
        print("")
    print("")

def do_step(state: State, action, states):
    new_state = copy.deepcopy(state)
    x_wind_move = state.wind[0]
    y_wind_move = state.wind[1]
    # x_wind_move = 0 if state.wind[0] == 0 else np.random.choice([state.wind[0]-1,state.wind[0],state.wind[0]+1])
    # y_wind_move = 0 if state.wind[1] == 0 else np.random.choice([state.wind[1]-1,state.wind[1],state.wind[1]+1])
    new_state.x += x_wind_move + action[0]
    new_state.y += y_wind_move + action[1]
    new_state.x = max(new_state.x, 0)
    new_state.x = min(new_state.x, max_x)
    new_state.y = max(new_state.y, 0)
    new_state.y = min(new_state.y, max_y)
    new_state_index = get_state_index(new_state, states)
    if(get_state_index(new_state, states) == None): return starting_state
    else: return states[new_state_index]

def get_appropriate_action(state, states, policy):
    state_index = get_state_index(state, states)
    chosen_action_index = np.random.choice(len(actions), p=policy[state_index])
    return actions[chosen_action_index]

def do_episode(current_state, states, q, policy, alpha, gamma, e, draw=False, current_action = None, target_policy = None, improve_policy = True):
    if(current_action == None): current_action = get_appropriate_action(current_state, states, policy)
    previous_sa = (copy.deepcopy(current_state), copy.deepcopy(current_action))
    current_state_index = get_state_index(current_state, states)
    current_action_index = get_action_index(current_action, actions)
    steps = 0
    while(current_state not in terminal_states):
        steps+=1
        if(draw): draw_map(current_state, states)
        current_state = do_step(current_state, current_action, states)
        current_action = get_appropriate_action(current_state, states, policy)
        current_state_index = get_state_index(current_state, states)
        current_action_index = get_action_index(current_action, actions)
        reward = rewards[current_state_index]
        previous_state, previous_action = previous_sa
        previous_state_index = get_state_index(previous_state, states)
        previous_action_index  = get_action_index(previous_action, actions)
        # if(previous_state.x == 6): 
        #     print(f"P x: {previous_state.x}, y: {previous_state.y}, action: {previous_action}, action_index: {previous_action_index}, q[current_state_index]: {q[current_state_index]}")
        #     print(f"N x: {current_state.x}, y: {current_state.y}, action: {current_action}, action_index: {current_action_index}, q[current_state_index]: {q[current_state_index]}")
        if(improve_policy): 
            # q[previous_state_index,previous_action_index] = (q[previous_state_index,previous_action_index] * (1-alpha)) + alpha * (reward + gamma * q[current_state_index,current_action_index])
            q[previous_state_index,previous_action_index] = (q[previous_state_index,previous_action_index] * (1-alpha)) + alpha * (reward + gamma * np.amax(q[current_state_index], 0))
            maximizing_indices = [index for index, value in enumerate(q[previous_state_index]) if value == max(q[previous_state_index])]
            policy[previous_state_index] = [((1-e)/len(maximizing_indices) + e/len(policy[previous_state_index])) if index in maximizing_indices else e/len(policy[previous_state_index]) for index, value in enumerate(policy[previous_state_index])]
        previous_sa = (copy.deepcopy(current_state), copy.deepcopy(current_action))
    if(draw): draw_map(current_state, states)
    return steps
        
steps = []
sum_of_steps = []
means = []
episodes = []
for i in range(1000):
    # print(f"{i}: {do_episode(starting_state, states, q, pi, alpha=0.1,gamma=1,e=0.05)}")
    steps.append(do_episode(starting_state, states, q, pi, improve_policy=True, alpha=0.5,gamma=1,e=0.1, draw=False))
    means.append(np.mean(steps))
    episodes.append(i)
    sum_of_steps.append(sum(steps))
    print(f"{i}: {steps[i]}")
    
plt.plot(means, "r")
# plt.plot(steps, "b")
plt.yscale('log')
# plt.plot(sum_of_steps, episodes)
plt.show()


