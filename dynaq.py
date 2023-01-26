import numpy as np
import matplotlib.pyplot as plt
import copy
import datetime

class State:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __eq__(self, __o: object) -> bool:
        return self.x == __o.x and self.y == __o.y
        
states = []

for j in range(6):
    for i in range(9):
        if(i == 2 and j in [1,2,3]): continue
        if(i == 5 and j == 4): continue
        if(i == 7 and j in [0,1,2]): continue
        states.append(State(i,j))

actions = [[-1,0], [0,-1],[1,0],[0,1]]
q = np.zeros((len(states),len(actions)))
pi = np.full((len(states),len(actions)), 1/len(actions))

def get_state_index(state: State, states):
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

starting_states = [states[get_state_index(State(0,2), states)]]
terminal_states = [states[get_state_index(State(8,0), states)]]
rewards = np.zeros((len(states), ))
rewards[get_state_index(State(8,0), states)] = 1

model = {}

def get_deterministic_policy(q):
    return np.array([[1/len([x for x in row if x == max(row)]) if field==max(row) else 0 for field in row] for row in q])

def get_appropriate_action(state, policy):
    state_index = get_state_index(state, states)
    return actions[np.random.choice(range(len(policy[state_index])), p=policy[state_index])]

def do_step(state, action, states):
    new_state = copy.deepcopy(state)
    new_state.x += action[0]
    new_state.y += action[1]
    if(get_state_index(new_state,states) == None): return state
    else: return new_state
    
def do_episode(current_state, states, actions, q, policy, alpha, e, gamma, planning_iterations,current_action = None):
    if(current_action == None): current_action = get_appropriate_action(current_state,policy)
    steps = 0
    while(current_state not in terminal_states):
        # print(vars(current_state))
        previous_state = copy.deepcopy(current_state)
        previous_action = copy.deepcopy(current_action)
        previous_state_index = get_state_index(previous_state, states)
        previous_action_index = get_action_index(previous_action, actions)
        current_state = do_step(current_state, current_action, states)
        current_action = get_appropriate_action(current_state, policy)
        current_state_index = get_state_index(current_state, states)
        current_action_index = get_action_index(current_action, actions)
        reward = rewards[current_state_index]
        model[(previous_state_index, previous_action_index)] = (current_state_index, reward)
        # print(reward)
        # reward = 0
        q[previous_state_index,previous_action_index] = q[previous_state_index,previous_action_index] + alpha * ((reward + gamma * np.amax(q[current_state_index],0)) - q[previous_state_index,previous_action_index])
        # print(q[previous_state_index,previous_action_index])
        maximizing_indices = [index for index, a in enumerate(q[previous_state_index]) if a == max(q[previous_state_index])]
        policy[previous_state_index] = [(1-e)/len(maximizing_indices) + e/len(q[previous_state_index]) if index in maximizing_indices else e/len(q[previous_state_index]) for index, row in enumerate(q[previous_state_index])]
        steps+=1
        for m in range(planning_iterations):
            random_key = list(model.keys())[np.random.choice(range(len(model.keys())))]
            next_state_index = model[random_key][0]
            modeled_reward = model[random_key][1]
            q[random_key[0],random_key[1]] = q[random_key[0],random_key[1]] + alpha * ((modeled_reward + gamma * np.amax(q[next_state_index],0)) - q[random_key[0],random_key[1]])
            
             
    return steps
        
print(rewards)

steps = []
model = {}
now = datetime.datetime.now()
for i in range(50):
    # print(i)
    steps.append(do_episode(starting_states[0], states, actions, q, pi, 0.1, 0.1, 0.95, 5))
print(datetime.datetime.now() - now)
plt.plot(steps, 'b')
plt.yscale('log')
plt.show()