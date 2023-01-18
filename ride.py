import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
import itertools
from datetime import datetime


class State:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __eq__(self, __o: object):
        return (self.x == __o.x and self.y == __o.y)

states = []
starting_states = []
terminal_states = []
gamma = 1
# np.random.seed(int(datetime.now().timestamp()))

# velocities = list([[a, b] for a,b in itertools.product(range(-5,6), range(-5,6))])

for y in range(30):
    for i in range(30):
        if(i > 20 and y < 5):
            # print(" ", end="")
            continue
        new_state = State(i,y)
        if(y==0 ): starting_states.append(new_state)
        elif(i >= 25): terminal_states.append(new_state)
        # if(velocity == [0,0]): print('s' if new_state in starting_states else 'f' if new_state in terminal_states else "x", end="")
        states.append(new_state)
    # print("")
states = np.array(states)
starting_states = np.array(starting_states)
terminal_states = np.array(terminal_states)

rewards = np.zeros((len(states),))
rewards.fill(-1)

actions = [[0, 0], [1, 0], [0, 1], [1, 1],
           [-1, 0], [0, -1], [-1, -1], [-1, 1], [1, -1]]
actions = np.array(actions)
tries = np.zeros((len(states),len(actions)))
probs = np.full((len(states),len(actions)), 1/len(actions))
pi = np.full((len(states),len(actions)), 1/len(actions))
q = np.zeros((len(states),len(actions)))
c = np.zeros((len(states),len(actions)))

def get_state_index(state: State):
    start_time = datetime.now()
    # help_array = [vars(stato) for stato in list(states)]
    index = False
    # if(vars(state) in help_array): index = help_array.index(vars(state))
    try:
        # index = np.where(states == state)[0][0]
        index = list(states).index(state)
    except:
        pass
    # print(index)
    # print(f"get_state_index: {datetime.now()-start_time}")
    return index

def draw_map(current_state: State):
    current_y = states[0].y
    for state in states:
        if(state.y != current_y): print("")
        current_y = state.y
        if(state == current_state): print("*", end="")
        else: print("f" if state in terminal_states else "s" if state in starting_states else "x", end="")
    print("", end="\n\n")
            

def do_step(state : State, action, velocity):
    start_time = datetime.now()
    new_state = copy.deepcopy(state)
    velocity[0] += action[0]
    velocity[1] += action[1]
    velocity[0] = min(5, velocity[0])
    velocity[0] = max(-5, velocity[0])
    velocity[1] = min(5, velocity[1])
    velocity[1] = max(-5, velocity[1])
    new_state.x += velocity[0]
    new_state.y += velocity[1]
    # print(f"do_step before if: {datetime.now()-start_time}")
    if(get_state_index(new_state) == False):
        new_state = np.random.choice(starting_states)
        velocity = [0,0]
    end_time = datetime.now()
    # print(f"do_step: {end_time-start_time}")
    return new_state, velocity

def do_episode(b_policy, pi_policy, current_state = False, random_action = False, draw = False, improve_policy=True, use_proper_q = True):
    if(current_state == False): current_state = np.random.choice(starting_states)
    if(random_action == False): random_action = copy.deepcopy(actions[np.random.choice(len(actions))])
    print(vars(current_state))
    are_b_and_pi_policies_same = np.array_equal(b_policy, pi_policy)
    sa = [[current_state, random_action]]
    count = 0
    velocity = [0,0]
    while(count == 0 or current_state not in terminal_states):
        start_time = datetime.now()
        count+=1
        # print(vars(current_state))
        state_index = get_state_index(current_state)
        random_action = copy.deepcopy(actions[np.random.choice(len(actions), p=b_policy[state_index])])
        # random_action = actions[probs[get_state_index(current_state)].argmax(0)]
        # print(random_action)
        current_state, velocity = do_step(current_state, random_action, velocity)
        sa.append([current_state, random_action])
        end_time = datetime.now()
        # print(f"do_episode while: {end_time-start_time}")
    sa.reverse()

    g = 0
    w = 1
    for state_action in sa:
        if (draw): draw_map(state_action[0])
        state_index = get_state_index(state_action[0])
        action_index = np.where(actions == state_action[1])[0][0]
        g = rewards[state_index] + gamma * g
        c[state_index,action_index] += w
        tries[state_index,action_index] += 1
        if(use_proper_q): q[state_index, action_index] += ((0 if c[state_index, action_index] == 0 else w/c[state_index, action_index])) * (g - q[state_index, action_index])
        else: q[state_index, action_index] = (q[state_index,action_index] * (tries[state_index,action_index] - 1) + g)/tries[state_index,action_index]
        # probs[state_index] = q[state_index].argmax(0)
        maximizing_index = q[state_index].argmax(0)
        if(improve_policy):
            pi_policy[state_index].fill(0)
            pi_policy[state_index,maximizing_index] = 1
        if(not are_b_and_pi_policies_same): 
            b_policy[state_index].fill(0.1/(len(b_policy[state_index])-1))
            b_policy[state_index,maximizing_index] = 0.9
        # if(action_index != maximizing_index): break
        w = w/(b_policy[state_index, action_index]) if b_policy[state_index, action_index] != 0 else 0
        # w = (pi_policy[state_index, action_index] * w)/(b_policy[state_index, action_index]) if b_policy[state_index, action_index] != 0 else 0
        # print(probs[state_index])
        # print(q[state_index].argmax(0))
        # print(probs[state_index])
        # print()
        # print(q[state_index, action_index])
    return count

# probs.fill(1/len(actions))
# pi.fill(1/len(actions))
q = np.zeros((len(states),len(actions)))
c = np.zeros((len(states),len(actions)))
counts = []

for i in range(1000):
    start_time = datetime.now()
    count = do_episode(probs, pi)
    counts.append(count)
    print(f"{i}: {count}, {datetime.now() - start_time}")

means = [np.mean(counts[:index+1]) for index, value in enumerate(counts)]
plt.plot(means, c='r')
plt.xscale("log")
plt.show()
