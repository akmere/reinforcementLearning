import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from matplotlib import cm
from tkinter import *

class State:
    def __init__(self, current_sum=0, dealer_card=0, has_usable_ace=False, special=""):
        self.current_sum = current_sum
        self.dealer_card = dealer_card
        self.has_usable_ace = has_usable_ace
        self.special = special


states = []
states.append(State(0, 0, False, "win"))
states.append(State(0, 0, False, "loss"))
states.append(State(0, 0, False, "draw"))
for current_sum in range(12, 22):
    for dealer_card in range(1, 11):
        for has_usable_ace in True, False:
            states.append(State(current_sum, dealer_card, has_usable_ace))
states = np.array(states)
rewards = np.zeros(len(states))
rewards[0] = 1
rewards[1] = -1
rewards[2] = 0
actions = np.array(['stick', 'hit'])
returns = []
probs = np.zeros((len(states), len(actions)))
gamma = 1
v = np.zeros(len(states))
q = np.zeros((len(states), len(actions)))
tries = np.zeros((len(states), len(actions)))

for state_index, state in enumerate(states):
    returns.append([])
    for action_index, action in enumerate(actions):
        if (state.current_sum > 19 and action == 'stick'):
            probs[state_index, action_index] = 1
        elif (state.current_sum <= 19 and action == 'hit'):
            probs[state_index, action_index] = 1


def get_card():
    card = np.random.randint(1, 14)
    return card

def get_state_index(state: State):
    result = [vars(x) for x in states]
    if (vars(state) in result):
        return result.index(vars(state))
    else:
        return -1

def play_step(initial_state: State, action):
    dealer_sum = initial_state.dealer_card if initial_state != 1 else 10
    new_state = copy.deepcopy(initial_state)
    if(action == "hit"):
        new_card = get_card()
        if (new_card >= 10 or new_card == 1):
            new_state.current_sum += 10
        else: new_state.current_sum += new_card
        if(new_card == 1): new_state.has_usable_ace = True
        if(new_state.current_sum > 21 and new_state.has_usable_ace == False): return State(special="loss")
        elif(new_state.current_sum > 21 and new_state.has_usable_ace == True):
            new_state.current_sum -= 9
            new_state.has_usable_ace = False
        return new_state
    elif(action == "stick"):
        should_dealer_get_card = True
        while(should_dealer_get_card): 
            new_card = get_card()
            if (new_card >= 10 or new_card == 1): dealer_sum += 10
            else: dealer_sum += new_card
            if(dealer_sum >= 17): should_dealer_get_card = False
    if(dealer_sum > 21 or new_state.current_sum > dealer_sum): return State(special="win")
    elif(dealer_sum == new_state.current_sum): return State(special="draw")
    else: return State(special="loss")

number_of_tries = 50000

for i in range(number_of_tries):
    random_state = states[np.random.randint(3, len(states) - 3)]
    random_action = actions[np.random.randint(len(actions))]
    sa = [[random_state,random_action]]
    current_state = copy.deepcopy(random_state)
    while(current_state.special == ""):
        # random_action = actions[np.random.randint(len(actions))]
        current_state = play_step(current_state, random_action)
        # random_action = "stick" if current_state.current_sum > 19 else "hit"
        random_action = np.random.choice(actions,p = probs[get_state_index(current_state)])
        sa.append([copy.deepcopy(current_state), copy.deepcopy(random_action)])
    sa.reverse()
    g = 0
    for state_action in sa:
        state_index = get_state_index(state_action[0])
        action_index = 1 if state_action[1] == "hit" else 0
        g = gamma * g + rewards[state_index]
        tries[state_index, action_index] += 1
        q[state_index, action_index] = (q[state_index, action_index]*(tries[state_index, action_index]-1) + g)/tries[state_index, action_index]
        probs[state_index].fill(0)
        probs[state_index][np.argmax(q[state_index], axis = 0)] = 1
        
v = np.max(q, axis=1)
x = []
y = []
z = []
l = []
for i in range(3, len(q)-3):
    if (states[i].has_usable_ace):
        print(f"{vars(states[i])}: {v[i]}")
        x.append(states[i].current_sum)
        y.append(states[i].dealer_card)
        z.append(v[i])
x = np.array(x)
y = np.array(y)
z = np.array(z)
ax = plt.figure().add_subplot(projection='3d')
ax.set_xlabel("current_sum")
ax.set_ylabel("dealer_card")
ax.set_zlabel("q")
# ax.plot_wireframe(x, y, z, rstride=2, cstride=2,color='green')
ax.scatter(x,y,z,zdir='z')
# ax.plot_trisurf(x, y, z, cmap = cm.jet)
ax.plot_trisurf(x, y, z, antialiased=True)

plt.show()

