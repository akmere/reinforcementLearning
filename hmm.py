import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class State:
    def __init__(self, x, y):
        self.x = x
        self.y = y

creditForRenting = 10
costToMove = 2
# number of cars requested and returned at each location are Poisson random variables, (l^n)/(factorial(n)) * e^(-l)
rent1 = 3
rent2 = 4
return1 = 3
return2 = 2
# no more than 20 cars at each location
gamma = 0.9  # discount rate
# continuing finite MDP , time steps are days, state is the number of cars at each location at the end of the day, actions are net numbers of cars moved (max 5)
states = []
actions = np.arange(-5, 6, 1)
for i in range(21):
    for y in range(21):
        states.append(State(i, y))
states = np.array(states)

rewards = np.zeros(states.shape)

for reward in rewards:
    

sv = np.zeros(states.shape)

probs = np.zeros(states.shape + (len(actions),))

print(actions)



# for state in states:
#     oldV = sv
