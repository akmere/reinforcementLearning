import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy


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

for state_index, state in enumerate(states):
    returns.append([])
    for action_index, action in enumerate(actions):
        if (state.current_sum > 19 and action == 'stick'):
            probs[state_index, action_index] = 1
        elif (state.current_sum <= 19 and action == 'hit'):
            probs[state_index, action_index] = 1


def get_state_index(state: State):
    result = [vars(x) for x in states]
    if (vars(state) in result):
        return result.index(vars(state))
    else:
        return -1


def play_episode():
    playing_states = []
    player_state = State(0, 0, False, "")
    player_aces = 0
    dealer_aces = 0
    dealer_sum = 0
    while (True):
        for y in range(2):
            new_card = np.random.randint(1, 14)
            if (new_card == 1):
                player_aces += 1
            if (new_card >= 10 or new_card == 1):
                player_state.current_sum += 10
            else:
                player_state.current_sum += new_card
        for y in range(2):
            new_card = np.random.randint(1, 14)
            if (new_card == 1):
                dealer_aces += 1
            if (new_card >= 10 or new_card == 1):
                dealer_sum += 10
            else:
                dealer_sum += new_card
            if (y == 0):
                player_state.dealer_card = new_card if new_card <= 10 else 10
        decision = "hit"
        while (decision == "hit"):
            if (player_aces > 0):
                player_state.has_usable_ace = True
            if (player_state.current_sum > 11 and player_state.current_sum <= 21):
                playing_states.append(copy.deepcopy(player_state))
            if (player_state.current_sum > 21):
                decision = "stick"
                break
            if (get_state_index(player_state) != -1 and np.random.rand() < probs[get_state_index(player_state)][0]):
                if (player_state.current_sum > 21 and player_aces > 0):
                    player_state.current_sum -= 9
                decision = "stick"
                break
            new_card = np.random.randint(1, 14)
            if (new_card == 1):
                player_aces += 1
            if (new_card >= 10 or new_card == 1):
                player_state.current_sum += 10
            else:
                player_state.current_sum += new_card
        if (player_state.current_sum > 21):
            playing_states.append(State(special="loss"))
            break
        elif (player_state.current_sum == 21 and dealer_sum != 21):
            playing_states.append(State(special="win"))
            break
        elif (player_state.current_sum == 21 and dealer_sum == 21):
            playing_states.append(State(special="draw"))
            break
        while (dealer_sum < 17):
            new_card = np.random.randint(1, 14)
            if (new_card == 1):
                dealer_aces += 1
            if (new_card >= 10 or new_card == 1):
                dealer_sum += 10
            else:
                dealer_sum += new_card
            if (dealer_sum > 21 and dealer_aces > 0):
                dealer_sum -= 9
        if (dealer_sum > 21 or player_state.current_sum > dealer_sum):
            playing_states.append(State(special="win"))
            break
        elif (dealer_sum > player_state.current_sum):
            playing_states.append(State(special="loss"))
            break
        else:
            playing_states.append(State(special="draw"))
            break
    return playing_states


number_of_tries = 500000

for i in range(number_of_tries):
    result = play_episode()
    result.reverse()
    g = 0
    for state in result:
        index = get_state_index(state)
        g = gamma * g + rewards[index]
        returns[index].append(g)
for value_index, value in enumerate(v):
    # print(sum(returns[value_index]))
    v[value_index] = sum(returns[value_index])/len(returns[value_index]
                                                   ) if len(returns[value_index]) != 0 else 0
# print(returns)
print(v)
