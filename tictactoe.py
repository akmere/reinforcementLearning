import numpy as np
import matplotlib.pyplot as plt
import copy
import itertools

class State:
    def __init__(self, grid, o_turn):
        self.grid = np.array(grid)
        self.o_turn = o_turn
        self.winner = -1
    def __eq__(self, __o: object):
        return np.array_equal(self.grid, __o.grid) and self.o_turn == __o.o_turn

starting_state = State([[0,0,0],[0,0,0],[0,0,0]], True)
states = [starting_state]
actions = np.array(list(itertools.product((0,1,2),(0,1,2))))
probs = {}
probs[0] = np.full(len(actions), 1/len(actions))

q = {}
r = {}
q[0] = np.full(len(actions),0.0, dtype=float)
a = 0.03

def get_state_index(state : State):
    index = False
    try:
        index = list(states).index(state)
    except:
        pass
    return index

def draw_state(state : State):
    for i in range(len(state.grid)):
        for y in range(len(state.grid[0])):
            print(state.grid[y,i], end="")
        print("")
    print("")

def check_winner(grid):
    options = [[[0,0],[0,1],[0,2]],[[1,0],[1,1],[1,2]],[[2,0],[2,1],[2,2]],[[0,0],[1,0],[2,0]],[[0,1],[1,1],[2,1]],[[0,2],[1,2],[2,2]],[[0,0],[1,1],[2,2]],[[0,2],[1,1],[2,0]]]
    blocked_options = 0
    for option in options:
        if(len(set([grid[a,b] for a,b in option])) == 1 and grid[tuple(option[0])] != 0): return grid[tuple(option[0])]
        if((len(set([grid[a,b] for a,b in option])) == 2 and not 0 in set([grid[a,b] for a,b in option])) or (len(set([grid[a,b] for a,b in option])) == 3)): blocked_options+=1
    if(blocked_options == len(options)): return 0 
    return -1

def do_step(state: State, action):
    old_state_index = get_state_index(state)
    new_state = copy.deepcopy(state)
    if(not old_state_index): 
        states.append(new_state)
        old_state_index = get_state_index(state)
    if(not old_state_index in probs):
        probs[old_state_index] = np.full(len(actions),1/len(actions), dtype=float)
        q[old_state_index] = np.full(len(actions),0, dtype=float)
    if(new_state.grid[action[0],action[1]] != 0):
        r[tuple([old_state_index,actions.tolist().index(action.tolist())])] = -10
        return new_state
    new_state.grid[action[0],action[1]] = 1 if new_state.o_turn else 2
    new_state.o_turn = not new_state.o_turn
    new_state.winner = check_winner(new_state.grid)
    new_state_index = get_state_index(new_state)
    if(not new_state_index): 
        states.append(new_state)
        new_state_index = get_state_index(new_state)
    if(not new_state_index in probs):
        probs[new_state_index] = np.full(len(actions),1/len(actions))
        q[new_state_index] = np.full(len(actions),0, dtype=float)
    return new_state

def do_episode(draw = False):
    current_state = copy.deepcopy(starting_state)
    state_index = get_state_index(current_state)
    chosen_action = actions[np.random.choice(len(actions), p=probs[state_index])]
    sa = [[current_state, chosen_action]]
    winner = -1
    while(winner == -1):
        draw_state(current_state)
        print(f"q: {q[state_index]}")
        print(f"probs: {probs[state_index]}")
        chosen_action = actions[np.random.choice(len(actions), p=probs[state_index])]
        print(f"chosen action: {chosen_action}")
        action_index = actions.tolist().index(list(chosen_action.tolist()))
        current_state = do_step(current_state, chosen_action)
        print(f"reward for action: {'None' if (state_index, action_index) not in r else r[(state_index, action_index)]}")
        print(f"action index: {action_index}")
        # draw_state(current_state)
        sa.append([current_state, chosen_action])
        state_index = get_state_index(current_state)
        # print(f"q: {q[state_index]}")
        # print(f"probs: {probs[state_index]}")
        winner = current_state.winner
    for state, action in sa:
        state_index = get_state_index(state)
        action_index = actions.tolist().index(action.tolist())
        if(draw): draw_state(state)
        print(f"winner: {state.winner}")
        reward = 1 if (winner == 1 and state.o_turn == False) or (winner == 2 and state.o_turn == True) else -1 if (winner == 1 and state.o_turn == True) or (winner == 2 and state.o_turn == False) else 0
        if((state_index, action_index) in r): reward = r[(state_index, action_index)]
        print(f"reward: {reward}")
        q[state_index][action_index] += 0.03 * (reward - q[state_index][action_index])
        # maximizing_index = np.argmax(q[state_index],0)
        print(f"q[state_index]: {q[state_index]}")
        maximizing_indices = np.argwhere(q[state_index] == np.amax(q[state_index])).flatten()
        print(f"maximizing indices: {maximizing_indices}")
        # print(f"q: {q[state_index]}")
        # print(f"maximizing: {maximizing_index}")
        # if(not len(set(q[state_index])) == 1):
        probs[state_index] = np.full(len(actions), 0, dtype=float)
        probs[state_index][np.array(maximizing_indices)] = 1/len(maximizing_indices)
        print(f"YEE: {probs[state_index]}")
        # print(f"eee: {probs}")

# action = actions[5]
# print(action)
# print("INDEX ", actions.tolist().index(action.tolist()))

for i in range(100): 
    print(i)
    do_episode(draw=True)
print(f"r: {r}")