from fastapi import FastAPI
from pydantic import BaseModel, Field, HttpUrl
from liba import *
import json
import tictac
from numpyencoder import NumpyEncoder

class State(BaseModel):
    x: int
    y: int

class ValueIterationItem(BaseModel):
    states: list[State] = []
    terminal_states: list[State] = []
    rewards: list[float] = []
    actions: list[str] = []
    probs: list[list[float]] = []
    transitional_probs: list[list[list[float]]] = []
    v: list[float] = []
    theta: float | None = 0.01
    gamma: float | None = 0.9

    class Config:
        arbitrary_types_allowed = True

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World", "specialTest" : "he"}


@app.post("/items/")
async def create_item(item: ValueIterationItem):
    states = np.array(item.states)
    terminal_states = np.array(item.terminal_states)
    rewards = np.array(item.rewards)
    actions = np.array(item.actions)
    probs = np.array(item.probs)
    transitional_probs = np.array(item.transitional_probs)
    v = np.array(item.v)
    theta = item.theta
    gamma = item.gamma
    iterate_policy(states, terminal_states, rewards, actions, probs, transitional_probs, v, theta, gamma)
    return {"v": json.dumps(v, cls=NumpyEncoder),"probs": json.dumps(probs, cls=NumpyEncoder)}

states_return, q_return, probs_return = tictac.load_data()
deterministic_policy = tictac.get_deterministic_policy(q_return)

@app.post("/tictac")
async def play(fields : str):
    state = tictac.State([[int(fields[0]),int(fields[1]),int(fields[2])],[int(fields[3]),int(fields[4]),int(fields[5])],[int(fields[6]),int(fields[7]),int(fields[8])]])
    winner = tictac.check_winner(state)
    if(winner == -1):
        chosen_action = tictac.get_appropriate_action(state, deterministic_policy, states_return)
        state = tictac.do_step(state, chosen_action)
        winner = tictac.check_winner(state)
    grid = state.grid.flatten()
    grid_string = f"{grid[0]}{grid[1]}{grid[2]}{grid[3]}{grid[4]}{grid[5]}{grid[6]}{grid[7]}{grid[8]}"
    return {
        "grid" : grid_string,
        "winner" : str(winner)
    }